import torch
from torch import nn
from torch.distributions import Normal
from einops import rearrange

from ..layers.spatial_conv import SpatialConvOrderK
from ..layers.spatial_attention import SpatialAttention
from ..utils.ops import reverse_tensor
from ...utils.parser_utils import str_to_bool


class Nonlinear(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Nonlinear, self).__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

        self.act = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = torch.nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = torch.nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x


class TimeEncode(nn.Module):
    def __init__(self, expand_dim, factor=5):
        super(TimeEncode, self).__init__()

        self.time_dim = expand_dim
        self.factor = factor
        self.basis_freq = nn.Parameter((1 / 10 ** torch.linspace(0, 9, self.time_dim)).float())
        self.phase = nn.Parameter(torch.zeros(self.time_dim).float())

    def forward(self, ts):
        # ts: [N, L]
        batch_size = ts.size(0)
        seq_len = ts.size(1)

        ts = ts.view(batch_size, seq_len, 1)  # [N, L, 1]
        map_ts = ts * self.basis_freq.view(1, 1, -1)  # [N, L, time_dim]
        map_ts += self.phase.view(1, 1, -1)

        map_ts[:, :, 0::2] = torch.cos(map_ts[:, :, 0::2].clone())
        map_ts[:, :, 1::2] = torch.sin(map_ts[:, :, 1::2].clone())

        return map_ts


class AttnDecoder(nn.Module):
    def __init__(self, d_in, d_model, d_out, support_len, order=1, nheads=2, dropout=0., use_h=False):
        super(AttnDecoder, self).__init__()
        self.order = order
        self.use_h = use_h
        self.lin_in = nn.Linear(d_in, d_model)
        self.spatial_att = SpatialAttention(d_in=d_model,
                                            d_model=d_model,
                                            nheads=nheads,
                                            dropout=dropout)
        if use_h:
            self.lin_out = nn.Linear(4 * d_model, d_model)
            self.read_out = nn.Linear(3 * d_model, d_out)
        else:
            self.lin_out = nn.Linear(2 * d_model, d_model)
            self.read_out = nn.Linear(d_model, d_out)
        self.activation = nn.PReLU()

    def forward(self, x, h_node, h):
        x_in = [x, h_node, h]
        x = torch.cat(x_in, -1)
        x_in = self.lin_in(x)
        if len(x.shape) == 3:
            x_in = x_in.unsqueeze(1)
            h_node = h_node.unsqueeze(1)
            h = h.unsqueeze(1)
        out_att = self.spatial_att(x_in, torch.eye(x_in.size(2), dtype=torch.bool, device=x_in.device))
        out = torch.cat([x_in, out_att], -1)
        out = torch.cat([out, h_node, h], -1)
        out = self.activation(self.lin_out(out))
        out = torch.cat([out, h_node, h], -1)
        return self.read_out(out).squeeze(1), out.squeeze(1)


class VaeBlock(nn.Module):
    def __init__(self,
                input_size,
                hidden_size,
                anchor_size) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.anchor_size = int(anchor_size)
        self.z_size = int(self.hidden_size // 4)
        rnn_input_size = 2 * self.input_size

        self.timeencode = TimeEncode(self.hidden_size)

        self.encode = nn.GRU(rnn_input_size + self.anchor_size, self.hidden_size, num_layers=2, batch_first=True)
        self.mean = nn.Linear(self.hidden_size * 2, self.z_size)
        self.var = nn.Linear(self.hidden_size * 2, self.z_size)

        self.node_decode = nn.Linear(rnn_input_size + self.anchor_size, self.hidden_size)
        self.node_merge = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.edge_linear = Nonlinear(self.hidden_size * 3, self.hidden_size, 1)
        # self.node_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.graph_conv_1 = SpatialConvOrderK(c_in=self.hidden_size, c_out=self.hidden_size,
                                            support_len=1, order=1, include_self=False)
        self.graph_conv_2 = SpatialConvOrderK(c_in=self.hidden_size, c_out=self.hidden_size,
                                            support_len=1, order=1, include_self=False)

        self.h_decode = nn.Linear(self.hidden_size, self.input_size)
        self.decay = nn.Linear(self.input_size + 1, self.hidden_size)
        self.gru = nn.GRUCell(rnn_input_size + self.z_size +  self.hidden_size, self.hidden_size)
        self.decode = AttnDecoder(d_in=rnn_input_size + self.z_size + self.hidden_size + self.hidden_size,
                                              d_model=self.hidden_size,
                                              d_out=self.input_size,
                                              support_len=2,
                                              use_h=True)

    def init_hidden_states(self, n_nodes):
        std = 1. / torch.sqrt(torch.tensor(self.hidden_size, dtype=torch.float))
        vals = torch.distributions.Normal(0, std).sample((n_nodes, self.hidden_size))
        return nn.Parameter(vals)

    def get_h0(self, x):
        h0 = self.init_hidden_states(x.shape[2])
        return h0.expand(x.shape[0], -1, -1)
    
    def forward(self, x, mask=None, adjs=None, pos=None, timestamp=None, h=None):

        # get anchors
        anchors = pos

         # Get shape
        bs, steps, n_node, f = x.size()

        # infer all valid if mask is None
        if mask is None:
            mask = torch.ones_like(x, dtype=torch.uint8).to(x.device)
        
        # init hidden state using node embedding or the empty state
        if h is None:
            h = self.get_h0(x).to(x.device)
        h_in = h.reshape(bs * n_node, -1)

        # init output adjs
        adjs_pred = adjs.detach().clone()

        # output list
        imputations, h_pred, representations = [], [], []

        # time encode
        time = self.timeencode(timestamp)

        # vae encode
        x_in = torch.cat([x, anchors, mask], -1)
        x_in = rearrange(x_in, 'b s n f -> (b n) s f')
        z = self.encode(x_in)[-1]
        z = z.transpose_(0, 1).contiguous()
        z = z.view(z.shape[0], -1)
        z = rearrange(z, '(b n) f -> b n f', b=bs, n=n_node)
        z_mean = self.mean(z)
        z_var = self.var(z).exp_()
        dist = Normal(z_mean, z_var)
        if self.training:
            z = dist.rsample()
        else:
            z = z_mean

        # vae decode
        for step in range(steps):
            x_s = x[:, step, :, :]
            ms = mask[:, step, :, :]
            t = time[:, step, :]

            # vae decode 1
            x_1 = self.h_decode(h)
            x_s = torch.where(ms, x_s, x_1)

            # link prediction
            h_u = self.node_decode(torch.cat([x_s, ms, anchors[:, step, :, :]], -1))
            h_u = self.node_merge(torch.cat([h_u, h], -1))
            masked_nodes = torch.nonzero(ms, as_tuple=True)

            h_u_prime = h_u[masked_nodes[0], :, :]
            h_v_prime = h[masked_nodes[0], masked_nodes[1], :].unsqueeze(1).expand(len(masked_nodes[0]), n_node, -1)
            t = t[masked_nodes[0], :].unsqueeze(1).expand(len(masked_nodes[0]), n_node, -1)
            e = torch.sigmoid(self.edge_linear(torch.cat([h_u_prime, h_v_prime, t], -1))).squeeze(-1)
            adjs_pred[masked_nodes[0], step, masked_nodes[1], :] = e
            adjs_pred[masked_nodes[0], step, :, masked_nodes[1]] = e
            
            adj = adjs_pred[:, step, :, :].clone()
            fwd_adj = adj / (adj.sum(-1, keepdims=True) + 1e-12)
            bwd_adj = adj.transpose(-1, -2) / (adj.transpose(-1, -2).sum(-1, keepdims=True) + 1e-12)
            adj = (fwd_adj + bwd_adj) / 2
            h_out_1 = self.graph_conv_1(h_u.transpose(-1, -2), adj)
            h_out_2 = self.graph_conv_2(h_out_1, adj)
            h_out = h_out_1 + h_out_2
            h_out = h_out.transpose(-1, -2)

            # vae decode 2
            x_2, x_out = self.decode(torch.cat([z, x_s, ms], -1), h_out, h)
            x_s = torch.where(ms, x_s, x_2)

            # update gru decoder state
            x_in = torch.cat([z.reshape(bs * n_node, -1), x_s.reshape(bs * n_node, -1), ms.reshape(bs * n_node, -1), h_out.reshape(bs * n_node, -1)], dim=-1)
            h_in = self.gru(x_in, h_in)
            h = h_in.reshape(bs, n_node, -1)

            h_pred.append(x_1)
            imputations.append(x_2)
            representations.append(x_out)

        h_pred = torch.stack(h_pred, dim=1)
        imputations = torch.stack(imputations, dim=1)
        representations = torch.stack(representations, dim=1)

        return imputations, h_pred, representations, dist, adjs_pred


class PoGeVon(nn.Module):
    def __init__(self,
                 adj,
                 pos,
                 anchors,
                 d_in,
                 d_hidden,
                 d_ff,
                 ff_dropout,
                 d_u=0,
                 d_emb=0,
                 impute_only_holes=True):
        super(PoGeVon, self).__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.d_u = int(d_u) if d_u is not None else 0
        self.d_emb = int(d_emb) if d_emb is not None else 0
        self.register_buffer('adj', torch.tensor(adj).float())
        self.register_buffer('pos', torch.tensor(pos).float())
        self.register_buffer('anchors', torch.tensor(anchors).long())
        self.impute_only_holes = impute_only_holes

        self.seq1 = VaeBlock(d_in, d_hidden, self.anchors.shape[-1])
        self.seq2 = VaeBlock(d_in, d_hidden, self.anchors.shape[-1])

        self.out1 = nn.Linear(d_hidden, d_in)
        self.out2 = nn.Linear(d_hidden, d_in)

        self.emb = nn.Parameter(torch.empty(self.adj.shape[0], d_emb))
        nn.init.kaiming_normal_(self.emb, nonlinearity='relu')
        self.out = nn.Sequential(
            nn.Linear(6 * d_hidden + d_emb, d_ff),
            nn.ReLU(),
            nn.Dropout(ff_dropout),
            nn.Linear(d_ff, d_in)
        )

    def forward(self, x, mask=None, adj=None, pos=None, timestamp=None, *kwargs):

        # merge, x: [batches, steps, nodes, channels]
        b, s, n, f = x.shape
        # if not self.training:
        #     pos = self.pos.expand((x.shape[0], x.shape[1], self.pos.shape[0], self.pos.shape[1]))

        # seq1
        fwd_impute, fwd_hpred, fwd_rep, fwd_dist, fwd_adjs_pred = self.seq1(x, mask, adj, pos, timestamp)

        # seq2
        rev_x = reverse_tensor(x, axis=1)
        rev_m = reverse_tensor(mask, axis=1)
        rev_a = reverse_tensor(adj, axis=1)
        rev_p = reverse_tensor(pos, axis=1)
        rev_t = reverse_tensor(timestamp, axis=1)
        # rev_p = reverse_tensor(pos, axis=1) if self.training else pos
        bwd_impute, bwd_hpred, bwd_rep, bwd_dist, bwd_adjs_pred = self.seq2(rev_x, rev_m, rev_a, rev_p, rev_t)
        bwd_impute = reverse_tensor(bwd_impute, axis=1)
        bwd_hpred = reverse_tensor(bwd_hpred, axis=1)
        bwd_rep = reverse_tensor(bwd_rep, axis=1)
        bwd_adjs_pred = reverse_tensor(bwd_adjs_pred, axis=1)

        # imputation: [batches, steps, nodes, channels]
        inputs = [fwd_rep, bwd_rep, self.emb.view(1, 1, *self.emb.shape).expand(b, s, -1, -1)]
        imputation = torch.cat(inputs, dim=-1)
        imputation = self.out(imputation)
        if self.impute_only_holes and not self.training:
            imputation = torch.where(mask, x, imputation)

        prediction = torch.stack([fwd_impute, bwd_impute, fwd_hpred, bwd_hpred], dim=0)
        
        if self.training:
            return imputation, prediction, [fwd_dist, bwd_dist], [fwd_adjs_pred, bwd_adjs_pred]
        return imputation, prediction, [fwd_dist, bwd_dist], [fwd_adjs_pred, bwd_adjs_pred]

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument('--d-hidden', type=int, default=64)
        parser.add_argument('--d-ff', type=int, default=64)
        parser.add_argument('--ff-dropout', type=int, default=0.)
        parser.add_argument('--d-emb', type=int, default=8)
        parser.add_argument('--impute-only-holes', type=str_to_bool, nargs='?', const=True, default=True)
        return parser
