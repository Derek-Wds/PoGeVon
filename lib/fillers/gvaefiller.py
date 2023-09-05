import torch
from torch.distributions import Normal, kl_divergence
from pytorch_lightning.utilities import move_data_to_device

from . import Filler
from ..nn.models import PoGeVon


class GraphVAEFiller(Filler):

    def __init__(self,
                 model_class,
                 model_kwargs,
                 optim_class,
                 optim_kwargs,
                 loss_fn,
                 scaled_target=False,
                 whiten_prob=0.05,
                 pred_loss_weight=1.,
                 warm_up=0,
                 metrics=None,
                 scheduler_class=None,
                 scheduler_kwargs=None):
        super(GraphVAEFiller, self).__init__(model_class=model_class,
                                          model_kwargs=model_kwargs,
                                          optim_class=optim_class,
                                          optim_kwargs=optim_kwargs,
                                          loss_fn=loss_fn,
                                          scaled_target=scaled_target,
                                          whiten_prob=whiten_prob,
                                          metrics=metrics,
                                          scheduler_class=scheduler_class,
                                          scheduler_kwargs=scheduler_kwargs)

        self.tradeoff = pred_loss_weight
        if model_class in [PoGeVon]:
            self.trimming = (warm_up, warm_up)

    def trim_seq(self, *seq):
        seq = [s[:, self.trimming[0]:s.size(1) - self.trimming[1]] for s in seq]
        if len(seq) == 1:
            return seq[0]
        return seq

    def predict_batch(self, batch, preprocess=False, postprocess=True, return_target=False):
        """
        This method takes as an input a batch as a two dictionaries containing tensors and outputs the predictions.
        Prediction should have a shape [batch, nodes, horizon]

        :param batch: list dictionary following the structure [data:
                                                                {'x':[...], 'y':[...], 'u':[...], ...},
                                                              preprocessing:
                                                                {'bias': ..., 'scale': ..., 'x_trend':[...], 'y_trend':[...]}]
        :param preprocess: whether the data need to be preprocessed (note that inputs are by default preprocessed before creating the batch)
        :param postprocess: whether to postprocess the predictions (if True we assume that the model has learned to predict the trasformed signal)
        :param return_target: whether to return the prediction target y_true and the prediction mask
        :return: (y_true), y_hat, (mask)
        """
        batch_data, batch_preprocessing = self._unpack_batch(batch)
        if preprocess:
            x = batch_data.pop('x')
            x = self._preprocess(x, batch_preprocessing)
            y_hat = self.forward(x, **batch_data)
        else:
            y_hat = self.forward(**batch_data)
        y_hat = list(y_hat)
        # Rescale outputs
        if postprocess:
            y_hat[0] = self._postprocess(y_hat[0], batch_preprocessing)
        if return_target:
            y = batch_data.get('y')
            mask = batch_data.get('mask', None)
            return y, y_hat, mask
        return y_hat

    def predict_loader(self, loader, preprocess=False, postprocess=True, return_mask=True):
        """
        Makes predictions for an input dataloader. Returns both the predictions and the predictions targets.

        :param loader: torch dataloader
        :param preprocess: whether to preprocess the data
        :param postprocess: whether to postprocess the data
        :param return_mask: whether to return the valid mask (if it exists)
        :return: y_true, y_hat
        """
        targets, imputations, masks = [], [], []
        graph_loss = []
        for batch in loader:
            batch = move_data_to_device(batch, self.device)
            batch_data, batch_preprocessing = self._unpack_batch(batch)
            # Extract mask and target
            eval_mask = batch_data.pop('eval_mask', None)
            y = batch_data.pop('y')
            if 'adj_label' in batch_data:
                adj_label = batch_data.pop('adj_label')

            y_hat = self.predict_batch(batch, preprocess=preprocess, postprocess=postprocess)

            if isinstance(y_hat, (list, tuple)):
                adjs_pred = y_hat[-1]
                y_hat = y_hat[0]

            # adjacency loss
            masked_nodes = torch.nonzero(eval_mask, as_tuple=True)
            adj_loss_1 = torch.norm(adjs_pred[0][masked_nodes[0], masked_nodes[1], masked_nodes[2], :] - 
                                adj_label[masked_nodes[0], masked_nodes[1], masked_nodes[2], :], p='fro')
            adj_loss_2 = torch.norm(adjs_pred[1][masked_nodes[0], masked_nodes[1], masked_nodes[2], :] -
                                adj_label[masked_nodes[0], masked_nodes[1], masked_nodes[2], :], p='fro')
            graph_loss.append((adj_loss_1+adj_loss_2).detach().cpu())

            targets.append(y)
            imputations.append(y_hat)
            masks.append(eval_mask)

        y = torch.cat(targets, 0)
        y_hat = torch.cat(imputations, 0)
        graph_loss = torch.mean(torch.stack(graph_loss))
        print('---------------------------')
        print(f'Graph Loss: {graph_loss}.')
        print('---------------------------')
        if return_mask:
            mask = torch.cat(masks, 0) if masks[0] is not None else None
            return y, y_hat, mask
        return y, y_hat

    def training_step(self, batch, batch_idx):
        # Unpack batch
        batch_data, batch_preprocessing = self._unpack_batch(batch)

        # Compute masks
        mask = batch_data['mask'].clone().detach()
        mask_ = torch.ones_like(mask)
        batch_data['mask'] = torch.bernoulli(mask.clone().detach().float() * self.keep_prob).byte()
        eval_mask = batch_data.pop('eval_mask', None)
        eval_mask = (mask | eval_mask) - batch_data['mask']  # all unseen data

        y = batch_data.pop('y')
        adj_y = batch_data.pop('adj_label')

        # Compute predictions and compute loss
        res = self.predict_batch(batch, preprocess=False, postprocess=False)
        imputation, predictions, dist, adjs_pred = (res[0], res[1], res[2], res[3]) if isinstance(res, (list, tuple)) else (res, [], None)

        # trim to imputation horizon len
        imputation, mask, eval_mask, y = self.trim_seq(imputation, mask, eval_mask, y)
        predictions = self.trim_seq(*predictions)

        if self.scaled_target:
            target = self._preprocess(y, batch_preprocessing)
        else:
            target = y
            imputation = self._postprocess(imputation, batch_preprocessing)
            for i, _ in enumerate(predictions):
                predictions[i] = self._postprocess(predictions[i], batch_preprocessing)

        # adjacency loss
        masked_nodes = torch.nonzero(mask, as_tuple=True)
        # masked_nodes = torch.nonzero(eval_mask, as_tuple=True)
        adj_loss_1 = torch.norm(adjs_pred[0][masked_nodes[0], masked_nodes[1], masked_nodes[2], :] - 
                            adj_y[masked_nodes[0], masked_nodes[1], masked_nodes[2], :], p='fro')
        adj_loss_2 = torch.norm(adjs_pred[1][masked_nodes[0], masked_nodes[1], masked_nodes[2], :] -
                            adj_y[masked_nodes[0], masked_nodes[1], masked_nodes[2], :], p='fro')

        loss = self.loss_fn(imputation, target, mask)
        loss += 0.01 * (adj_loss_1 + adj_loss_2)
        for pred in predictions:
            loss += self.tradeoff * self.loss_fn(pred, target, mask)
        if dist is not None:
            normal = Normal(torch.zeros(dist[0].mean.size()).to(target.device),
                        torch.ones(dist[0].stddev.size()).to(target.device))
            KLD1 = kl_divergence(dist[0], normal).mean()
            KLD2 = kl_divergence(dist[1], normal).mean()
            loss += 0.2 * (KLD1 + KLD2)

        # Logging
        if self.scaled_target:
            imputation = self._postprocess(imputation, batch_preprocessing)
        self.train_metrics.update(imputation.detach(), y, eval_mask)  # all unseen data
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('train_loss', loss.detach(), on_step=False, on_epoch=True, logger=True, prog_bar=False)
        self.log('graph_loss', (adj_loss_1 + adj_loss_2).detach(), on_step=False, on_epoch=True, logger=True, prog_bar=False)
        self.log('kld_loss', (KLD1 + KLD2).detach(), on_step=False, on_epoch=True, logger=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        # Unpack batch
        batch_data, batch_preprocessing = self._unpack_batch(batch)

        # Extract mask and target
        mask = batch_data.get('mask')
        eval_mask = batch_data.pop('eval_mask', None)
        y = batch_data.pop('y')
        adj_y = batch_data.pop('adj_label')

        # Compute predictions and compute loss
        res = self.predict_batch(batch, preprocess=False, postprocess=False)
        imputation, predictions, dist, adjs_pred = (res[0], res[1], res[2], res[3]) if isinstance(res, (list, tuple)) else (res, [], None)

        # trim to imputation horizon len
        imputation, mask, eval_mask, y = self.trim_seq(imputation, mask, eval_mask, y)

        if self.scaled_target:
            target = self._preprocess(y, batch_preprocessing)
        else:
            target = y
            imputation = self._postprocess(imputation, batch_preprocessing)

        val_loss = self.loss_fn(imputation, target, eval_mask)
        
        # adjacency loss
        masked_nodes = torch.nonzero(eval_mask, as_tuple=True)
        adj_loss_1 = torch.norm(adjs_pred[0][masked_nodes[0], masked_nodes[1], masked_nodes[2], :] - 
                            adj_y[masked_nodes[0], masked_nodes[1], masked_nodes[2], :], p='fro')
        adj_loss_2 = torch.norm(adjs_pred[1][masked_nodes[0], masked_nodes[1], masked_nodes[2], :] -
                            adj_y[masked_nodes[0], masked_nodes[1], masked_nodes[2], :], p='fro')
        adj_loss = adj_loss_1 + adj_loss_2

        # Logging
        if self.scaled_target:
            imputation = self._postprocess(imputation, batch_preprocessing)
        self.val_metrics.update(imputation.detach(), y, eval_mask)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('val_loss', val_loss.detach(), on_step=False, on_epoch=True, logger=True, prog_bar=False)
        self.log('graph_loss', adj_loss.detach(), on_step=False, on_epoch=True, logger=True, prog_bar=False)
        return val_loss

    def test_step(self, batch, batch_idx):
        # Unpack batch
        batch_data, batch_preprocessing = self._unpack_batch(batch)

        # Extract mask and target
        eval_mask = batch_data.pop('eval_mask', None)
        y = batch_data.pop('y')
        adj_y = batch_data.pop('adj_label')

        # Compute outputs and rescale
        res = self.predict_batch(batch, preprocess=False, postprocess=True)
        imputation, predictions, dist, adjs_pred = (res[0], res[1], res[2], res[3]) if isinstance(res, (list, tuple)) else (res, [], None)
        test_loss = self.loss_fn(imputation, y, eval_mask)
        
        # adjacency loss
        masked_nodes = torch.nonzero(eval_mask, as_tuple=True)
        adj_loss_1 = torch.norm(adjs_pred[0][masked_nodes[0], masked_nodes[1], masked_nodes[2], :] - 
                            adj_y[masked_nodes[0], masked_nodes[1], masked_nodes[2], :], p='fro')
        adj_loss_2 = torch.norm(adjs_pred[1][masked_nodes[0], masked_nodes[1], masked_nodes[2], :] -
                            adj_y[masked_nodes[0], masked_nodes[1], masked_nodes[2], :], p='fro')
        adj_loss = adj_loss_1 + adj_loss_2

        # Logging
        self.test_metrics.update(imputation.detach(), y, eval_mask)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('test_loss', test_loss.detach(), on_step=False, on_epoch=True, logger=True, prog_bar=False)
        self.log('graph_loss', adj_loss.detach(), on_step=False, on_epoch=True, logger=True, prog_bar=False)
        return test_loss
