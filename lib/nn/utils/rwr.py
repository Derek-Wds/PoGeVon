import numpy as np
import torch
from numpy.linalg import norm
from scipy.sparse import spdiags
from tqdm import tqdm

def row_normalize(A):
    '''
    Perform row-normalization of the given matrix
    inputs
        A : crs_matrix
            (n x n) input matrix where n is # of nodes
    outputs
        nA : crs_matrix
             (n x n) row-normalized matrix
    '''
    n = A.shape[0]

    # do row-wise sum where d is out-degree for each node
    d = A.sum(axis=1)
    d = np.asarray(d).flatten()

    # handle 0 entries in d
    d = np.maximum(d, np.ones(n))
    invd = 1.0 / d

    invD = spdiags(invd, 0, n, n)

    # compute row normalized adjacency matrix by nA = invD * A
    nA = invD.dot(A)

    return nA

def rwr(A, q, c=0.15, epsilon=1e-9, max_iters=100, handles_deadend=True, norm_type=1):
    x = q
    old_x = q
    residuals = np.zeros(max_iters)

    pbar = tqdm(total=max_iters)
    for i in range(max_iters):
        if handles_deadend:
            x = (1 - c) * (A.dot(old_x))
            S = np.sum(x)
            x = x + (1 - S) * q
        else:
            x = (1 - c) * (A.dot(old_x)) + (c * q)

        residuals[i] = norm(x - old_x, norm_type)
        pbar.set_description("Residual at %d-iter: %e" % (i, residuals[i]))

        if residuals[i] <= epsilon:
            pbar.set_description("The iteration has converged at %d-iter" % (i))
            #  pbar.update(max_iters)
            break

        old_x = x
        pbar.update(1)

    pbar.close()

    return x, residuals[0:i + 1]

def get_position(A):
    n = len(A)
    A = row_normalize(A)
    seeds = np.random.choice(n, int(np.log2(n)))
    positions = list()
    for seed in seeds:
        q = np.zeros(n)
        q[seed] = 1.0
        x, _ = rwr(A.T, q)
        positions.append(x)
    return [np.stack(positions, axis=1), seeds]
    