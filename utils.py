import csv
import os
from datetime import datetime

import numpy as np
import scipy.sparse as sp
import torch


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    Function that converts a Scipy sparse matrix to a sparse Torch tensor
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def normalize_adjacency(A):
    """
    Function that normalizes an adjacency matrix
    """
    n = A.shape[0]
    A = A + sp.identity(n)
    degs = A.dot(np.ones(n))
    inv_degs = np.power(degs, -1)
    D = sp.diags(inv_degs)
    A_normalized = D.dot(A)

    return A_normalized


def get_unique_file_path(folder, name, ext):
    path = os.path.join(folder, f'{name}_{datetime.now().strftime("%y-%m-%d_%H-%M-%S")}.{ext}')
    i = 1
    while os.path.exists(path):
        path = os.path.join(folder, f'{name}_{datetime.now().strftime("%y-%m-%d_%H-%M-%S")}_({i}).{ext}')
        i += 1
    return path