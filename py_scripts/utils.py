
import numpy as np
import torch


# constants
NB_OBSERVATION = 410
NB_TIKERS = 4


def tensor_to_arr(ten):
    # dim of ten
    n, p = ten.shape

    rows = []
    # explode
    for col in range(p):
        rows.append([ele.item() for ele in ten[:, col]])

    return np.array(rows, dtype=float).T


def gen_noise(nb_rows: int = 410, nb_cols: int = NB_TIKERS):

    normal_samples = np.random.normal(size=(nb_rows, nb_cols))
    return torch.Tensor(normal_samples)
