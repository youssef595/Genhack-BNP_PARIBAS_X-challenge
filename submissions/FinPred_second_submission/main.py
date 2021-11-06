# imports
import numpy as np
import pandas as pd
import torch
from torch import nn

import os

os.chdir("./submissions/FinPred_second_submission")

print(os.getcwd())

# general params
NB_OBS = 410
NB_TICKERS = 2


# architecture generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 64),  # 64
            nn.Softplus(),
            nn.Linear(64, 128),  # 128
            nn.Softplus(),
            nn.Linear(128, 32),  # 32
            nn.Softplus(),
            nn.Linear(32, 2),
            nn.Softplus(),
        )

    def forward(self, x):
        output = self.model(x)
        return output


# some utils
# generator noramal noise
def gen_noise(nb_rows: int = NB_OBS, nb_cols: int = NB_TICKERS):
    # sample normal noise
    normal_samples = np.random.normal(size=(nb_rows, nb_cols))
    # convert to tensor
    return torch.Tensor(normal_samples)

# convert tensor to arr


def tensor_to_arr(ten):
    # dim of ten
    n, p = ten.shape

    rows = []
    # explode
    for col in range(p):
        rows.append([ele.item() for ele in ten[:, col]])

    return np.array(rows, dtype=float).T


# init model
generator = Generator()
# load params
generator.load_state_dict(torch.load("params.pt"))
generator.eval()


# generate stock 1, 2
noise_12 = gen_noise()
generated_12 = generator(noise_12)

# generate stock 3, 4
noise_34 = gen_noise()
generated_34 = generator(noise_34)


# arrange noise
noise_data = pd.DataFrame(
    np.concatenate(
        (tensor_to_arr(noise_12), tensor_to_arr(noise_34)),
        axis=1
    )
)

# arrange generated stocks
generated_data = pd.DataFrame(
    np.concatenate(
        (tensor_to_arr(generated_12), tensor_to_arr(generated_34)),
        axis=1
    )
)

# save
noise_data.to_csv("noise.csv", index=False, header=False)
generated_data.to_csv("simulated_data.csv", index=False, header=False)
