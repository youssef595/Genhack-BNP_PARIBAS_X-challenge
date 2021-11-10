
import numpy as np


def Pseudo_observations(data):
    n, d = data.shape
    arr = np.zeros(n)

    for i in range(n):
        Z_i = 0

        for j in range(n):
            if i != j:
                Z_i_k = 1

                for k in range(d):
                    Z_i_k *= (data[i][k] > data[j][k])

                Z_i += Z_i_k

        Z_i /= (n - 1)
        arr[i] = Z_i

    return arr


def Absolute_Kendall_error(validation_data, synthetic_data):

    val = Pseudo_observations(validation_data)
    syn = Pseudo_observations(synthetic_data)

    val_sorted = np.sort(val)
    syn_sorted = np.sort(syn)

    # n = len(val_sorted)
    loss = np.mean(abs(val_sorted - syn_sorted))

    return(loss)
