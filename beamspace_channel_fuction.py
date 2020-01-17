"""
Copyright(C) 2019-2020 Haozhi Wang
All rights reserved

Filename: Beamspace_channel_function
Description: Realization for the beamspace channel with lens array.

Created by Haozhi Wang at 17/01/2020
"""

import numpy as np
import math
from spatial_channel_fuction import spatial_channel
pi = math.pi


def beamspace_channel(N, K, L, lamaba):
    """
    fuction for beamspace channel with lens array.

    N: the number of the base station antennas
    K: the number of the user equipments
    L: the number of multipath
    lamada: the wavelength of carrier
    return: beamspace channel matrix
    """

    # get the spatial domain channel
    channel_matrix = spatial_channel(N=N, K=K, L=L, lamaba=lamaba)

    # the beamspace channel is transformed from the spatial domain channel using a lens antenna array.
    # the lens antenna array plays the role of a spatial discrete fourier transform (DFT) matrix
    Nidex = np.arange(-(N - 1), N, 2) / 2

    # lens array response
    U = np.zeros(N, N)
    for i in Nidex:
        U[:, i + (N + 1) / 2] = np.sqrt(1 / N) * np.exp(1j * 2 * pi * (1 / N) * i)

    # beamspace channel
    H_beam = np.matmul(U.transpose(), channel_matrix)

    return  H_beam