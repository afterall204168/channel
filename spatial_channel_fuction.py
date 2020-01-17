"""
Copyright(C) 2019-2020 Haozhi Wang
All rights reserved

Filename: spatial_channel_function
Description: Realization for the spatial domain channel.

Created by Haozhi Wang at 17/01/2020
"""

import numpy as np
import math
pi = math.pi


def spatial_channel(N, K, L, lamaba):
    """
    fuction for spatial domain channel.

    N: the number of the base station antennas
    K: the number of the user equipments
    L: the number of multipath
    lamaba: the wavelength of carrier
    return: the channel matrix
    """
    d = lamaba / 2  # the antenna spacing

    # the channel matrix
    channel = np.zeros(dtype=np.complex128, shape=(N, K))

    # the sub-channel matrix
    sub_channel = np.zeros(dtype=np.complex128, shape=(N, L))
    for k in range(K):
        for l in range(L):
            # the complex gain
            alpha = (np.random.normal() + 1j * np.random.normal()) / np.sqrt(2)

            # the spatial angle for uniform linear arrays(ULAs)
            theta = pi * np.random.uniform(0, 1) - pi / 2

            # array response
            # the array steering vector can be determined by one angle
            a_theta = (1 / np.sqrt(N)) * np.exp(1j * 2 * pi * np.arange(N).reshape(N, 1), *d * np.sin(theta) / lamaba)

            # the sub_channel
            sub_channel[..., l] = np.squeeze(alpha * a_theta)

        # the spatial domain channel matrix
        channel[..., k] = np.sqrt(N/(L)) * np.sum(sub_channel, axis=1)

    return channel