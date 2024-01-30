import numpy as np
import pandas as pd
import scipy.constants as const
from munch import Munch


def cal_k_bellan(b_wt, j_wt):
    """Eq. (8) of Bellan et al. (2016)"""
    num_t = b_wt.epoch.size
    num_p = b_wt.period
    b_arr = np.zeros((3, num_p, num_t), dtype=complex)  # unit: nT
    b_arr[0, ...] = b_wt.x_wt
    b_arr[1, ...] = b_wt.y_wt
    b_arr[2, ...] = b_wt.z_wt
    j_arr = np.zeros((3, num_p, num_t), dtype=complex)  # unit: muA
    j_arr[0, ...] = j_wt.x_wt
    j_arr[1, ...] = j_wt.y_wt
    j_arr[2, ...] = j_wt.z_wt
    numerator = np.cross(j_arr * np.conj(b_arr), axis=0)
    denominator = np.tile(np.sum(b_arr * np.conj(b_arr), axis=0)[None, ...], (3, 1, 1))
    k_km_arr = np.real(1j * const.mu_0 * numerator / denominator) * 1.e6

    return k_km_arr


def cal_j_invert_bellan(k_arr, b_wt):
    """Eq. (6) of Bellan et al. (2016)"""
    '''unit: k: 1/km, b: nT'''
    num_t = b_wt.epoch.size
    num_p = b_wt.period.size
    b_arr = np.zeros((3, num_p, num_t), dtype=complex)  # unit: nT
    b_arr[0, ...] = b_wt.x_wt
    b_arr[1, ...] = b_wt.y_wt
    b_arr[2, ...] = b_wt.z_wt
    j_arr = 1j * np.cross(k_arr, b_arr, axis=0) / const.mu_0 * 1.e-6  # unit: muA  #(modified by XieHaoen on 26/1/2024)
    '''save as a dataframe'''
    j_wt = Munch()
    j_wt.epoch = b_wt.epoch
    j_wt.period = b_wt.period
    j_wt.x_wt = j_arr[0, ...]
    j_wt.y_wt = j_arr[1, ...]
    j_wt.z_wt = j_arr[2, ...]

    return j_wt
