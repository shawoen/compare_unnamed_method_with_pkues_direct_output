import numpy as np
import pandas as pd
from munch import Munch
import scipy.constants as const


def cal_ecr_and_pdr(j_wt, e_wt, psd_b_dict, psd_e_dict):
    """calculate the energy conversion rate and the pseudo-damping rate (Eqs. 2&3 in He et al. 2019)"""
    '''unit: j-muA/m^2, e-mV/m, b-nT'''
    '''ecr unit: J·s^{-1}·m^{-3}·Hz^{-1}'''
    '''pdr unit: s^{-1}'''
    dt = (j_wt.epoch[1] - j_wt.epoch[0]).total_seconds()

    '''psd for e and b energy in si unit'''
    psd_b_x_si_unit = psd_b_dict.x_psd_arr * 1.e-9 ** 2 / const.mu_0 / 2
    psd_b_y_si_unit = psd_b_dict.y_psd_arr * 1.e-9 ** 2 / const.mu_0 / 2
    psd_b_z_si_unit = psd_b_dict.z_psd_arr * 1.e-9 ** 2 / const.mu_0 / 2
    psd_b_para_si_unit = psd_b_dict.para_psd_arr * 1.e-9 ** 2 / const.mu_0 / 2
    psd_b_perp_si_unit = psd_b_dict.perp_psd_arr * 1.e-9 ** 2 / const.mu_0 / 2
    psd_b_trace_si_unit = psd_b_dict.trace_psd_arr * 1.e-9 ** 2 / const.mu_0 / 2
    psd_e_x_si_unit = psd_e_dict.x_psd_arr * 1.e-3 * const.epsilon_0 / 2
    psd_e_y_si_unit = psd_e_dict.y_psd_arr * 1.e-3 * const.epsilon_0 / 2
    psd_e_z_si_unit = psd_e_dict.z_psd_arr * 1.e-3 * const.epsilon_0 / 2
    psd_e_para_si_unit = psd_e_dict.para_psd_arr * 1.e-3 * const.epsilon_0 / 2
    psd_e_perp_si_unit = psd_e_dict.perp_psd_arr * 1.e-3 * const.epsilon_0 / 2
    psd_e_trace_si_unit = psd_e_dict.trace_psd_arr * 1.e-3 * const.epsilon_0 / 2

    '''energy conversion rate (unit: J·s^{-1}·m^{-3}·Hz^{-1})'''
    ecr_x = np.real(j_wt.x_wt * np.conj(e_wt.x_wt) + np.conj(j_wt.x_wt) * e_wt.x_wt) * dt * 1.e-9 / 4
    ecr_y = np.real(j_wt.y_wt * np.conj(e_wt.y_wt) + np.conj(j_wt.y_wt) * e_wt.y_wt) * dt * 1.e-9 / 4
    ecr_z = np.real(j_wt.z_wt * np.conj(e_wt.z_wt) + np.conj(j_wt.z_wt) * e_wt.z_wt) * dt * 1.e-9 / 4
    ecr_para = np.real(j_wt.para_wt * np.conj(e_wt.para_wt) + np.conj(j_wt.para_wt) * e_wt.para_wt) * dt * 1.e-9 / 4
    ecr_trace = ecr_x + ecr_y + ecr_z
    ecr_perp = ecr_trace - ecr_para

    '''pseudo-damping rate (unit: s^{-1})'''
    pdr_x = -ecr_x / (psd_b_x_si_unit + psd_e_x_si_unit) / 2
    pdr_y = -ecr_y / (psd_b_y_si_unit + psd_e_y_si_unit) / 2
    pdr_z = -ecr_z / (psd_b_z_si_unit + psd_e_z_si_unit) / 2
    pdr_para = -ecr_para / (psd_b_para_si_unit + psd_e_para_si_unit) / 2
    pdr_perp = -ecr_perp / (psd_b_perp_si_unit + psd_e_perp_si_unit) / 2
    pdr_trace = -ecr_trace / (psd_b_trace_si_unit + psd_e_trace_si_unit) / 2

    '''calculate 1d spectrum'''
    pdr_x_1d = - np.mean(ecr_x, axis=1) / (np.mean(psd_b_x_si_unit + psd_e_x_si_unit, axis=1)) / 2
    pdr_y_1d = - np.mean(ecr_y, axis=1) / (np.mean(psd_b_y_si_unit + psd_e_y_si_unit, axis=1)) / 2
    pdr_z_1d = - np.mean(ecr_z, axis=1) / (np.mean(psd_b_z_si_unit + psd_e_z_si_unit, axis=1)) / 2
    pdr_trace_1d = - np.mean(ecr_trace, axis=1) / (np.mean(psd_b_trace_si_unit + psd_e_trace_si_unit, axis=1)) / 2
    pdr_para_1d = - np.mean(ecr_para, axis=1) / (np.mean(psd_b_para_si_unit + psd_e_para_si_unit, axis=1)) / 2
    pdr_perp_1d = - np.mean(ecr_perp, axis=1) / (np.mean(psd_b_perp_si_unit + psd_e_perp_si_unit, axis=1)) / 2

    '''save 2d results into a dictionary'''
    ecr_pdr_dict = Munch()
    ecr_pdr_dict.epoch = j_wt.epoch
    ecr_pdr_dict.period = j_wt.period
    ecr_pdr_dict.x_ecr = ecr_x
    ecr_pdr_dict.y_ecr = ecr_y
    ecr_pdr_dict.z_ecr = ecr_z
    ecr_pdr_dict.para_ecr = ecr_para
    ecr_pdr_dict.perp_ecr = ecr_perp
    ecr_pdr_dict.trace_ecr = ecr_trace
    ecr_pdr_dict.x_pdr = pdr_x
    ecr_pdr_dict.y_pdr = pdr_y
    ecr_pdr_dict.z_pdr = pdr_z
    ecr_pdr_dict.para_pdr = pdr_para
    ecr_pdr_dict.perp_pdr = pdr_perp
    ecr_pdr_dict.trace_pdr = pdr_trace

    '''save 1d results as a dataframe'''
    ecr_pdr_df = pd.DataFrame(index=j_wt.period)
    ecr_pdr_df["pdr_x"] = pdr_x_1d
    ecr_pdr_df["pdr_y"] = pdr_y_1d
    ecr_pdr_df["pdr_z"] = pdr_z_1d
    ecr_pdr_df["pdr_trace"] = pdr_trace_1d
    ecr_pdr_df["pdr_para"] = pdr_para_1d
    ecr_pdr_df["pdr_perp"] = pdr_perp_1d

    return ecr_pdr_dict, ecr_pdr_df
