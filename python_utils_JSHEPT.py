import os
import stat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib
import datetime
from mpl_toolkits.mplot3d import Axes3D
import math
from scipy import signal
from scipy.io import readsav
import julian
from scipy.optimize import least_squares
import sys
from itertools import chain


# a from goto import with_goto
from wavelets import WaveletAnalysis
from wavelets import cwt, Morlet


def get_plot_WaveletAnalysis_of_var_vect(time_vect, var_vect, period_range=np.array([0., 1.]), num_periods=16):
    # get period_vect according to period_range and num_periods
    period_min = period_range[0]
    period_max = period_range[1]
    k0 = 6.0
    fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + k0 ** 2))
    scale_min = period_min / fourier_factor
    scale_max = period_max / fourier_factor
    J_wavlet = num_periods
    dj_wavlet = np.log10(scale_max / scale_min) / np.log10(2.) / (J_wavlet - 1)
    scale_vect = scale_min * 2. ** (np.linspace(0, J_wavlet - 1, J_wavlet) * dj_wavlet)
    period_vect = scale_vect * fourier_factor
    # call WaveletAnalysis in wavelets package to get wavelet_obj_arr and WaveletCoeff_var_arr
    dtime = np.diff(time_vect).mean()
    wavelet_obj_arr = WaveletAnalysis(var_vect, time=time_vect, dt=dtime)
    wavelet_obj_arr.fourier_periods = period_vect
    WaveletCoeff_var_arr = wavelet_obj_arr.wavelet_transform
    WaveletCoeff_var_arr = np.transpose(WaveletCoeff_var_arr)
    # get sub_wave_var_arr through decomposition/reconstruction
    num_times = len(time_vect)
    num_periods = len(period_vect)
    sub_wave_var_arr = np.zeros([num_times, num_periods])  # , dtype=float)
    for i_period in range(0, num_periods):
        period_tmp = period_vect[i_period:i_period + 1]
        # a scale_tmp = scale_vect[i_period:i_period+1]
        wavelet_obj_arr_tmp = wavelet_obj_arr
        wavelet_obj_arr_tmp.fourier_periods = period_tmp
        var_vect_tmp = wavelet_obj_arr_tmp.reconstruction()
        var_vect_tmp -= wavelet_obj_arr_tmp.data.mean(axis=wavelet_obj_arr_tmp.axis,
                                                      keepdims=True)  # subtract the mean back again, which was added in the
        sub_wave_var_arr[0:num_times, i_period] = np.real(var_vect_tmp)

    return time_vect, period_vect, wavelet_obj_arr, WaveletCoeff_var_arr, sub_wave_var_arr


def get_local_mean_variable(time_vect, var_vect, period_range=np.array([0., 1.]), \
                            width2period=10, num_periods=16):
    '''Equation 22 of podesta 2009'''
    '''get period_vect according to period_range and num_periods'''
    period_min = period_range[0]
    period_max = period_range[1]
    k0 = 6.0
    fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + k0 ** 2))
    scale_min = period_min / fourier_factor
    scale_max = period_max / fourier_factor
    J_wavlet = num_periods
    dj_wavlet = np.log10(scale_max / scale_min) / np.log10(2.) / (J_wavlet - 1)
    scale_vect = scale_min * 2. ** (np.linspace(0, J_wavlet - 1, J_wavlet) * dj_wavlet)
    period_vect = scale_vect * fourier_factor
    # a print('np.linspace(0,J_wavlet-1,J_wavlet): ',np.linspace(0,J_wavlet-1,J_wavlet))
    # a print('period_vect: ',period_vect)
    # get var_LocalBG_arr
    num_times = len(time_vect)
    var_LocalBG_arr = np.zeros([num_times, num_periods])
    for i_period in range(0, num_periods):
        ##set window for convolution
        period_tmp = period_vect[i_period:i_period + 1]
        duration_tmp = period_tmp * width2period
        sigma_tmp = duration_tmp / 6
        # a sigma_tmp = period_tmp * 3
        # a duration_tmp = (sigma_tmp * 6) * 2
        dtime = np.diff(time_vect).mean()
        # a print('i_period, num_periods: ', i_period, num_periods)
        # a print('dtime, period_tmp: ', dtime, period_tmp)
        num_times_for_window = int(duration_tmp / dtime / 2) * 2 + 1
        time_vect_for_window = np.linspace(-duration_tmp / 2, +duration_tmp / 2, num_times_for_window)
        window_vect = np.exp(-(time_vect_for_window) ** 2 / (2 * sigma_tmp ** 2)) / (np.sqrt(2 * np.pi) * sigma_tmp)
        window_vect = np.reshape(window_vect, len(window_vect))
        ##get var_LocalBG_vect through convolution between var_vect and window_vect
        # d var_LocalBG_vect = np.convolve(var_vect, window_vect, mode='same')*dtime
        var_LocalBG_vect = signal.convolve(var_vect, window_vect, mode='same') * dtime
        var_LocalBG_arr[:, i_period] = var_LocalBG_vect
    # a print('mean(var_vect), mean(var_LocalBG_vect): ', np.mean(var_vect), np.mean(var_LocalBG_vect))
    # a input('Press any key to continue...')
    return time_vect, period_vect, var_LocalBG_arr


def dE_from_FaradayLaw(omega2k_gamma2k, dBt_complex, dBn_complex):
    omega2k_complex = np.complex(omega2k_gamma2k[0], omega2k_gamma2k[1])
    dEn_complex = -dBt_complex * omega2k_complex
    dEt_complex = +dBn_complex * omega2k_complex
    return dEt_complex, dEn_complex


"""
def residual_between_dE_obs_and_dE_from_FaradayLaw(omega2k):
    dEt_complex_predict, dEn_complex_predict = dE_from_FaradayLaw(omega2k, dBt_complex, dBn_complex)
    residual_Re_En = np.real(dEn_complex_predict - dEn_complex_observe)
    residual_Im_En = np.imag(dEn_complex_predict - dEn_complex_observe)
    residual_Re_Et = np.real(dEt_complex_predict - dEt_complex_observe)
    residual_Im_Et = np.imag(dEt_complex_predict - dEt_complex_observe)
    residual_vect = [residual_Re_En, residual_Im_En, residual_Re_Et, residual_Im_Et]
    return residual_vect
"""


def wavelet_reconstruction(wavelet_obj_arr, WaveletCoeff_var_arr):
    """Reconstruct the original signal from the wavelet
    transform. See S3.i.

    For non-orthogonal wavelet functions, it is possible to
    reconstruct the original time series using an arbitrary
    wavelet function. The simplest is to use a delta function.

    The reconstructed time series is found as the sum of the
    real part of the wavelet transform over all scales,

    x_n = (dj * dt^(1/2)) / (C_d * Y_0(0)) \
            * Sum_(j=0)^J { Re(W_n(s_j)) / s_j^(1/2) }

    where the factor C_d comes from the reconstruction of a delta
    function from its wavelet transform using the wavelet
    function Y_0. This C_d is a constant for each wavelet
    function.
    This function is orginally inherited from the method function 'reconstruction',
    which is one of the method affiliated to the class of 'WaveletAnalysis' in 'transform.py'
    The reason why we modify the original function 'reconstruction' and put it here as a new function is:
    we have modified the result of 'self.wavelet_transform' and assigned it to 'WaveletCoeff_var_arr' as the input
    """
    period_vect = wavelet_obj_arr.fourier_periods
    num_periods = len(period_vect)
    num_times = len(wavelet_obj_arr.data)
    sub_wave_var_arr = np.zeros([num_times, num_periods])  # , dtype=float)
    for i_period in range(0, num_periods):
        period_tmp = period_vect[i_period:i_period + 1]
        # a print('i_period, period_tmp: ', i_period, period_tmp)
        wavelet_obj_arr_tmp = wavelet_obj_arr
        wavelet_obj_arr_tmp.fourier_periods = period_tmp
        dj = wavelet_obj_arr_tmp.dj
        dt = wavelet_obj_arr_tmp.dt
        C_d = wavelet_obj_arr_tmp.C_d
        Y_00 = wavelet_obj_arr_tmp.wavelet.time(0)
        s = wavelet_obj_arr_tmp.scales
        W_n = np.transpose(WaveletCoeff_var_arr[:, i_period:i_period + 1])
        # a print('np.shape(s), s: ', np.shape(s), s)
        # use the transpose to allow broadcasting
        # a real_sum = np.sum(W_n.real.T / s ** .5, axis=-1).T
        real_sum = np.sum(np.real(W_n).T / s ** .5, axis=-1).T
        var_vect_tmp = real_sum * (dj * dt ** .5 / (C_d * np.real(Y_00)))
        # a print('dj, dt, C_d, Y_00: ', dj, dt, C_d, Y_00)
        # a print('np.shape(var_vect_tmp), min_var, max_var: ', np.shape(var_vect_tmp), np.min(real_sum), np.max(real_sum))
        sub_wave_var_arr[0:num_times, i_period] = var_vect_tmp

    return sub_wave_var_arr


def get_PhaseAngle_and_PoyntingFlux_from_E_and_B(Et_arr, En_arr, Bt_arr, Bn_arr, Br_LocalBG_arr):
    """
    get the phase angle between dE and dB (or exactly from dE to dB) at every time and every scales
    get the Poynting flux in the radial direction
    """
    print('np.shape(Bt_arr), np.shape(Bn_arr): ')
    print(np.shape(Bt_arr), np.shape(Bn_arr))
    # input('Press any key to continue...')
    # d phi_E_arr = np.arctan2(Et_arr, En_arr) / np.pi * 180. # this is wrong
    # d phi_B_arr = np.arctan2(Bt_arr, Bn_arr) / np.pi * 180. # this is wrong, the correct format is np.arctan2(y,x) rather than np.arctan2(x,y)
    phi_E_arr = np.arctan2(En_arr, Et_arr) / np.pi * 180.
    # d phi_E_arr = np.angle(En_arr/Et_arr)
    phi_B_arr = np.arctan2(Bn_arr, Bt_arr) / np.pi * 180.
    phi_from_E_to_B_arr = phi_B_arr - phi_E_arr
    sub_gt_p180 = (phi_from_E_to_B_arr > +180.0)
    sub_lt_m180 = (phi_from_E_to_B_arr < -180.0)
    phi_from_E_to_B_arr[sub_gt_p180] = phi_from_E_to_B_arr[sub_gt_p180] - 360.0
    phi_from_E_to_B_arr[sub_lt_m180] = phi_from_E_to_B_arr[sub_lt_m180] + 360.0
    ##get the poynting flux in the radial direction, in unit of "watt/m^2"
    ##judge whether the poynting flux is anti-sunward or sunward
    mu0 = 4 * np.pi * 1.e-7  # unit: H/m
    from_mVpm_to_Vpm = 1.e-3  # 1 [mV/m]=1.e-3 [V/m]
    from_nT_to_T = 1.e-9  # 1 [nT] = 1.e-9 [T]
    PoyntingFlux_r_arr = (Et_arr * Bn_arr - En_arr * Bt_arr) * \
                         (from_mVpm_to_Vpm * from_nT_to_T) / mu0  # unit: watt/m^2
    median_PoyntingFlux = np.median(PoyntingFlux_r_arr, axis=1)
    print('median_PoyntingFlux: ', median_PoyntingFlux)
    # input('Press any key to continue...')
    sub_AntiSunward_prop = (PoyntingFlux_r_arr > 0.0)
    sub_Sunward_prop = (PoyntingFlux_r_arr < 0.0)
    '''
    ##judge whether dJi.dE & dJe.dE is positive or negative 
    ##by combining the conditions: (1) the polarity of Br, (2) the poynting flux's polarity, (3) the angle from dE to dB
    ##Following is the table to determine the polarity of dJi.dE & dJe.dE 
    ##based on the three conditions(1) B0r, (2) PFr, and (3) Phi(dB)-Phi(dE)
    '''
    sub_Br_gt_0 = (Br_LocalBG_arr > 0.0)
    sub_Br_lt_0 = (Br_LocalBG_arr > 0.0)
    sub_PFr_gt_0 = (PoyntingFlux_r_arr > 0.0)
    sub_PFr_lt_0 = (PoyntingFlux_r_arr > 0.0)
    sub_phi_m180_m90 = (-180 < phi_from_E_to_B_arr) & (phi_from_E_to_B_arr < -90)
    sub_phi_m90_m0 = (-90 < phi_from_E_to_B_arr) & (phi_from_E_to_B_arr < 0)
    sub_phi_p0_p90 = (0 < phi_from_E_to_B_arr) & (phi_from_E_to_B_arr < 90)
    sub_phi_p90_p180 = (90 < phi_from_E_to_B_arr) & (phi_from_E_to_B_arr < 180)

    sub_dJidE_gt_0_PlusBr = (sub_Br_gt_0 & sub_PFr_gt_0 & sub_phi_p90_p180) | \
                            (sub_Br_gt_0 & sub_PFr_lt_0 & sub_phi_m90_m0)
    sub_dJidE_gt_0_MinusBr = (sub_Br_lt_0 & sub_PFr_gt_0 & sub_phi_p0_p90) | \
                             (sub_Br_lt_0 & sub_PFr_lt_0 & sub_phi_m180_m90)
    sub_dJidE_lt_0_PlusBr = (sub_Br_gt_0 & sub_PFr_gt_0 & sub_phi_p0_p90) | \
                            (sub_Br_gt_0 & sub_PFr_lt_0 & sub_phi_m180_m90)
    sub_dJidE_lt_0_MinusBr = (sub_Br_lt_0 & sub_PFr_gt_0 & sub_phi_p90_p180) | \
                             (sub_Br_lt_0 & sub_PFr_lt_0 & sub_phi_m90_m0)

    sub_dJedE_lt_0_PlusBr = (sub_Br_gt_0 & sub_PFr_gt_0 & sub_phi_p90_p180) | \
                            (sub_Br_gt_0 & sub_PFr_lt_0 & sub_phi_m90_m0)
    sub_dJedE_lt_0_MinusBr = (sub_Br_lt_0 & sub_PFr_gt_0 & sub_phi_p0_p90) | \
                             (sub_Br_lt_0 & sub_PFr_lt_0 & sub_phi_m180_m90)
    sub_dJedE_gt_0_PlusBr = (sub_Br_gt_0 & sub_PFr_gt_0 & sub_phi_p0_p90) | \
                            (sub_Br_gt_0 & sub_PFr_lt_0 & sub_phi_m180_m90)
    sub_dJedE_gt_0_MinusBr = (sub_Br_lt_0 & sub_PFr_gt_0 & sub_phi_p90_p180) | \
                             (sub_Br_lt_0 & sub_PFr_lt_0 & sub_phi_m90_m0)

    '''
    sub_dJidE_gt_0_MinusBr = ((Br_LocalBG_arr < 0.0) & (PoyntingFlux_r_arr > 0.0) & \
                              ((0 < phi_from_E_to_B_arr) & (phi_from_E_to_B_arr < 90))) | \
                            ((Br_LocalBG_arr < 0.0) & (PoyntingFlux_r_arr < 0.0) & \
                              ((-180 < phi_from_E_to_B_arr) & (phi_from_E_to_B_arr < -90)))
    sub_dJidE_lt_0_PlusBr = ((Br_LocalBG_arr > 0.0) & (PoyntingFlux_r_arr > 0.0) & \
                              ((0 < phi_from_E_to_B_arr) & (phi_from_E_to_B_arr < 90))) | \
                            ((Br_LocalBG_arr > 0.0) & (PoyntingFlux_r_arr < 0.0) & \
                              ((-180 < phi_from_E_to_B_arr) & (phi_from_E_to_B_arr < -90)))
    sub_dJidE_lt_0_MinusBr = ((Br_LocalBG_arr < 0.0) & (PoyntingFlux_r_arr > 0.0) & \
                              ((90 < phi_from_E_to_B_arr) & (phi_from_E_to_B_arr < 180))) | \
                            ((Br_LocalBG_arr < 0.0) & (PoyntingFlux_r_arr < 0.0) & \
                              ((-90 < phi_from_E_to_B_arr) & (phi_from_E_to_B_arr < 0)))
    sub_dJedE_lt_0_PlusBr = ((Br_LocalBG_arr > 0.0) & (PoyntingFlux_r_arr > 0.0) & \
                                 ((90 < phi_from_E_to_B_arr) & (phi_from_E_to_B_arr < 180))) | \
                             ((Br_LocalBG_arr > 0.0) & (PoyntingFlux_r_arr < 0.0) & \
                                  ((-90 < phi_from_E_to_B_arr) & (phi_from_E_to_B_arr < 0)))
    sub_dJedE_lt_0_MinusBr = ((Br_LocalBG_arr < 0.0) & (PoyntingFlux_r_arr > 0.0) & \
                                 ((0 < phi_from_E_to_B_arr) & (phi_from_E_to_B_arr < 90))) | \
                             ((Br_LocalBG_arr < 0.0) & (PoyntingFlux_r_arr < 0.0) & \
                                  ((-180 < phi_from_E_to_B_arr) & (phi_from_E_to_B_arr < -90)))
    sub_dJedE_gt_0_PlusBr = ((Br_LocalBG_arr > 0.0) & (PoyntingFlux_r_arr > 0.0) & \
                                 ((0 < phi_from_E_to_B_arr) & (phi_from_E_to_B_arr < 90))) | \
                             ((Br_LocalBG_arr > 0.0) & (PoyntingFlux_r_arr < 0.0) & \
                                  ((-180 < phi_from_E_to_B_arr) & (phi_from_E_to_B_arr < -90)))
    sub_dJedE_gt_0_MinusBr = ((Br_LocalBG_arr < 0.0) & (PoyntingFlux_r_arr > 0.0) & \
                                  ((90 < phi_from_E_to_B_arr) & (phi_from_E_to_B_arr < 180))) | \
                             ((Br_LocalBG_arr < 0.0) & (PoyntingFlux_r_arr < 0.0) & \
                                  ((-90 < phi_from_E_to_B_arr) & (phi_from_E_to_B_arr < 0))) 
    '''
    sub_dJidE_gt_0 = sub_dJidE_gt_0_PlusBr | sub_dJidE_gt_0_MinusBr
    sub_dJidE_lt_0 = sub_dJidE_lt_0_PlusBr | sub_dJidE_lt_0_MinusBr
    sub_dJedE_gt_0 = sub_dJedE_gt_0_PlusBr | sub_dJedE_gt_0_MinusBr
    sub_dJedE_lt_0 = sub_dJedE_lt_0_PlusBr | sub_dJedE_lt_0_MinusBr
    return phi_from_E_to_B_arr, PoyntingFlux_r_arr, sub_dJidE_gt_0, sub_dJidE_lt_0, sub_dJedE_gt_0, sub_dJedE_lt_0


# get 'PoyntingFlux_r_arr' based on Real(ExB*)/mu0
def get_PoyntingFlux_r_arr(time_vect, \
                           WaveletCoeff_Et_in_SW_arr, WaveletCoeff_En_in_SW_arr, \
                           WaveletCoeff_Bt_arr, WaveletCoeff_Bn_arr):
    mu0 = 4 * np.pi * 1.e-7  # unit: H/m
    from_mVpm_to_Vpm = 1.e-3  # 1 [mV/m]=1.e-3 [V/m]
    from_nT_to_T = 1.e-9  # 1 [nT] = 1.e-9 [T]
    dtime = np.diff(time_vect).mean()
    PoyntingFlux_r_arr = (np.real(WaveletCoeff_Et_in_SW_arr * np.conj(WaveletCoeff_Bn_arr)) - \
                          np.real(WaveletCoeff_En_in_SW_arr * np.conj(WaveletCoeff_Bt_arr))) * \
                         (2 * dtime) * \
                         (from_mVpm_to_Vpm * from_nT_to_T) / mu0  # unit: watt/m^2/Hz
    # a median_PoyntingFlux = np.median(PoyntingFlux_r_arr,axis=1)
    # a print('median_PoyntingFlux based on Real(ExB*)/mu0: ', median_PoyntingFlux)
    # a input('Press any key to continue...')
    return PoyntingFlux_r_arr


# get 'ComplexOmega2k_arr' based on omega/k=(ExB*)/(B.B*)
def get_ComplexOmega2k_arr(time_vect, \
                           WaveletCoeff_Et_in_SW_arr, WaveletCoeff_En_in_SW_arr, \
                           WaveletCoeff_Bt_arr, WaveletCoeff_Bn_arr):
    mu0 = 4 * np.pi * 1.e-7  # unit: H/m
    from_mVpm_to_Vpm = 1.e-3  # 1 [mV/m]=1.e-3 [V/m]
    from_nT_to_T = 1.e-9  # 1 [nT] = 1.e-9 [T]
    dtime = np.diff(time_vect).mean()
    ComplexOmega2k_arr = ((WaveletCoeff_Et_in_SW_arr * np.conj(WaveletCoeff_Bn_arr)) - \
                          (WaveletCoeff_En_in_SW_arr * np.conj(WaveletCoeff_Bt_arr))) / \
                         (np.abs(WaveletCoeff_Bt_arr) ** 2 + np.abs(WaveletCoeff_Bn_arr) ** 2) * \
                         (from_mVpm_to_Vpm * from_nT_to_T / from_nT_to_T ** 2) * 1.e-3  # unit: km/s
    # a median_PoyntingFlux = np.median(PoyntingFlux_r_arr,axis=1)
    # a print('median_PoyntingFlux based on Real(ExB*)/mu0: ', median_PoyntingFlux)
    # a input('Press any key to continue...')
    return ComplexOmega2k_arr


# get 'VA_kmps'
def get_AlfvenSpeed(N_cm3, B_nT):
    mass_proton = 1.67e-27  # unit: kg
    mu0 = 4 * np.pi * 1.e-7  # unit: H/m
    VA_mps = B_nT * 1.e-9 / np.sqrt(mu0 * N_cm3 * 1.e6 * mass_proton)
    VA_kmps = VA_mps * 1.e-3
    return VA_kmps


# get 'VA_kmps'
def test_get_AlfvenSpeed(N_cm3, B_nT):
    mass_proton = 1.67e-27  # unit: kg
    mu0 = 4 * np.pi * 1.e-7  # unit: H/m
    VA_mps = B_nT * 1.e-9 / np.sqrt(mu0 * N_cm3 * 1.e6 * mass_proton)
    VA_kmps = VA_mps * 1.e-3
    return VA_kmps


# create bounds for grid cells
def _cell_bounds(points, bound_position=0.5):
    """
    Calculate coordinate cell boundaries.

    Parameters
    ----------
    points: numpy.array
        One-dimensional array of uniformly spaced values of shape (M,)
    bound_position: bool, optional
        The desired position of the bounds relative to the position
        of the points.

    Returns
    -------
    bounds: numpy.array
        Array of shape (M+1,)

    Examples
    --------
    >>> a = np.arange(-1, 2.5, 0.5)
    >>> a
    array([-1. , -0.5,  0. ,  0.5,  1. ,  1.5,  2. ])
    >>> cell_bounds(a)
    array([-1.25, -0.75, -0.25,  0.25,  0.75,  1.25,  1.75,  2.25])
    """
    assert points.ndim == 1, "Only 1D points are allowed"
    diffs = np.diff(points)
    delta = diffs[0] * bound_position
    bounds = np.concatenate([[points[0] - delta], points + delta])
    return bounds


# the purpose of this function is to define the center-position of the 3d-model to be at (0,0,0) (namely called "position-centered"), and to define the size of the 3d-model to be about [-0.5, 0.5] (namely called "size-normalized")
# the input array (num_points, num_components=3) includes the xyz-positions of the points of a certain 3d-model, which is read from a file like (.obj)
# An calling example is like:
#  filename = '/Users/jshept/Downloads/source/ParkerSolarProbe.obj'
#  psp_model = pv.read(filename)
#  psp_model.rotate_x(90)
#  psp_model.points = model_position_centered_size_normalized(psp_model.points)
def model_position_centered_size_normalized(model_points):
    # def model_points_update(model_points):
    psp_model_points = model_points
    psp_model_mean_xyz = np.mean(psp_model_points, axis=0)
    psp_model_min_xyz = np.amin(psp_model_points, axis=0)
    psp_model_max_xyz = np.amax(psp_model_points, axis=0)
    num_points, num_components = psp_model_points.shape
    psp_model_mean_xyz_2d_arr = np.outer(np.ones(num_points), psp_model_mean_xyz)
    psp_model_max_xyz_2d_arr = np.outer(np.ones(num_points), psp_model_max_xyz)
    psp_model_min_xyz_2d_arr = np.outer(np.ones(num_points), psp_model_min_xyz)
    psp_model_points = (psp_model_points - psp_model_mean_xyz_2d_arr) / (
            psp_model_max_xyz_2d_arr - psp_model_min_xyz_2d_arr)
    psp_model_points /= 1.
    model_points_new = psp_model_points
    return model_points_new


# create this bar_progress method which is invoked automatically from wget
def bar_progress(current, total, width=80):
    progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
    # Don't use print() as it will print in new line every time.
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()


# 实现断点续传（下载）文件
def download_JSHEPT(url, file_path):
    import sys
    import requests
    import os
    # 屏蔽warning信息
    requests.packages.urllib3.disable_warnings()
    # 第一次请求是为了得到文件总大小
    r1 = requests.get(url, stream=True, verify=False)
    total_size = int(r1.headers['Content-Length'])

    # 这重要了，先看看本地文件下载了多少
    if os.path.exists(file_path):
        temp_size = os.path.getsize(file_path)  # 本地已经下载的文件大小
    else:
        temp_size = 0
    # 显示一下下载了多少
    print(temp_size)
    print(total_size)
    # 核心部分，这个是请求下载时，从本地文件已经下载过的后面下载
    headers = {'Range': 'bytes=%d-' % temp_size}
    # 重新请求网址，加入新的请求头的
    r = requests.get(url, stream=True, verify=False, headers=headers)

    # 下面写入文件也要注意，看到"ab"了吗？
    # "ab"表示追加形式写入文件
    with open(file_path, "ab") as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                temp_size += len(chunk)
                f.write(chunk)
                f.flush()

                ###这是下载实现进度显示####
                done = int(50 * temp_size / total_size)
                sys.stdout.write("\r[%s%s] %d%%" % ('█' * done, ' ' * (50 - done), 100 * temp_size / total_size))
                sys.stdout.flush()
    print()  # 避免上面\r 回车符
