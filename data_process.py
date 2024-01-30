import numpy as np
import pandas as pd
import pycwt as wavelet
import julian
from munch import Munch
from python_utils_JSHEPT import get_local_mean_variable as get_bg
from python_utils_JSHEPT import get_plot_WaveletAnalysis_of_var_vect

def interp_and_smooth(df, t_win='0.02S', smooth=False, t_smooth=None):
    if t_smooth is None:
        t_smooth = t_win
    idx = pd.date_range(start=df.index[0], end=df.index[-1], freq=t_win)
    df_new = df.reindex(df.index.union(idx)).interpolate('time').loc[idx]
    if smooth:
        df_new = df_new.rolling(t_smooth, center=True).mean()
    return df_new


def interp_with_index(df, idx=1):
    return df.reindex(df.index.union(idx)).interpolate('time', limit_direction="both").loc[idx]


def cal_b_wavelet(index, bx, by, bz, s0=1.0, s1=100.0, num_periods=32):
    """calculate wavelet coefficients"""
    julian_lst = np.array([julian.to_jd(x) for x in index])
    time_vect = (julian_lst - julian_lst[0]) * (24. * 60. * 60)
    mwt = wavelet.Morlet(6)
    dt = index.to_series().diff().iloc[1].total_seconds()
    periods = np.logspace(np.log10(s0), np.log10(s1), num_periods)
    freqs = 1 / periods
    bx_wave, scales, freqs, coi, bx_fft, fftfreqs = wavelet.cwt(bx.values, dt, wavelet=mwt, freqs=freqs)
    by_wave, scales, freqs, coi, by_fft, fftfreqs = wavelet.cwt(by.values, dt, wavelet=mwt, freqs=freqs)
    bz_wave, scales, freqs, coi, bz_fft, fftfreqs = wavelet.cwt(bz.values, dt, wavelet=mwt, freqs=freqs)

    # period_range = [periods[0], periods[-1]]
    # var_vect = bx.values
    # time_vect, period_vect, wavelet_obj_arr, WaveletCoeff_var_arr, sub_wave_var_arr = \
    #     get_plot_WaveletAnalysis_of_var_vect(time_vect, var_vect, period_range=period_range, num_periods=num_periods)
    # bx_wave = WaveletCoeff_var_arr.T
    #
    # var_vect = by.values
    # time_vect, period_vect, wavelet_obj_arr, WaveletCoeff_var_arr, sub_wave_var_arr = \
    #     get_plot_WaveletAnalysis_of_var_vect(time_vect, var_vect, period_range=period_range, num_periods=num_periods)
    # by_wave = WaveletCoeff_var_arr.T
    #
    # var_vect = bz.values
    # time_vect, period_vect, wavelet_obj_arr, WaveletCoeff_var_arr, sub_wave_var_arr = \
    #     get_plot_WaveletAnalysis_of_var_vect(time_vect, var_vect, period_range=period_range, num_periods=num_periods)
    # bz_wave = WaveletCoeff_var_arr.T
    #
    # freqs = 1 / period_vect

    '''calculate background mean magnetic field and derive Bpara and Bperp'''
    var_vect = bx.values
    width2period = 10.0
    time_vect, period_vect, var_lbg_arr = get_bg(time_vect, var_vect, period_range=(s0, s1),
                                                 num_periods=num_periods,
                                                 width2period=width2period)
    bx_lbg_arr = var_lbg_arr.T
    var_vect = by.values
    width2period = 10.0
    time_vect, period_vect, var_lbg_arr = get_bg(time_vect, var_vect, period_range=(s0, s1),
                                                 num_periods=num_periods,
                                                 width2period=width2period)
    by_lbg_arr = var_lbg_arr.T
    var_vect = bz.values
    width2period = 10.0
    time_vect, period_vect, var_lbg_arr = get_bg(time_vect, var_vect, period_range=(s0, s1),
                                                 num_periods=num_periods,
                                                 width2period=width2period)
    bz_lbg_arr = var_lbg_arr.T

    '''calculate unit direction vector '''
    babs_lbg_arr = np.sqrt(bx_lbg_arr ** 2 + by_lbg_arr ** 2 + bz_lbg_arr ** 2)
    ebx_lbg_arr = bx_lbg_arr / babs_lbg_arr
    eby_lbg_arr = by_lbg_arr / babs_lbg_arr
    ebz_lbg_arr = bz_lbg_arr / babs_lbg_arr

    '''calculate parallel wavelet'''
    bpara_wave = bx_wave * ebx_lbg_arr + by_wave * eby_lbg_arr + bz_wave * ebz_lbg_arr

    '''create wavelet dictionary '''
    wt_dict = Munch()
    wt_dict.epoch = index.to_pydatetime()
    wt_dict.x_wt = bx_wave
    wt_dict.y_wt = by_wave
    wt_dict.z_wt = bz_wave
    wt_dict.para_wt = bpara_wave
    wt_dict.freq = freqs
    wt_dict.period = periods
    wt_dict.coi = coi

    '''create local background magnetic field dictionary'''
    lbg_dict = Munch()
    lbg_dict.epoch = index.to_pydatetime()
    lbg_dict.freq = freqs
    lbg_dict.period = periods
    lbg_dict.bx_lbg_arr = bx_lbg_arr
    lbg_dict.by_lbg_arr = by_lbg_arr
    lbg_dict.bz_lbg_arr = bz_lbg_arr

    return wt_dict, lbg_dict


def cal_wavelet(index, bx, by, bz, s0=1.0, s1=100.0, num_periods=32):
    """calculate wavelet coefficients"""
    mwt = wavelet.Morlet(6)
    dt = index.to_series().diff().iloc[1].total_seconds()
    periods = np.logspace(np.log10(s0), np.log10(s1), num_periods)
    freqs = 1 / periods
    bx_wave, scales, freqs, coi, bx_fft, fftfreqs = wavelet.cwt(bx.values, dt, wavelet=mwt, freqs=freqs)
    by_wave, scales, freqs, coi, by_fft, fftfreqs = wavelet.cwt(by.values, dt, wavelet=mwt, freqs=freqs)
    bz_wave, scales, freqs, coi, bz_fft, fftfreqs = wavelet.cwt(bz.values, dt, wavelet=mwt, freqs=freqs)

    '''create dictionary for wavelets'''
    wt_dict = Munch()
    wt_dict.epoch = index.to_pydatetime()
    wt_dict.x_wt = bx_wave
    wt_dict.y_wt = by_wave
    wt_dict.z_wt = bz_wave
    wt_dict.freq = freqs
    wt_dict.period = periods
    wt_dict.coi = coi

    return wt_dict


def cal_para_wavelet(wt_dict, lbg_dict):
    """calculate parallel wavelet coefficients"""

    '''calculate unit direction vector '''
    babs_lbg_arr = np.sqrt(lbg_dict.bx_lbg_arr ** 2 + lbg_dict.by_lbg_arr ** 2 + lbg_dict.bz_lbg_arr ** 2)
    ebx_lbg_arr = lbg_dict.bx_lbg_arr / babs_lbg_arr
    eby_lbg_arr = lbg_dict.by_lbg_arr / babs_lbg_arr
    ebz_lbg_arr = lbg_dict.bz_lbg_arr / babs_lbg_arr

    wt = wt_dict.x_wt * ebx_lbg_arr + wt_dict.y_wt * eby_lbg_arr + wt_dict.z_wt * ebz_lbg_arr

    '''add to wavelet dictionary'''
    wt_dict.para_wt = wt

    return wt_dict


def cal_psd(wt_dict):
    """calculate wavelet psd"""
    dt = np.diff(wt_dict.epoch)[0].total_seconds()
    x_psd_arr = np.abs(wt_dict.x_wt) ** 2 * 2 * dt
    y_psd_arr = np.abs(wt_dict.y_wt) ** 2 * 2 * dt
    z_psd_arr = np.abs(wt_dict.z_wt) ** 2 * 2 * dt
    trace_psd_arr = x_psd_arr + y_psd_arr + z_psd_arr
    para_psd_arr = np.abs(wt_dict.para_wt) ** 2 * 2 * dt
    perp_psd_arr = 0.5 * (trace_psd_arr - para_psd_arr)

    '''calculate 1d wavelet spectra'''
    x_psd_lst = np.mean(x_psd_arr, axis=1)
    y_psd_lst = np.mean(y_psd_arr, axis=1)
    z_psd_lst = np.mean(z_psd_arr, axis=1)
    para_psd_lst = np.mean(para_psd_arr, axis=1)
    perp_psd_lst = np.mean(perp_psd_arr, axis=1)
    trace_psd_lst = np.mean(trace_psd_arr, axis=1)

    x_lst = np.mean(np.abs(wt_dict.x_wt), axis=1)
    y_lst = np.mean(np.abs(wt_dict.y_wt), axis=1)
    z_lst = np.mean(np.abs(wt_dict.z_wt), axis=1)
    para_lst = np.mean(np.abs(wt_dict.para_wt), axis=1)
    trace_wt = np.sqrt(trace_psd_arr)
    trace_lst = np.mean(trace_wt, axis=1)
    perp_wt = np.sqrt(trace_wt**2 - np.abs(wt_dict.para_wt)**2)
    perp_lst = np.mean(perp_wt, axis=1)

    '''create psd dictionary'''
    psd_dict = Munch()
    psd_dict.epoch = wt_dict.epoch
    psd_dict.freq = wt_dict.freq
    psd_dict.period = wt_dict.period
    psd_dict.x_psd_arr = x_psd_arr
    psd_dict.y_psd_arr = y_psd_arr
    psd_dict.z_psd_arr = z_psd_arr
    psd_dict.para_psd_arr = para_psd_arr
    psd_dict.perp_psd_arr = perp_psd_arr
    psd_dict.trace_psd_arr = trace_psd_arr
    psd_dict.x_psd_lst = x_psd_lst
    psd_dict.y_psd_lst = y_psd_lst
    psd_dict.z_psd_lst = z_psd_lst
    psd_dict.para_psd_lst = para_psd_lst
    psd_dict.perp_psd_lst = perp_psd_lst
    psd_dict.trace_psd_lst = trace_psd_lst
    psd_dict.x_lst = x_lst
    psd_dict.y_lst = y_lst
    psd_dict.z_lst = z_lst
    psd_dict.trace_lst = trace_lst
    psd_dict.para_lst = para_lst
    psd_dict.perp_lst = perp_lst


    return psd_dict


def cal_helicity(epoch, period, wt_1, wt_2):
    """This is equivalent to the calculation of sense polarization"""
    helicity = np.real(2 * np.imag(wt_1 * np.conj(wt_2)) / (wt_1 * np.conj(wt_1) + wt_2 * np.conj(wt_2)))

    '''save into a dictionary'''
    hel_dict = Munch()
    hel_dict.epoch = epoch
    hel_dict.period = period
    hel_dict.helicity = helicity

    return hel_dict


def cal_bg_velocity(df, t_win='1000S'):
    v_bg_df = df.rolling(t_win, center=True).mean()
    return v_bg_df


def cal_plasma_frame_efield(e_df, b_df, v_bg_df):
    """calculate electric field in plasma frame with pre-interpolated data"""
    '''please use rtn data'''
    er_conv = -(v_bg_df.vpt_mom * b_df.bn - v_bg_df.vpn_mom * b_df.bt) * 1.e-3
    et_conv = -(v_bg_df.vpn_mom * b_df.br - v_bg_df.vpr_mom * b_df.bn) * 1.e-3
    en_conv = -(v_bg_df.vpr_mom * b_df.bt - v_bg_df.vpt_mom * b_df.br) * 1.e-3

    '''calculation'''
    e_df['er_pl'] = e_df.er - er_conv
    e_df['et_pl'] = e_df.et - et_conv
    e_df['en_pl'] = e_df.en - en_conv

    return e_df
