
import time
from data_process import interp_and_smooth, cal_b_wavelet, interp_with_index, cal_wavelet, cal_bg_velocity
from data_process import cal_psd, cal_para_wavelet
from data_process import cal_plasma_frame_efield
from svd_santolik import em_svd
from svd_santolik import cal_phase_k_omega_em_svd, cal_propagation_em_svd
from bellan import cal_j_invert_bellan
from energy_conversion import cal_ecr_and_pdr
from load_pkues_data import load_pkues_data
from plot import fake_log, plot_original_dispersion_relation, plot_em_svd_results
from plot import plot_dispersion_relation, plot_ecr_and_pdr_2d, plot_ecr_and_pdr_1d
from plot import plot_angle, plot_j_and_psd_j, plot_j_compare, plot_j_1d, plot_j_time_series

fig_dir = 'figures/'

"""load pkues data"""
dir_data = r'D:\PKUES\PKUES-main\input'
mag_df, ele_df, mom_df, j_pkues_df, kdp_vect, omega_vect = load_pkues_data(dir_data)

'''interpolate and smooth'''
mag_df_int = interp_and_smooth(mag_df)

'''interpolate electric field to magnetic field timestamp'''
ele_df_int = interp_with_index(ele_df, idx=mag_df_int.index)

'''interpolate plasma moment to magnetic field timestamp'''
mom_df_int = interp_with_index(mom_df, idx=mag_df_int.index)

'''interpolate j_pkues to magnetic field timestamp'''
j_pkues_df = interp_with_index(j_pkues_df, idx=mag_df_int.index)

'''calculate background velocity, the default smoothing time window is 1000 s'''
v_bg_df = cal_bg_velocity(mom_df_int, t_win='1000S')

'''assign electric field in R direction'''
# ele_df_int['er'] = 0

'''calculate electric field in plasma frame and append the data to the interpolated electric field df'''
'''make sure the three input dataframes have the same timestamp!!!'''
ele_df_int = cal_plasma_frame_efield(ele_df_int, mag_df_int, v_bg_df)

'''calculate wavelet coefficient and psd'''
mean_dt = mag_df_int.index.to_series().diff().mean().total_seconds()
s0, s1 = round(mean_dt * 2, 2), 100  # min & max scales
s0, s1 = 3.5, 300  # min & max scales
num_periods = 16
mag_wt_dict, mag_lbg = cal_b_wavelet(mag_df_int.index, mag_df_int.br, mag_df_int.bt, mag_df_int.bn, s0=s0,
                                     s1=s1, num_periods=num_periods)

ele_wt_dict = cal_wavelet(ele_df_int.index, ele_df_int.er_pl, ele_df_int.et_pl, ele_df_int.en_pl, s0=s0, s1=s1,
                          num_periods=num_periods)

'''calculate parallel wavelet for electric field'''
ele_wt_dict = cal_para_wavelet(ele_wt_dict, mag_lbg)

'''calculate psd'''
mag_psd = cal_psd(mag_wt_dict)
ele_psd = cal_psd(ele_wt_dict)

'''apply to em_svd'''
print("start em_svd")
start_time = time.time()
em_svd_dict = em_svd(mag_wt_dict, ele_wt_dict)

'''calculate phase velocity in plasma frame'''
em_svd_dict = cal_phase_k_omega_em_svd(em_svd_dict, v_bg_df)

'''calculate propagation properties using em-svd results'''
em_svd_dict = cal_propagation_em_svd(em_svd_dict, mag_lbg)
end_time = time.time()
print("em_svd process last", end_time - start_time)

'''calculate current using invert Bellan method'''
start_time = time.time()
j_wt_dict = cal_j_invert_bellan(em_svd_dict.k_pl, mag_wt_dict)

'''calculate parallel wavelet for current density'''
j_wt_dict = cal_para_wavelet(j_wt_dict, mag_lbg)
end_time = time.time()
print("calculate j_wt last", end_time - start_time)

'''calculate wavelet for pkues current density'''
start_time = time.time()
j_pkues_wt_dict = cal_wavelet(j_pkues_df.index, j_pkues_df.j_z, j_pkues_df.j_x, j_pkues_df.j_y, s0=s0, s1=s1,
                              num_periods=num_periods)
j_pkues_wt_dict = cal_para_wavelet(j_pkues_wt_dict, mag_lbg)
end_time = time.time()
print("calculate j_pkues last", end_time - start_time)

'''calculate ecr and pdr'''
ecr_pdr_dict_bellan, ecr_pdr_df_bellan = cal_ecr_and_pdr(j_wt_dict, ele_wt_dict, mag_psd, ele_psd)
ecr_pdr_dict_pkues, ecr_pdr_df_pkues = cal_ecr_and_pdr(j_pkues_wt_dict, ele_wt_dict, mag_psd, ele_psd)

'''plot time series'''
fig_file = 'em_svd_' + '.png'
title_str = 'em_svd_result'
plot_em_svd_results(mag_df, em_svd_dict, fig_dir, fig_file, title_str)

fig_file = "j_pkues_time_series"
plot_j_time_series(j_pkues_df, mag_df_int, ele_df_int, fig_dir, fig_file)

# fig_file = 'j_pkues_time_series_smooth'
# j_pkues_df = interp_and_smooth(j_pkues_df)
# plot_j_time_series(j_pkues_df, mag_df_int, ele_df_int, fig_dir, fig_file)

'''plot dispersion relation'''
fig_file = 'dispersion_relation_em_svd_' + '.png'
title_str = 'em_svd_'
plot_dispersion_relation(em_svd_dict, fig_dir, fig_file, title_str)

"""plot pkues dispersion relation"""
fig_file = 'pkues dispersion relation' + 'png'
title_str = 'pkues '
plot_original_dispersion_relation(kdp_vect, omega_vect, fig_dir, fig_file, title_str)

'''plot j and psd_j'''
fig_file = 'j and psd_j'
title_str = 'reverse bellan'
start_time = time.time()
plot_j_and_psd_j(j_wt_dict, fig_dir, fig_file, title_str)
fig_file = 'j_angle'
title_str = 'unnamed method J'
plot_angle(j_wt_dict, fig_dir, fig_file, title_str)
end_time = time.time()
print("plot_j and psd j last", end_time - start_time)

# j_psd_wt = cal_psd(j_wt_dict)
# fig_file = "1d_j_and_psd_j"
# start_time = time.time()
# plot_j_1d(j_psd_wt, fig_dir, fig_file, title_str)
# end_time = time.time()
# print("plot j 1d last ", end_time - start_time)

fig_file = "j_pkues"
title_str = "pkues"
start_time = time.time()
plot_j_and_psd_j(j_pkues_wt_dict, fig_dir, fig_file, title_str)
fig_file = "j_pkues_angle"
title_str = 'pkues J'
plot_angle(j_pkues_wt_dict, fig_dir, fig_file, title_str)
end_time = time.time()
print("plot j pkues last", end_time - start_time)

# start_time = time.time()
# j_psd_pkues = cal_psd(j_pkues_wt_dict)
# fig_file = "1d_j_pkues"
# plot_j_1d(j_psd_pkues, fig_dir, fig_file, title_str)
# end_time = time.time()
# print("plot 1d j pkues last", end_time - start_time)

'''compare pkues and reverse bellan '''
fig_file = 'j_compare'
title_str = 'compare'
start_time = time.time()
plot_j_compare(j_pkues_wt_dict, j_wt_dict, fig_dir, fig_file, title_str)
end_time = time.time()
print("plot j compare last", end_time - start_time)

'''plot ecr and pdr'''
fig_file = 'bellan_energy_conversion_rate_and_pseudo_damping_rate_2d_' + '.png'
title_str = 'bellan_'
plot_ecr_and_pdr_2d(ecr_pdr_dict_bellan, fig_dir, fig_file, title_str)
title_str = 'pkues'
fig_file = 'pkues_energy_conversion_rate_and_pseudo_damping_rate_2d_' + '.png'
plot_ecr_and_pdr_2d(ecr_pdr_dict_pkues, fig_dir, fig_file, title_str)
