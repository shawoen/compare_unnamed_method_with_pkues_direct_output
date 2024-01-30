
import scipy.constants as const
import numpy as np
import julian
import pandas as pd

def load_pkues_data(dir_data):
    dir_data = r'D:\PKUES\PKUES-main\input'
    dir_fig = r'D:\PKUES\PKUES-main/figures'

    '''set up background fluctuations with power-law spectrum'''
    Vsw_km, B0_nT = 0, 121.

    '''read pdrk.in file'''
    # params_arr = np.loadtxt(dir_data + '/' + 'pdrk.in', skiprows=1)
    # qc, qe = params_arr[0, 0], params_arr[1, 0]  # unit: proton charge
    # mc, me = params_arr[0, 1], params_arr[1, 1]  # unit: proton mass
    # n_c, n_e = params_arr[0, 2], params_arr[1, 2]  # unit: m^{-3}
    # Tzc, Tze = params_arr[0, 3], params_arr[1, 3]  # unit: eV
    # Tpc, Tpe = params_arr[0, 4], params_arr[1, 4]  # unit: eV
    # vdc_c, vde_c = params_arr[0, 7], params_arr[1, 7]  # unit: c
    # n_p = n_c
    # vA_km = B0_nT / np.sqrt(const.mu_0 * (const.m_p * n_p + const.m_e * n_e)) * 1.e-12
    # omega_cp = const.e * B0_nT * 1.e-9 / const.m_p
    # vd_c, vd_e = vdc_c * const.c / vA_km * 1.e-3, vde_c * const.c / vA_km * 1.e-3

    qc, qe = 1., -1.  # unit: proton charge
    mc, me = 1., 5.447e-4  # unit: proton mass
    n_c, n_e = 4.12e9, 4.12e9  # unit: m^{-3}
    Tzc, Tze = 21.493, 17.189  # unit: eV
    Tpc, Tpe = 21.493, 17.189  # unit: eV
    vdc_c, vde_c = 0, 0  # unit: c
    n_p = n_c
    vA_km = B0_nT / np.sqrt(const.mu_0 * (const.m_p * n_p + const.m_e * n_e)) * 1.e-12
    omega_cp = const.e * B0_nT * 1.e-9 / const.m_p
    vd_c, vd_e = vdc_c * const.c / vA_km * 1.e-3, vde_c * const.c / vA_km * 1.e-3

    '''read polarization file'''
    pola_arr = np.loadtxt('Polarization.dat')
    kdp_vect, omega_vect, gamma_vect = pola_arr[:, 0], abs(pola_arr[:, 1]), pola_arr[:, 2]
    dEx_vect, dEy_vect, dEz_vect = pola_arr[:, 3] + 1j * pola_arr[:, 4], pola_arr[:, 5] + 1j * pola_arr[:, 6], \
                                   pola_arr[:, 7] + 1j * pola_arr[:, 8]
    dBx_vect, dBy_vect, dBz_vect = pola_arr[:, 9] + 1j * pola_arr[:, 10], pola_arr[:, 11] + 1j * pola_arr[:, 12], \
                                   pola_arr[:, 13] + 1j * pola_arr[:, 14]
    dVx_e_vect, dVy_e_vect, dVz_e_vect = pola_arr[:, 15] + 1j * pola_arr[:, 16], pola_arr[:, 17] + 1j * pola_arr[:, 18], \
                                         pola_arr[:, 19] + 1j * pola_arr[:, 20]
    dVx_pc_vect, dVy_pc_vect, dVz_pc_vect = pola_arr[:, 21] + 1j * pola_arr[:, 22], pola_arr[:, 23] + 1j * pola_arr[:,
                                                                                                           24], \
                                            pola_arr[:, 25] + 1j * pola_arr[:, 26]

    '''pick out solutions'''
    gamma_lst = np.empty(0)
    dEx_lst, dEy_lst, dEz_lst = np.empty(0), np.empty(0), np.empty(0)
    dBx_lst, dBy_lst, dBz_lst = np.empty(0), np.empty(0), np.empty(0)
    dVx_e_lst, dVy_e_lst, dVz_e_lst = np.empty(0), np.empty(0), np.empty(0)
    dVx_pc_lst, dVy_pc_lst, dVz_pc_lst = np.empty(0), np.empty(0), np.empty(0)
    # print("kdp",kdp_vect)
    # kdp_vect = kdp_vect * 84507.37 / 299492758
    # print("omega_vect",omega_vect)
    # omega_vect = 11.59 * omega_vect

    if Vsw_km != 0.0:
        index_MaxGamma = np.argmax(gamma_vect)
        kdp_MaxGamma = kdp_vect[index_MaxGamma]
        kdp_min = np.min(kdp_vect)
        kdp_max = np.max(kdp_vect)
    elif Vsw_km == 0.0:
        index_MaxGamma = np.argmax(gamma_vect)
        omega_MaxGamma = omega_vect[index_MaxGamma]
        omega_min = np.min(omega_vect)
        omega_max = np.max(omega_vect)
        # print('omega_min, omega_MaxGamma / 100., np.min(omega_vect): ')
        # print(omega_min, omega_MaxGamma / 10., np.min(omega_vect))
        # input('After determininng omega_min, omega_max!')

    num_kdp = 100
    if Vsw_km != 0.0:
        kdp_lst = np.logspace(np.log10(kdp_min), np.log10(kdp_max), num_kdp)
        omega_lst = np.empty(0)
        for kdp in kdp_lst:
            ind = np.argmin(np.abs(np.log10(kdp_vect) - np.log10(kdp)))
            omega_lst = np.append(omega_lst, omega_vect[ind])
            gamma_lst = np.append(gamma_lst, gamma_vect[ind])
            dEx_lst = np.append(dEx_lst, dEx_vect[ind])
            dEy_lst = np.append(dEy_lst, dEy_vect[ind])
            dEz_lst = np.append(dEz_lst, dEz_vect[ind])
            dBx_lst = np.append(dBx_lst, dBx_vect[ind])
            dBy_lst = np.append(dBy_lst, dBy_vect[ind])
            dBz_lst = np.append(dBz_lst, dBz_vect[ind])
            dVx_e_lst = np.append(dVx_e_lst, dVx_e_vect[ind])
            dVy_e_lst = np.append(dVy_e_lst, dVy_e_vect[ind])
            dVz_e_lst = np.append(dVz_e_lst, dVz_e_vect[ind])
            dVx_pc_lst = np.append(dVx_pc_lst, dVx_pc_vect[ind])
            dVy_pc_lst = np.append(dVy_pc_lst, dVy_pc_vect[ind])
            dVz_pc_lst = np.append(dVz_pc_lst, dVz_pc_vect[ind])

    if Vsw_km == 0.0:
        omega_lst = np.logspace(np.log10(omega_min), np.log10(omega_max), num_kdp).reshape(-1)
        kdp_lst = np.empty(0)
        if num_kdp == 1:
            gamma_ind = np.where(gamma_vect == np.max(gamma_vect))
            omega_lst = omega_vect[gamma_ind]
        for omega in omega_lst:
            ind = np.argmin(np.abs(np.log10(omega_vect) - np.log10(omega)))
            kdp_lst = np.append(kdp_lst, kdp_vect[ind])
            gamma_lst = np.append(gamma_lst, gamma_vect[ind])
            dEx_lst = np.append(dEx_lst, dEx_vect[ind])
            dEy_lst = np.append(dEy_lst, dEy_vect[ind])
            dEz_lst = np.append(dEz_lst, dEz_vect[ind])
            dBx_lst = np.append(dBx_lst, dBx_vect[ind])
            dBy_lst = np.append(dBy_lst, dBy_vect[ind])
            dBz_lst = np.append(dBz_lst, dBz_vect[ind])
            dVx_e_lst = np.append(dVx_e_lst, dVx_e_vect[ind])
            dVy_e_lst = np.append(dVy_e_lst, dVy_e_vect[ind])
            dVz_e_lst = np.append(dVz_e_lst, dVz_e_vect[ind])
            dVx_pc_lst = np.append(dVx_pc_lst, dVx_pc_vect[ind])
            dVy_pc_lst = np.append(dVy_pc_lst, dVy_pc_vect[ind])
            dVz_pc_lst = np.append(dVz_pc_lst, dVz_pc_vect[ind])

    '''set power law index and amplitude of magnetic field'''
    index = -1.5
    if Vsw_km != 0.0:
        power_psd_lst = 1.e-4 * (kdp_lst ** index)
        bulge_psd_lst = 0.1 - 2 * (np.log10(kdp_lst) - np.log10(kdp_lst[np.argmax(gamma_lst)])) ** 2
    if Vsw_km == 0.0:
        power_psd_lst = 1.e-4 * (omega_lst ** index)
        bulge_psd_lst = 0.1 - 0.5 * (np.log10(omega_lst) - np.log10(omega_lst[np.argmax(gamma_lst)])) ** 2
    bulge_psd_lst[np.where(np.logical_or(gamma_lst <= 0.0, bulge_psd_lst < power_psd_lst))] = 0.0
    psd_lst = power_psd_lst + bulge_psd_lst

    '''reconstruct time series of magnetic field'''
    phase_lst = np.random.uniform(low=0.0, high=2 * np.pi, size=(len(kdp_lst),))
    dBy_init_lst = np.sqrt(psd_lst) * np.exp(-1j * phase_lst)
    dBx_init_lst = dBx_lst / dBy_lst * dBy_init_lst
    dBz_init_lst = dBz_lst / dBy_lst * dBy_init_lst
    dEx_init_lst = dEx_lst / dBy_lst * dBy_init_lst
    dEy_init_lst = dEy_lst / dBy_lst * dBy_init_lst
    dEz_init_lst = dEz_lst / dBy_lst * dBy_init_lst
    dVx_pc_init_lst = dVx_pc_lst / dBy_lst * dBy_init_lst
    dVy_pc_init_lst = dVy_pc_lst / dBy_lst * dBy_init_lst
    dVz_pc_init_lst = dVz_pc_lst / dBy_lst * dBy_init_lst
    dVx_e_init_lst = dVx_e_lst / dBy_lst * dBy_init_lst
    dVy_e_init_lst = dVy_e_lst / dBy_lst * dBy_init_lst
    dVz_e_init_lst = dVz_e_lst / dBy_lst * dBy_init_lst
    ComplexOmega_lst = (dEx_init_lst * np.conj(dBy_init_lst) - dEy_init_lst * np.conj(dBx_init_lst)) / (
                        dBx_init_lst * np.conj(dBx_init_lst) + dBy_init_lst * np.conj(dBy_init_lst) +
                        dBz_init_lst * np.conj(dBz_init_lst)) * kdp_lst
    Omega_lst, Gamma_lst = np.real(ComplexOmega_lst), np.imag(ComplexOmega_lst)
    phi_dEx_dBx_lst = np.abs(np.rad2deg(np.angle(dEx_init_lst / dBx_init_lst)))
    phi_dEy_dBy_lst = np.abs(np.rad2deg(np.angle(dEy_init_lst / dBy_init_lst)))

    '''calculate time resolution and time length'''
    omega_sc_lst = (omega_lst + kdp_lst * Vsw_km * np.cos(np.radians(1.)) / vA_km) * omega_cp  # unit: rad/s)
    # print("omega_sc_lst",omega_sc_lst)
    # omega_sc_lst = omega_lst
    if Vsw_km == 0.0:
        T = 2 * np.pi / np.min(np.abs(omega_sc_lst)) * 3
        dt = 2 * np.pi / (np.max(np.abs(omega_sc_lst)) * 2)
    elif Vsw_km != 0.0:
        T = 2 * np.pi / np.min(np.abs(omega_sc_lst)) * 6
        dt = 2 * np.pi / (np.max(np.abs(omega_sc_lst)) * 4)

    num_times = int(T / dt)
    t_plot = np.linspace(0, dt * num_times, num_times)  # unit: s
    # print("t_plot",t_plot)

    '''calculate time series of fluctuations'''
    alpha = 1
    dBx_time_lst = np.sum(np.real(np.tile(dBx_init_lst, (num_times, 1)) * np.exp(
        +alpha * 1j * np.tile(omega_sc_lst, (num_times, 1)) * np.tile(t_plot.reshape(-1, 1), (1, len(kdp_lst))))),
                          axis=1)
    dBy_time_lst = np.sum(np.real(np.tile(dBy_init_lst, (num_times, 1)) * np.exp(
        +alpha * 1j * np.tile(omega_sc_lst, (num_times, 1)) * np.tile(t_plot.reshape(-1, 1), (1, len(kdp_lst))))),
                          axis=1)
    dBz_time_lst = np.sum(np.real(np.tile(dBz_init_lst, (num_times, 1)) * np.exp(
        +alpha * 1j * np.tile(omega_sc_lst, (num_times, 1)) * np.tile(t_plot.reshape(-1, 1), (1, len(kdp_lst))))),
                          axis=1)
    dEx_time_lst = np.sum(np.real(np.tile(dEx_init_lst, (num_times, 1)) * np.exp(
        +alpha * 1j * np.tile(omega_sc_lst, (num_times, 1)) * np.tile(t_plot.reshape(-1, 1), (1, len(kdp_lst))))),
                          axis=1)
    dEy_time_lst = np.sum(np.real(np.tile(dEy_init_lst, (num_times, 1)) * np.exp(
        +alpha * 1j * np.tile(omega_sc_lst, (num_times, 1)) * np.tile(t_plot.reshape(-1, 1), (1, len(kdp_lst))))),
                          axis=1)
    dEz_time_lst = np.sum(np.real(np.tile(dEz_init_lst, (num_times, 1)) * np.exp(
        +alpha * 1j * np.tile(omega_sc_lst, (num_times, 1)) * np.tile(t_plot.reshape(-1, 1), (1, len(kdp_lst))))),
                          axis=1)
    dVx_pc_time_lst = np.sum(np.real(np.tile(dVx_pc_init_lst, (num_times, 1)) * np.exp(
        +alpha * 1j * np.tile(omega_sc_lst, (num_times, 1)) * np.tile(t_plot.reshape(-1, 1), (1, len(kdp_lst))))),
                             axis=1)
    dVy_pc_time_lst = np.sum(np.real(np.tile(dVy_pc_init_lst, (num_times, 1)) * np.exp(
        +alpha * 1j * np.tile(omega_sc_lst, (num_times, 1)) * np.tile(t_plot.reshape(-1, 1), (1, len(kdp_lst))))),
                             axis=1)
    dVz_pc_time_lst = np.sum(np.real(np.tile(dVz_pc_init_lst, (num_times, 1)) * np.exp(
        +alpha * 1j * np.tile(omega_sc_lst, (num_times, 1)) * np.tile(t_plot.reshape(-1, 1), (1, len(kdp_lst))))),
                             axis=1)
    dVx_e_time_lst = np.sum(np.real(np.tile(dVx_e_init_lst, (num_times, 1)) * np.exp(
        +alpha * 1j * np.tile(omega_sc_lst, (num_times, 1)) * np.tile(t_plot.reshape(-1, 1), (1, len(kdp_lst))))),
                            axis=1)
    dVy_e_time_lst = np.sum(np.real(np.tile(dVy_e_init_lst, (num_times, 1)) * np.exp(
        +alpha * 1j * np.tile(omega_sc_lst, (num_times, 1)) * np.tile(t_plot.reshape(-1, 1), (1, len(kdp_lst))))),
                            axis=1)
    dVz_e_time_lst = np.sum(np.real(np.tile(dVz_e_init_lst, (num_times, 1)) * np.exp(
        +alpha * 1j * np.tile(omega_sc_lst, (num_times, 1)) * np.tile(t_plot.reshape(-1, 1), (1, len(kdp_lst))))),
                            axis=1)

    '''calculate time series in real unit'''
    # magnetic field from pkues (unit: nT)
    Bx_time_lst, By_time_lst, Bz_time_lst = dBx_time_lst * B0_nT, dBy_time_lst * B0_nT, (1. + dBz_time_lst) * B0_nT

    # electric field from pkues (unit: mV/m)
    Ex_time_lst, Ey_time_lst, Ez_time_lst = dEx_time_lst * B0_nT * vA_km * 1.e-3, \
                                            dEy_time_lst * B0_nT * vA_km * 1.e-3, \
                                            dEz_time_lst * B0_nT * vA_km * 1.e-3

    # mass center velocity of proton (unit: km/s)
    Vx_p_time_lst = (n_c * dVx_pc_time_lst) / n_p * vA_km
    Vy_p_time_lst = (n_c * dVy_pc_time_lst) / n_p * vA_km
    Vz_p_time_lst = (n_c * (dVz_pc_time_lst + vd_c)) / n_p * vA_km
    Vx_e_time_lst = (n_c * dVx_e_time_lst) / n_p * vA_km
    Vy_e_time_lst = (n_c * dVy_e_time_lst) / n_p * vA_km
    Vz_e_time_lst = (n_c * (dVz_e_time_lst + vd_c)) / n_p * vA_km
    # Vz_time_lst = (n_c * (dVz_pc_time_lst + vd_c)) / n_p * vA_km + Vsw_km

    temp = list(map(julian.from_jd, t_plot / (24 * 3600) + 2400000))    #The constructed time series is virtual,
    t_plot = temp                  # focusing only on the amount of change without looking at the specific value

    mag_df = pd.DataFrame()
    mag_df['epoch'] = t_plot
    mag_df['bt'] = Bx_time_lst
    mag_df['bn'] = By_time_lst
    mag_df['br'] = Bz_time_lst
    mag_df.set_index('epoch', inplace=True)
    '''load electric field data'''
    ele_df = pd.DataFrame()
    ele_df['epoch'] = t_plot
    ele_df['et'] = Ex_time_lst
    ele_df['en'] = Ey_time_lst
    ele_df['er'] = Ez_time_lst
    ele_df.set_index('epoch', inplace=True)
    '''load plasma moment data'''

    mom_df = pd.DataFrame()
    mom_df['epoch'] = t_plot
    mom_df['np_mom'] = n_c
    mom_df['tp_mom'] = Tzc
    mom_df['vpt_mom'] = Vx_p_time_lst
    mom_df['vpn_mom'] = Vy_p_time_lst
    mom_df['vpr_mom'] = Vz_p_time_lst
    mom_df['vpx_mom'] = Vx_p_time_lst
    mom_df['vpy_mom'] = Vy_p_time_lst
    mom_df['vpz_mom'] = Vz_p_time_lst
    mom_df.set_index('epoch', inplace=True)

    j_pkues_df = pd.DataFrame()
    j_pkues_df['epoch'] = t_plot
    # units: miuA
    j_pkues_df['j_x'] = (Vx_p_time_lst - Vx_e_time_lst) * n_c * const.e * 1e9
    j_pkues_df['j_y'] = (Vy_p_time_lst - Vy_e_time_lst) * n_c * const.e * 1e9
    j_pkues_df['j_z'] = (Vz_p_time_lst - Vz_e_time_lst) * n_c * const.e * 1e9
    j_pkues_df.set_index('epoch', inplace=True)

    return mag_df, ele_df, mom_df, j_pkues_df, kdp_vect, omega_vect


