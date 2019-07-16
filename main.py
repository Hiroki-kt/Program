# coding:utf-8
import numpy as np
import math
# import scipy
# from matplotlib import pyplot as plt

from beamforming import BeamForming
from shape_from_sound import ShapeFromSound


def main(true_direction, sign='plus'):
    bm = BeamForming('./config.ini')
    sfs = ShapeFromSound('./config.ini')
    file_path = '../_exp/190612/test_data/'
    r = np.zeros((len(sfs.ss_list), len(bm.origin_freq_list)), dtype=np.complex)
    for i, name in enumerate(sfs.ss_list):
        print(name)
        if name == -0.25:
            data_path = 'left_25'
        elif name == -0.15:
            data_path = 'left_15'
        elif name == 0.15:
            data_path = 'right_15'
        elif name == 0.25:
            data_path = 'right_25'
        else:
            data_path = 0
            print("ERROR")
        if sign == 'plus':
            wave_data = file_path + data_path + '_' + str(true_direction) + '.wav'
        else:
            wave_data = file_path + data_path + '_m' + str(true_direction) + '.wav'
        print('===============================================')
        print(wave_data)
        print('===============================================')
        bm_data, direction_data, intensity = bm.execute_beam_forming(wave_data, 100)
        r[i, :] = intensity
        # bm.checkDOA()
    print(r)
    theta_list = []
    rho_list = []
    normal_list = np.zeros((4,2), dtype=np.complex)
    for i in range(r.shape[1]):
        n = sfs.shape_from_sound(r[:, i])
        # print(n)
        normal = n / np.sqrt(n[0] ** 2 + n[1] ** 2)
        theta = math.degrees(math.atan(normal[0]/normal[1]))
        rho = np.sqrt(n[0] ** 2 + n[1] ** 2)
        print("+++++++++++++++++++++++++++++++++++++++")
        print('surface normal:', normal, 'theta:', theta, 'rho:', rho)
        print("+++++++++++++++++++++++++++++++++++++++")
        theta_list.append(theta)
        rho_list.append(rho)
        normal_list[i, :] = normal
    return theta_list, rho_list, normal_list, r




if __name__ == '__main__':

    '''ini save data'''
    direction_list = [0, 10, 20, 30, 40]
    theta_array = np.zeros((4,9), dtype=np.float)
    rho_array = np.zeros_like(theta_array)
    normal_array = np.zeros((4, 2, 9), dtype=np.complex)
    intensity_array = np.zeros((4, 4, 9), dtype=np.float)

    '''main'''
    for i, name in enumerate(direction_list):
        theta_list, rho_list, normal_list, intensity_list = main(name)
        theta_array[:, i] = theta_list
        rho_array[:, i] = rho_list
        normal_array[:, :, i] = normal_list
        intensity_array[:, :, i] = intensity_list
    for i, name in enumerate(direction_list[1:]):
        theta_list, rho_list, normal_list, intensity_list = main(name, sign='minus')
        theta_array[:, i+5] = theta_list
        rho_array[:, i+5] = rho_list
        normal_array[:, :, i+5] = normal_list
        intensity_array[:, :, i+5] = intensity_list
    # print(theta_array)
    true_list = [0, 10, 20, 30, 40, -10, -20, -30, -40]
    error_array = theta_array - true_list * np.ones_like(theta_array)
    print(error_array)

    '''save data'''
    np.save('../_array/theta_array', theta_array)
    np.save('../_array/rho_array', rho_array)
    np.save('../_array/normal_array', normal_array)
    np.save('../_array/intensity_array', intensity_array)