# coding:utf-8
import os
from datetime import datetime
import numpy as np
from _beamforming import BeamForming


def my_makedirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)
        
        
if __name__ == '__main__':
    bm = BeamForming('./config.ini')
    # file_path = '../_exp/190619/test_data/left_10_20_-10_2000.wav'
    # min_theta = 45
    # max_theta = 135
    # bm.check_beam_forming(file_path, 0)
    direction_list = [-40, -30, -20, -10, 0, 10, 20, 30, 40]
    file_path = '../_exp/190630/recode_data/dis_40/'
    # file_name = 'left_10_20_' + str(direction_list[0]) + '_2000.wav'
    file_name = str(direction_list[0]) + '.wav'
    print('【' + file_path + str(direction_list[0]) + '】')
    wave_data = file_path + file_name
    bm_list, time_list = bm.check_beam_forming(wave_data, direction_list[0], semiauto_data=True)
    bm_array = np.zeros((len(direction_list), len(time_list)))
    bm_array[0, :] = bm_list
    for i, name in enumerate(direction_list[1:]):
        print('【' + str(name) + '】')
        # file_name = 'left_10_20_' + str(name) + '_2000.wav'
        file_name = str(name) + '.wav'
        wave_data = file_path + file_name
        bm_list, time_list = bm.check_beam_forming(wave_data, name, semiauto_data=True)
        bm_array[i + 1, :] = bm_list
    
    array_path = '../_array/19' + datetime.today().strftime("%m%d")
    my_makedirs(array_path)
    file_name = '/' + datetime.today().strftime("%m%d")
    np.save(array_path + file_name + '_dis40_bm_time', bm_array)
    np.save(array_path + file_name + '_dis40_time', time_list)
    print('Saved')