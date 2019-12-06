# coding:utf-8
from matplotlib import pyplot as plt
import numpy as np
import configparser
import os
from scipy import signal
from _function import MyFunc
from _beamforming import BeamForming


class CheckSoundData():
    def __init__(self, path, config_path):
        if os.path.exists(config_path):
            config = configparser.ConfigParser()
            config.read(config_path)
        else:
            print("default param")
        self.data, self.channels, self.sampling, self.frames = MyFunc().wave_read_func(path)
        self.tf = BeamForming().steering_vector()
        
    def check_amptitude_about_time(self, origin):
        origin_data, o_channels, o_sampling, o_frames = MyFunc().wave_read_func(origin)
        start_time = MyFunc().zero_cross(self.data, 128, self.sampling, 512)
        o_start_time = MyFunc().zero_cross(origin_data, 128, self.sampling, 512)
        print(start_time, o_start_time)

    def beamforming(self, time_data):
        freq_list, time_list, stft_data = signal.stft(time_data, self.sampling)
    
    def mic_data(self):
    
if __name__ == '__main__':
    dir_name = MyFunc().data_search(191029, 'P', '03', calibration=True)
    path = MyFunc().recode_data_path + dir_name + '10.wav'
    origin_path = MyFunc().speaker_sound_path + '2_up_tsp_8num.wav'

    check = CheckSoundData()