# coding:utf-8
import numpy as np
import wave
# import os
# import sys
# import configparser
import math


class SimulationEnvs(object):
    def __init__(self):
        pass
    
    @staticmethod
    def mic_positions(mic_r, mic_num):
        # return mic positions of mic array
        mic_pos_list = []
        for mic_id in range(mic_num):
            theta = mic_id / mic_num * 360 + 90
            mic_pos_list.append(Position(mic_r, theta))
        print('#Create circular microphone array position')
        # print(mic_pos_list[0].y, mic_pos_list[1].x, mic_pos_list[2].y, mic_pos_list[3].x)
        return mic_pos_list
    
    @staticmethod
    def ss_positions(radius, min_theta, max_theta, theta_interval):
        # return sound source theta list , and sound source position class list
        theta_list = np.arange(min_theta, max_theta + theta_interval, theta_interval)
        ss_pos_list = []
        for theta in theta_list:
            ss_pos_list.append(Position(radius, theta))
        print('#Create temporal sound source position list')
        print("theta 0 's direction is", ss_pos_list[0].pos())
        return theta_list, ss_pos_list
    
    @staticmethod
    def wave_read_func(wave_path):
        with wave.open(wave_path, 'r') as waveFIle:
            w_channel = waveFIle.getnchannels()
            w_sanpling_rate = waveFIle.getframerate()
            w_frames_num = waveFIle.getnframes()
            w_sample_width = waveFIle.getsampwidth()
            
            data = waveFIle.readframes(w_frames_num)
            if w_sample_width == 2:
                data = np.frombuffer(data, dtype='int16').reshape((w_frames_num, w_channel)).T
            elif w_sample_width == 4:
                data = np.frombuffer(data, dtype='int32').reshape((w_frames_num, w_channel)).T
            
            '''
            print('*****************************')
            print('Read wave file:', wave_path)
            print('Mic channel num:', w_channel)
            print('Sampling rate:', w_sanpling_rate)
            print('Frame_num:', w_frames_num, ' time:', w_frames_num / float(w_sanpling_rate))
            print('sound data shape:', data.shape)
            print('*****************************')
            '''
            
            return data, w_channel, w_sanpling_rate, w_frames_num


class Position(object):
    def __init__(self, r, theta):
        # r[m], theta[deg]
        theta = theta * math.pi / 180
        self.x = r * math.cos(theta)
        self.y = r * math.sin(theta)
    
    def pos(self):
        return self.x, self.y


class Id(object):
    @staticmethod
    def id(_list, des):
        id_des = np.abs(_list - des).argmin()
        return id_des
