#coding:utf-8
import numpy as np
import wave
import os
import sys
import configparser
import math

class SimulationEnvs(object):
    def __init__(self):
        pass

    def mic_positions(self, mic_r, mic_num):
        # return mic positions of mic array
        mic_pos_list = []
        for mic_id in range(mic_num):
            theta = mic_id / mic_num * 360
            mic_pos_list.append(position(mic_r, theta))
        print('#Create circular microphone array position')
        #print(len(mic_pos_list), mic_pos_list[0].pos())
        return mic_pos_list

    def ss_positions(self, radius, min_theta, max_theta, theta_interval):
        # return sound source theta list , and sound source position class list
        theta_list = np.arange(min_theta, max_theta + theta_interval, theta_interval)
        ss_pos_list = []
        for theta in theta_list:
            ss_pos_list.append(position(radius, theta))
        print('#Create temporal sound source position list')
        return theta_list, ss_pos_list

    def wave_read_func(self, wave_path):
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

            print('*****************************')
            print('Read wave file:', wave_path)
            print('Mic channel num:', w_channel)
            print('Sampling rate:', w_sanpling_rate)
            print('Frame_num:', w_frames_num, ' time:', w_frames_num / float(w_sanpling_rate))
            print('sound data shape:', data.shape)
            print('*****************************')

            return data, w_channel, w_sanpling_rate, w_frames_num

    def create_mic_input(self, sound_r, sound_dir, ch=1):
        s_theta = sound_dir * math.pi / 180.
        s_x = sound_r * math.cos(s_theta)
        s_y = sound_r * math.sin(s_theta)
        center2sound_dis = math.sqrt(s_x ** 2 + s_y ** 2)
        delay_list = []
        for mic in self.mic_pos_list:
            mx, my = mic.pos()
            mic2sound_dis = math.sqrt((mx - s_x) ** 2 + (my - s_y) ** 2)
            delay_point = round((mic2sound_dis - center2sound_dis) / self.sound_speed * self.w_sampling_rate)
            delay_list.append(delay_point)

        delay_min, delay_max = min(delay_list), max(delay_list)
        target_sound_data = self.sound_data[ch - 1, :]
        sound_fnum = self.w_frames_num + delay_max - delay_min
        data = np.zeros((len(self.mic_pos_list), sound_fnum))
        for i in range(len(self.mic_pos_list)):
            data[i, -self.w_frames_num + delay_list[i] + delay_min:self.w_frames_num + delay_list[i] - delay_min] = target_sound_data
        self.w_frames_num = sound_fnum
        self.w_channel = len(self.mic_pos_list)
        return data

    def freq_id(self):
        id_min = np.abs(self.freq_list - self.freq_min).argmin()
        id_max = np.abs(self.freq_list - self.freq_max).argmin()
        return id_min, id_max


class position(object):
    def __init__(self, r, theta):
        #r[m], theta[deg]
        theta = theta * math.pi / 180
        self.x = r*math.cos(theta)
        self.y = r*math.sin(theta)

    def pos(self):
        return self.x, self.y
