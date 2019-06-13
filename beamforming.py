# coding:utf-8
import configparser
import math
import os
import sys
import wave
import matplotlib.pyplot as plt

import numpy as np

from simulation_envs import SimulationEnvs

'''
beamformer class by python 3
configparser depend on python 3
'''


class beamforming(SimulationEnvs):
    def __init__(self, config_path):
        super().__init__()
        if os.path.exists(config_path):
            config = configparser.ConfigParser()
            config.read(config_path)
            # soudn param
            self.sound_speed = float(config['SoundParam']['sound_speed'])
            self.FFT_sample_num = int(config['SoundParam']['FFT_sample'])
            self.hamming_window = np.hamming(self.FFT_sample_num)
            self.FFT_shift = int(config['SoundParam']['FFT_SHIFT'])
            self.bm_window = int(config['SoundParam']['beamformer_window'])
            self.bm_period = int(config['SoundParam']['beamformer_period'])
            self.freq_min = int(config['SoundParam']['freq_min'])
            self.freq_max = int(config['SoundParam']['freq_max'])
            self.diff_dir = int(config['SoundParam']['diffraction_diraction'])
            self.wall_interval = float(config['SoundParam']['wall_interval'])
            self.wall_thick = float(config['SoundParam']['wall_thick'])

            # circular microphone
            mic_radius = float(config['MicArray']['radius'])
            mic_channel_num = int(config['MicArray']['mic_array_num'])
            self.mic_pos_list = self.mic_positions(mic_radius, mic_channel_num)

            # sound source
            ss_radius = float(config['SoundSource']['radius'])
            ss_min_theta = float(config['SoundSource']['min_theta'])
            ss_max_theta = float(config['SoundSource']['max_theta'])
            ss_theta_interval = float(config['SoundSource']['theta_interval'])
            self.ss_theta_list, self.ss_pos_list = self.ss_positions(ss_radius, ss_min_theta, ss_max_theta,
                                                                     ss_theta_interval)

            # sound
            sound_method = config['sound']['sound_method']
            sound_r = float(config['sound']['sound_r'])
            sound_dir = float(config['sound']['sound_dir'])
            self.wave_data = config['sound']['wave_data']
            self.sound_data_func(self.wave_data, sound_method, sound_r, sound_dir)
            print('#Success parse config param')

            self.freq_list = np.fft.rfftfreq(self.FFT_sample_num, d=1. / self.w_sampling_rate)
            self.freq_min_id, self.freq_max_id = self.freq_id()

        else:
            print("#Couldn't find", config_path)
            sys.exit()

    def sound_data_func(self, wave_data, sound_method, sound_r=10., sound_dir=90.):
        if sound_method == 'create':
            self.sound_data, self.w_channel, self.w_sampling_rate, self.w_frames_num = self.wave_read_func(wave_data)
            self.sound_data = self.create_mic_input(sound_r, sound_dir)
            print('#Create microphone array input')
        elif sound_method == 'wavuse':
            self.sound_data, self.w_channel, self.w_sampling_rate, self.w_frames_num = self.wave_read_func(wave_data)
            print('#Read wave file')
        else:
            print('#Data input by othet method.')
            print('class variabel of sound data is self.sound_data')

    def steering_vector(self, freq_array):
        # freq_array = self.freq_list
        freq_num = freq_array.shape[0]
        # print(freq_num)
        temp_ss_num = len(self.ss_theta_list)

        tf = np.zeros((temp_ss_num, freq_num, self.w_channel), dtype=np.complex)
        beam_conf = np.zeros((temp_ss_num, freq_num, self.w_channel), dtype=np.complex)

        # create tf
        l_w = math.pi * freq_array / self.sound_speed
        micx_list = []
        micy_list = []
        for micp in self.mic_pos_list:
            x, y = micp.pos()
            micx_list.append(x)
            micy_list.append(y)
        micx_array = np.array(micx_list)
        micy_array = np.array(micy_list)

        freq_repeat_array = np.ones((freq_num, self.w_channel), dtype=np.complex) * freq_array.reshape(
            (freq_num, -1)) * -1j * 2 * np.pi # ??

        for i, ss_pos in enumerate(self.ss_pos_list):
            sx, sy = ss_pos.pos()
            center2ss_dis = math.sqrt(sx ** 2 + sy ** 2)
            mic2ss_dis = np.sqrt((micx_array - sx) ** 2 + (micy_array - sy) ** 2)
            dis_diff = (mic2ss_dis - center2ss_dis) / self.sound_speed  # * self.w_sampling_rate 打消
            dis_diff_repeat_array = np.ones((freq_num, self.w_channel)) * dis_diff.reshape((-1, self.w_channel))
            tf[i, :, :] = np.exp(freq_repeat_array * dis_diff_repeat_array)
            # beam_conf[i,:,:] = tf[i,:,:]/ ()
        print('#Create transfer funtion')
        self.tf = tf.conj()#360*257*8

    def beamformer_localization(self, fdata):
        fdata = fdata.transpose(1, 0)
        #tf = self.tf[:, self.freq_min_id:self.freq_max_id + 1, :]#360*257*8
        tf = self.tf
        tf[:,:self.freq_min_id,:] = tf[:,:self.freq_min_id,:]*0
        tf[:,self.freq_max_id:,:] = tf[:,self.freq_max_id:,:]*0
        bm = tf * fdata #(360*257*8)*(257*8)
        bms = bm.sum(axis=2) #mic distance sum
        bmp = np.sqrt(bms.real ** 2 + bms.imag ** 2)
        self.bms = bms
        self.bmp = bmp
        
        return bmp,bms #360*257

    def beamformer_separation(self, max_theta, fdata):
        tf = self.tf[int(max_theta), :, :]
        sep = tf*fdata.T #要素計算
        #sep_signal = np.concatenate((sep,sep.conj()),axis=0)
        return sep

    def wave_save(self,sep_data,number,name):
        sep_data = sep_data.astype(np.int16)
        w = wave.Wave_write("./sep_data/"+str(number)+"/"+name+".wav")
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(self.w_sampling_rate)
        w.writeframes(b''.join(sep_data))
        w.close()
        print('finish saving')


