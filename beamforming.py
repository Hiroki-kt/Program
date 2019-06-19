# coding:utf-8
import configparser
import math
import os
import sys
import wave
import matplotlib.pyplot as plt
import scipy.signal

import numpy as np

from simulation_envs import SimulationEnvs
from shape_from_sound import ShapeFromSound

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
            self.combine_num = int(config['SoundParam']['combine_num'])
            odigin_freq_list = []
            for i in range(self.combine_num):
                id = 'original' + str(i+1) + '_freq'
                odigin_freq_list.append(float(config['SoundParam'][id]))
            self.odigin_freq_list = odigin_freq_list

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
            wave_data = config['sound']['wave_data']
            # self.sound_data_func(wave_data, sound_method, sound_r, sound_dir)
            print('#Success parse config param')

            #self.freq_list = np.fft.rfftfreq(self.FFT_sample_num, d=1. / self.w_sampling_rate)
            #self.freq_min_id, self.freq_max_id = self.freq_id()

        else:
            print("#Couldn't find", config_path)
            sys.exit()

    def sound_data_func(self, wave_data, sound_method='wavuse', sound_r=10., sound_dir=90.):
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

    def steering_vector(self, freq_array, combine_num):
        # freq_array = self.freq_list
        freq_num = freq_array.shape[0]
        # print(freq_num)
        temp_ss_num = len(self.ss_theta_list)

        tf = np.zeros((temp_ss_num, freq_num, self.w_channel), dtype=np.complex)
        beam_conf = np.zeros((temp_ss_num, freq_num, self.w_channel), dtype=np.complex)
        new_tf = np.zeros((temp_ss_num, freq_num, self.w_channel, combine_num), dtype=np.complex)

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
        print('#Create transfer funtion', tf.shape)
        self.tf = tf.conj()#360*257*8
        for i in range(combine_num):
            new_tf[:, :, :, i] = self.tf
        self.new_tf = new_tf

    def beamformer_localization(self, fdata, freq_list):
        fdata = fdata.transpose(1, 0)
        #tf = self.tf[:, self.freq_min_id:self.freq_max_id + 1, :]#360*257*8
        tf = self.tf
        # tf = self.new_tf
        freq_min_id, freq_max_id = self.freq_id(freq_list)
        tf[:, :freq_min_id, :] = tf[:, :freq_min_id, :]*0
        tf[:, freq_max_id:, :] = tf[:, freq_max_id:, :]*0
        bm = tf * fdata #(360*257*8)*(257*8)
        bms = bm.sum(axis=2) #mic distance sum
        bmp = np.sqrt(bms.real ** 2 + bms.imag ** 2)
        self.bms = bms
        self.bmp = bmp

        # print("Succsess beamforming", bmp.shape)
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

    def checkDOA(self, frq):
        frq, t, Pxx = scipy.signal.stft(self.sound_data, self.w_sampling_rate)
        # print(frq.shape, t.shape, Pxx.shape)
        self.steering_vector(frq, 1)
        doa_data = []
        for i in range(t.shape[0]):
            bm_result_array, bms = self.beamformer_localization(Pxx[:, :, i], frq)
            doa_data.append(np.argmax(bm_result_array.sum(axis=1)))
            print(i)

        plt.plot(t, doa_data)
        plt.show()

    def executeBeamForming(self, wave_data, direction, plot=False):
        '''
        Use for make beamforming data
        :param direction:  your want direction
        :param plot:  if you want plot, write True, and you can plot spectrogram
        :return: direction data (sound kind num, 360, time), want direction data (sound kind num , araound 10 data avarge , time)
        '''
        combine_num = 4
        self.sound_data_func(wave_data)
        use_data = self.makeUsingSound(self.sound_data, self.w_sampling_rate, 10, 1, combine_num, 2)
        frq, t, Pxx = scipy.signal.stft(use_data, self.w_sampling_rate)
        print(frq.shape, t.shape, Pxx.shape)
        direction_data = np.zeros((combine_num, 360, int(t.shape[0])), dtype=np.complex)  # 実験データ×角度×時間のデータ
        # Pxx = 10 * np.log(np.abs(Pxx)) #対数表示に直す
        self.steering_vector(frq, combine_num)
        for i in range(t.shape[0]):
            '''ここで特定の周波数領域だけ取るのありかも'''
            for j ,name in enumerate(self.odigin_freq_list):
                freq_id = np.abs(frq - name).argmin()
                #print(freq_id)
                bm_result_array, bms = self.beamformer_localization(Pxx[j, :, :, i], frq)
                # print(bm_result_array.shape)
                direction_data[j, :, i] = np.mean(bm_result_array[:, freq_id - 3 : freq_id + 3], axis=1)

        if plot == True:
            X, Y = np.meshgrid(t, range(360))
            # print(X.shape, Y.shape)
            for i in range(combine_num):
                plt.contourf(X, Y, direction_data[i, :, :], cmap="jet", levels=np.arange(1000, 3500, 100))
                plt.colorbar()
                plt.show()

        # ほしい角度の周りプラスマイナス5°を切り取ったデータも作っておく
        want_direction_data = direction_data[:, direction - 10: direction + 10, :]
        want_direction_data = np.mean(want_direction_data, axis=1)
        intensity = np.mean(want_direction_data, axis = 1)
        print("make r data", intensity.shape)
        return direction_data, want_direction_data, intensity

    def makeUsingSound(self, sound_data, rate, interval_time, want_data_time, combine_num, start_time):
        print("Frame Rate : ", rate)
        print("Sound Data : ", sound_data.shape)

        use_sound_data = np.zeros((combine_num, sound_data.shape[0], want_data_time * rate), dtype=np.complex)
        start_time = start_time * rate
        for i in range(combine_num):
            use_sound_data[i, :, :] = sound_data[:, start_time: start_time + (want_data_time * rate)]

            start_time += interval_time * rate

        # print('sccsess make use data :', use_sound_data.shape)
        return use_sound_data

    def checkIntensity(self, true_direction, origin_freq_list, sign='plus', check_specgam=False, fft_plot=False, fft_plot2=False, check_specgam2=False):
        '''

        :param true_direction:
        :param origin_freq_list:
        :param sign:
        :param check_specgam:
        :param fft_plot:
        :return: (true direction(9), ss_list(4), origin_freq_list(4), mic_list(4))
        '''
        sfs = ShapeFromSound('./config.ini')
        file_path = '../_exp/190612/test_data/'
        mic_list = ['m1', 'm2', 'm3', 'm4']
        amp_array = np.zeros((len(sfs.ss_list),len(origin_freq_list), len(mic_list)), dtype=np.float)
        for k, name in enumerate(sfs.ss_list):
            # print(name)
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
            self.sound_data_func(wave_data)
            print(self.sound_data.shape)
            plt.specgram(self.sound_data[0, :], Fs=self.w_sampling_rate)
            plt.title('origin' + data_path + '_' + str(true_direction) + '.wav')
            plt.clim(0, 20)
            plt.colorbar()
            plt.show()
            print("##########################################")
            print('wave data:' +  wave_data)
            use_data = self.makeUsingSound(self.sound_data, self.w_sampling_rate, 10, 1, 4, 2)
            if check_specgam:
                for i, freq in enumerate(origin_freq_list):
                    for j, ss in enumerate(mic_list):
                        plt.specgram(use_data[i, j, :], Fs=self.w_sampling_rate)
                        plt.title('origin'+ str(freq) + '_' + ss + '_' + str(true_direction))
                        plt.clim(-20, 20)
                        plt.colorbar()
                        plt.show()

            size = 512
            hammingWindow = np.hamming(size)
            fs = 44100 # サンプリングレート
            d = 1.0 / fs
            start = int(fs/2)
            freqList = np.fft.fftfreq(size, d)
            # print(freqList)
            windowedData = hammingWindow + use_data[:, :, start:start + size] # 切り出し波形データ(窓関数)
            data = np.fft.fft(windowedData)
            if fft_plot:
                for i, freq in enumerate(origin_freq_list):
                    for j, ss in enumerate(mic_list):
                        plt.plot(freqList, abs(data[i, j, :]))
                        # plt.ylim(0, 2.0 * 10**7)
                        plt.xlim([0, fs/4])
                        plt.title('befor filter' + str(freq) + ss)
                        plt.show()

            freq_min = 1000
            freq_max = 9000
            id_min = np.abs(freqList - freq_min).argmin()
            id_max = np.abs(freqList - freq_max).argmin()
            fil = np.ones((size,), dtype=np.complex)
            fil[int(-1 * id_min):] = 0
            fil[id_max:int(-1 * id_max)] = 0
            fil[:id_min] = 0
            data = data * fil
            if fft_plot2:
                for i, freq in enumerate(origin_freq_list):
                    for j, ss in enumerate(mic_list):
                        plt.plot(freqList, abs(data[i, j, :]))
                        # plt.ylim(0, 2.0 * 10**7)
                        # plt.xlim([0, fs/4])
                        plt.title('after filter' + str(freq) + ss)
                        plt.savefig('../Image/190618/after_filter' + str(freq) + ss + '.png')
                        plt.show()

            ifft_data = np.fft.ifft(data)
            # print(ifft_data.shape)
            if check_specgam2:
                for i, freq in enumerate(origin_freq_list):
                    for j, ss in enumerate(mic_list):
                        plt.specgram(ifft_data[i, j, :], Fs=self.w_sampling_rate)
                        plt.title('ifft' + str(freq) + ss)
                        plt.clim(0, 20)
                        plt.colorbar()
                        plt.show()

            amp_list = np.zeros((len(origin_freq_list), len(mic_list)))
            for i, freq in enumerate(origin_freq_list):
                for j, ss in enumerate(mic_list):
                    amp = max([np.sqrt(c.real ** 2 + c.imag ** 2) for c in data[i, j, :]])
                    amp_id = np.argmax([np.sqrt(c.real ** 2 + c.imag ** 2) for c in data[i, j, :]])
                    amp_list[i, j] = amp
                    # print(amp_id)
            amp_array[k, :, :] = amp_list
        return amp_array

if __name__ == '__main__':
    bm = beamforming('./config.ini')
    direction_list = [0, 10, 20, 30, 40]
    amp_data = np.zeros((9, 4, 4, 4))
    for i, name in enumerate(direction_list):
        amp_data[i, :, :, :] = bm.checkIntensity(name, bm.odigin_freq_list)
    for i, name in enumerate(direction_list[1:]):
        amp_data[i+5, :, :, :] =bm.checkIntensity(name, bm.odigin_freq_list, sign='minus')
    np.save('../_array/amp_data', amp_data)
    print("success save")
