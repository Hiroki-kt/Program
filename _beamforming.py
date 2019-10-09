# coding:utf-8
import configparser
import math
import os
import sys
import wave
import matplotlib.pyplot as plt
import scipy.signal
import numpy as np

from _beamforming_sub import SimulationEnvs

'''
beamformer class by python 3
configparser depend on python 3
'''


class BeamForming(SimulationEnvs):
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
            origin_freq_list = []
            for k in range(self.combine_num):
                freq_id = 'original' + str(k + 1) + '_freq'
                origin_freq_list.append(float(config['SoundParam'][freq_id]))
            self.origin_freq_list = origin_freq_list
            
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
            # sound_method = config['sound']['sound_method']
            # sound_r = float(config['sound']['sound_r'])
            # sound_dir = float(config['sound']['sound_dir'])
            self.wave_data = config['sound']['wave_data']
            # self.sound_data_func(wave_data, sound_method, sound_r, sound_dir)
            # print('#Success parse config param')
            
            # self.freq_list = np.fft.rfftfreq(self.FFT_sample_num, d=1. / self.w_sampling_rate)
            # self.freq_min_id, self.freq_max_id = self.freq_id()
        
        else:
            print("#Couldn't find", config_path)
            sys.exit()
    
    def check_beam_forming(self, wave_path, psi_deg, semiauto_data=False, test=False, sound_r=10., sound_dir=90.,
                           plot_beam=False, uniform_dist=None):
        sound_data, w_channel, w_sampling_rate, w_frames_num = self.wave_read_func(wave_path)
        if test:
            sound_data = self.create_mic_input(w_sampling_rate, sound_data, w_frames_num, sound_r, sound_dir)
            print('#Create microphone array input', sound_data.shape)
        if semiauto_data:
            sound_data = np.delete(sound_data, [0, 5], 0)
        # use_data = reshape_sound_data(sound_data, w_sampling_rate, 0, 14, 1, 0)
        # use_data = np.reshape(use_data, (4, -1))
        # plt.specgram(sound_data[0, :], Fs=w_sampling_rate)
        # plt.show()
        # use_data = reshape_sound_data_freq_time(sound_data, w_sampling_rate, 0.4, 0.1, 1.5, 100, 2500, 20)
        use_data = sound_data[:, int(2 * w_sampling_rate):]
        frq, time, Pxx = scipy.signal.stft(use_data, w_sampling_rate)
        tf = self.steering_vector(frq, 1, use_data.shape[0])
        # freq_time = (2000 - 100) / (14 - 0) * time + 100
        bm_time = np.zeros((len(frq), len(time)), dtype=np.float)
        # print(bm_time.shape)
        for t in range(len(time)):
            # id_time = np.abs(time - dis_time).argmin()
            bm_result_array, bms = self.beam_forming_localization(Pxx[:, :, t], tf, frq)
            if uniform_dist:
                bm_result_array[:uniform_dist[0], :] = 0
                bm_result_array[uniform_dist[1]:, :] = 0
            
            bmp_freq = bm_result_array.sum(axis=0)  # 周波数の数
            bmp_deg = bm_result_array.sum(axis=1)  # 360
            # print(bmp_freq.shape)
            bm_time[:, t] = bmp_freq
            
            if plot_beam:
                x = np.cos(np.deg2rad(self.ss_theta_list))
                y = np.sin(np.deg2rad(self.ss_theta_list))
                x_q = np.linspace(-20, 20)
                y_q = -1 * np.tan(math.radians(psi_deg)) * x_q + 20
                normalize_data = (bmp_deg - min(bmp_deg)) / (max(bmp_deg) - min(bmp_deg))
                x_s = 10 * normalize_data * x + 5
                y_s = 10 * normalize_data * y
                plt.plot(x_s, y_s)
                plt.plot(x_q, y_q, label='Reflection object')
                plt.scatter(5, 0, label='MIC')
                plt.scatter(-5, 0, label='Speaker')
                plt.title('Direction of beam $\psi$ = ' + str(psi_deg) + ', '
                                                                         'freq = ' + str(int(time[t])) + '[Hz]')
                plt.ylim(-6, 27)
                # ax.plot(np.deg2rad(self.ss_theta_list), normalize_data)
                plt.legend()
                # plt.savefig('../_img/190703/direction_of_beam/' + str(int(time[t])) + '_' + str(psi_deg))
                plt.show()
            
            # ifft_bpf_data = scipy.signal.istft(bmp_freq, w_frames_num)
        bm_time = bm_time.sum(axis=0)
        print(time.shape)
        return bm_time, time
    
    def steering_vector(self, freq_array, combine_num, w_channel):
        # freq_array = self.freq_list
        freq_num = freq_array.shape[0]
        # print(freq_num)
        temp_ss_num = len(self.ss_theta_list)
        
        tf = np.zeros((temp_ss_num, freq_num, w_channel), dtype=np.complex)
        # beam_conf = np.zeros((temp_ss_num, freq_num, w_channel), dtype=np.complex)
        new_tf = np.zeros((temp_ss_num, freq_num, w_channel, combine_num), dtype=np.complex)
        
        # create tf
        # l_w = math.pi * freq_array / self.sound_speed
        mic_x_list = []
        mic_y_list = []
        for mic_p in self.mic_pos_list:
            x, y = mic_p.pos()
            mic_x_list.append(x)
            mic_y_list.append(y)
        mic_x_array = np.array(mic_x_list)
        mic_y_array = np.array(mic_y_list)
        
        freq_repeat_array = np.ones((freq_num, w_channel), dtype=np.complex) * freq_array.reshape(
            (freq_num, -1)) * -1j * 2 * np.pi  # ??
        
        for k, ss_pos in enumerate(self.ss_pos_list):
            sx, sy = ss_pos.pos()
            center2ss_dis = math.sqrt(sx ** 2 + sy ** 2)
            mic2ss_dis = np.sqrt((mic_x_array - sx) ** 2 + (mic_y_array - sy) ** 2)
            dis_diff = (mic2ss_dis - center2ss_dis) / self.sound_speed  # * self.w_sampling_rate 打消
            dis_diff_repeat_array = np.ones((freq_num, w_channel)) * dis_diff.reshape((-1, w_channel))
            tf[k, :, :] = np.exp(freq_repeat_array * dis_diff_repeat_array)
            # beam_conf[k,:,:] = tf[k,:,:]/ ()
        # print('#Create transfer funtion', tf.shape)
        tf = tf.conj()  # 360*257*8
        if combine_num == 1:
            return tf
        else:
            for k in range(combine_num):
                new_tf[:, :, :, k] = tf
            return new_tf
    
    def beam_forming_localization(self, f_data, tf, freq_list):
        f_data = f_data.transpose(1, 0)
        # tf = tf[:, self.freq_min_id:self.freq_max_id + 1, :]#360*257*8
        freq_min_id, freq_max_id = self.freq_id(freq_list)
        tf[:, :freq_min_id, :] = tf[:, :freq_min_id, :] * 0
        tf[:, freq_max_id:, :] = tf[:, freq_max_id:, :] * 0
        bm_data = tf * f_data  # (360*257*8)*(257*8)
        bms = bm_data.sum(axis=2)  # mic distance sum
        bmp = np.sqrt(bms.real ** 2 + bms.imag ** 2)
        # self.bms = bms
        # self.bmp = bmp
        # print("Succsess beamforming", bmp.shape)
        return bmp, bms  # 360*257
    
    def direction_of_arrival(self, sound_data, w_sampling_rate, w_channel):
        frq, time, Pxx = scipy.signal.stft(sound_data, w_sampling_rate)
        tf = self.steering_vector(frq, 1, w_channel)
        doa_data = []
        for t in range(time.shape[0]):
            bm_result_array, bms = self.beam_forming_localization(Pxx[:, :, t], tf, frq)
            doa_data.append(np.argmax(bm_result_array.sum(axis=1)))
            print(t)
        
        plt.plot(time, doa_data)
        plt.show()
        
    def create_mic_input(self, w_sampling_rate, sound_data, w_frames_num, sound_r, sound_dir, ch=1):
        s_theta = sound_dir * math.pi / 180.
        s_x = sound_r * math.cos(s_theta)
        s_y = sound_r * math.sin(s_theta)
        center2sound_dis = math.sqrt(s_x ** 2 + s_y ** 2)
        delay_list = []
        for mic in self.mic_pos_list:
            mx, my = mic.pos()
            mic2sound_dis = math.sqrt((mx - s_x) ** 2 + (my - s_y) ** 2)
            delay_point = round((mic2sound_dis - center2sound_dis) / self.sound_speed * w_sampling_rate)
            delay_list.append(delay_point)

        delay_min, delay_max = min(delay_list), max(delay_list)
        target_sound_data = sound_data[ch - 1, :]
        sound_fnum = w_frames_num + delay_max - delay_min
        data = np.zeros((len(self.mic_pos_list), sound_fnum))
        for n in range(len(self.mic_pos_list)):
            data[n, -w_frames_num + delay_list[n] + delay_min:w_frames_num + delay_list[
                n] - delay_min] = target_sound_data
        # w_frames_num = sound_fnum
        # w_channel = len(self.mic_pos_list)
        return data

    def freq_id(self, freq_list):
        id_min = np.abs(freq_list - self.freq_min).argmin()
        id_max = np.abs(freq_list - self.freq_max).argmin()
        return id_min, id_max
    
    def execute_beam_forming(self, wave_path, direction, plot=False):
    
        """

        Use for make beamforming data
        :param wave_path: path
        :param direction:  your want direction
        :param plot:  if you want plot, write True, and you can plot spectrogram
        :return: direction data (sound kind num, 360, time),
                want direction data (sound kind num , araound 10 data avarge , time)
        """
        combine_num = 4
        sound_data, w_channel, w_sampling_rate, w_frames_num = self.wave_read_func(wave_path)
        use_data = reshape_sound_data(sound_data, w_sampling_rate, 10, 1, combine_num, 2)
        frq, time, Pxx = scipy.signal.stft(use_data, w_sampling_rate)
        print(frq.shape, time.shape, Pxx.shape)
        direction_data = np.zeros((combine_num, 360, int(time.shape[0])), dtype=np.complex)  # 実験データ×角度×時間のデータ
        # Pxx = 10 * np.log(np.abs(Pxx)) #対数表示に直す
        tf = self.steering_vector(frq, combine_num, w_channel)
        for t in range(time.shape[0]):
            '''ここで特定の周波数領域だけ取るのありかも'''
            for j, origin_freq in enumerate(self.origin_freq_list):
                freq_id = np.abs(frq - origin_freq).argmin()
                # print(freq_id)
                bm_result_array, bms = self.beam_forming_localization(Pxx[j, :, :, t], tf, frq)
                # print(bm_result_array.shape)
                direction_data[j, :, t] = np.mean(bm_result_array[:, freq_id - 3: freq_id + 3], axis=1)
        
        if plot:
            X, Y = np.meshgrid(time, range(360))
            # print(X.shape, Y.shape)
            for n in range(combine_num):
                plt.contourf(X, Y, direction_data[n, :, :], cmap="jet", levels=np.arange(1000, 3500, 100))
                plt.colorbar()
                plt.show()
        
        # ほしい角度の周りプラスマイナス5°を切り取ったデータも作っておく
        want_direction_data = direction_data[:, direction - 10: direction + 10, :]
        want_direction_data = np.mean(want_direction_data, axis=1)
        intensity = np.mean(want_direction_data, axis=1)
        print("make r data", intensity.shape)
        return direction_data, want_direction_data, intensity


def reshape_sound_data(sound_data, rate, interval_time, want_data_time, combine_num, start_time):
    # print("Frame Rate : ", rate)
    # print("Sound Data : ", sound_data.shape)
    
    use_sound_data = np.zeros((combine_num, sound_data.shape[0], want_data_time * rate), dtype=np.complex)
    start_time = start_time * rate
    for k in range(combine_num):
        use_sound_data[k, :, :] = sound_data[:, start_time: start_time + (want_data_time * rate)]
        
        start_time += interval_time * rate
    
    # print('sccsess make use data :', use_sound_data.shape)
    return use_sound_data


def reshape_sound_data_freq_time(sound_data, rate, interval_time, want_data_time, start_time, min_freq, max_freq, step):

    f_list = np.linspace(min_freq, max_freq, int((max_freq - min_freq) / step))
    start_time = int(start_time * rate)
    print('Sound data:', sound_data.shape)
    print('f_list', len(f_list))
    use_sound_data = sound_data[:, start_time:start_time + int(want_data_time * rate)]
    start_time = start_time + int(interval_time * rate)
    for k in range(len(f_list)):
        use_sound_data = np.append(use_sound_data, sound_data[:, start_time:start_time + int(want_data_time * rate)],
                                   axis=1)
        start_time += int(interval_time * rate)
    print('Complete reshape', use_sound_data.shape)
    return use_sound_data


def wave_save(sep_data, number, path, w_sampling_rate):
    sep_data = sep_data.astype(np.int16)
    w = wave.Wave_write("./sep_data/" + str(number) + "/" + path + ".wav")
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(w_sampling_rate)
    w.writeframes(b''.join(sep_data))
    w.close()
    print('finish saving')


def my_makedirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)


if __name__ == '__main__':
    bm = BeamForming('./config.ini')

