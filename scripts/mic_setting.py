# coding:utf-8
import numpy as np
from _function import MyFunc
from beamforming_sound_directions import SoundDirections
from _tf_generate_tsp import TSP
import sys
import math
from scipy import signal
from matplotlib import pyplot as plt


class MicSet(MyFunc):
    def __init__(self, mic_kind, image=False):
        super().__init__()
        if mic_kind == 'Respeaker':
            mic_radius = 0.03
            self.mic_channel_num = 4
            start_mic_no1 = 45
            clockwise = False
            
        elif mic_kind == 'Matrix':
            mic_radius = 0.05025
            self.mic_channel_num = 8
            start_mic_no1 = -90 + 45/2
            clockwise = True
        
        else:
            print('Do not have' + mic_kind + 'setting')
            sys.exit()
        
        self.mic_pos_list = self.mic_positions(mic_radius, self.mic_channel_num, start_mic_no1, clockwise)
        sd = SoundDirections()
        if image:
            print("Test 2D")
            self.ss_list = sd(bm_image=True)
            # self.steering_vector_azimuth(n, d)
        else:
            print("Test Vertical")
            self.ss_list = sd(bm_azimuth=True)
        
        self.sound_speed = 340

    def steering_vector_azimuth(self, n, d, use_list=None):
        if use_list is not None:
            freq_array=use_list
        else:
            freq_array = np.fft.rfftfreq(n, d)
        tf = np.zeros((self.ss_list.shape[0], freq_array.shape[0], self.mic_channel_num), dtype=np.complex)
        # beam_conf = np.zeros((temp_ss_num, freq_num, w_channel), dtype=np.complex)
    
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
    
        freq_repeat_array = np.ones((freq_array.shape[0], self.mic_channel_num), dtype=np.complex) * freq_array.reshape(
            (freq_array.shape[0], -1)) * -1j * 2 * np.pi  # フーリエ変換
    
        for k, ss_pos in enumerate(self.ss_list):
            sx, sy = ss_pos.pos()
            center2ss_dis = math.sqrt(sx ** 2 + sy ** 2)
            mic2ss_dis = np.sqrt((mic_x_array - sx) ** 2 + (mic_y_array - sy) ** 2)
            dis_diff = (mic2ss_dis - center2ss_dis) / self.sound_speed  # * self.w_sampling_rate 打消
            dis_diff_repeat_array = np.ones((freq_array.shape[0], self.mic_channel_num)) * \
                                    dis_diff.reshape((-1, self.mic_channel_num))
            tf[k, :, :] = np.exp(freq_repeat_array * dis_diff_repeat_array)
            # beam_conf[k,:,:] = tf[k,:,:]/ ()
        # print('#Create transfer funtion', tf.shape)
        tf = tf.conj()  # 360*257*8
        return tf

    def beam_forming_localization(self, f_data, tf, freq_list):
        f_data = f_data.transpose(1, 0)
        # tf = tf[:, self.freq_min_id:self.freq_max_id + 1, :]#360*257*8
        freq_min_id = self.freq_ids(freq_list, 800)
        freq_max_id = self.freq_ids(freq_list, 2200)
        tf[:, :freq_min_id, :] = tf[:, :freq_min_id, :] * 0
        tf[:, freq_max_id:, :] = tf[:, freq_max_id:, :] * 0
        bm_data = tf * f_data  # (360*257*8)*(257*8)
        bms = bm_data.sum(axis=2)  # mic distance sum
        bmp = np.sqrt(bms.real ** 2 + bms.imag ** 2)
        # self.bms = bms
        # self.bmp = bmp
        # print("Succsess beamforming", bmp.shape)
        return bmp  # 360*257
        
        
if __name__ == '__main__':
    mic_type = 'Respeaker'
    tsp = TSP('./config_tf.ini')
    ms = MicSet(mic_type)
    wave_path = ms.recode_data_path + ms.data_search(191015, 'Ts', '01', plane_wave=False) + '10.wav'
    data, channels, sampling, framse = ms.wave_read_func(wave_path)
    cut_data = tsp.cut_up_tsp_data(use_data=data)
    FRAME = 1024
    tf = ms.steering_vector_azimuth(FRAME, 1/sampling)
    frq, time, Pxx = signal.stft(cut_data, sampling, nperseg=FRAME)
    # print(tf.shape, len(frq), Pxx.shape)
    for t in time:
        bm_result = ms.beam_forming_localization(Pxx[:, :, int(t)], tf, frq)
        bmp = bm_result.sum(axis=1)
        # print(bmp.shape)
        x = np.cos(np.deg2rad(range(360)))
        y = np.sin(np.deg2rad(range(360)))
        x_s = bmp * x * 10
        y_s = bmp * y * 10
        plt.figure()
        plt.plot(x_s, y_s)
        plt.plot(0, 0, 'ro')
        plt.plot(np.linspace(0, 0, 7), range(-3, 4), 'r')
        plt.plot(range(-3, 4), np.linspace(0, 0, 7), 'r')
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        plt.show()