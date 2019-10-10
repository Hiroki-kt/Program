# coding:utf-8
"""
非常に短い音を離散的に周波数の違う音を鳴らした場合の音データを確認する
パラメータ：(例)
    INTERVAL_TIME = 1
    NEED_TIME = 0.5
    START_TIME = 1.5
    DESTINY_FREQUENCY_LIST = [500, 1000, 2000]
    FREQ_SIZE = 1024 * 4
    DELECTION_LIST = list(range(-50, 50))
入力：反射音データ × len(DELECTION_LIST)
出力：グラフ
    1. x_周波数, y_RMS × MIC
    2. x_周波数, y_BPF + RMS × MIC
    3. x_周波数, y_BF + BPF + RMS
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
import math
from _beamforming import BeamForming
from _function import MyFunc as mfunc


def cut_data(num, bm):
    file_path = "../_exp/190920/recode_data/Right_data/" + str(num)
    wave_path = file_path + ".wav"
    INTERVAL_TIME = 1
    NEED_TIME = 0.5
    START_TIME = 1.5
    DESTINY_FREQUENCY_LIST = [500, 1000, 2000]
    FREQ_SIZE = 1024 * 4
    
    '''main'''
    sound_data, w_channel, w_sampling_rate, w_frames_num = bm.wave_read_func(wave_path)
    sound_data = np.delete(sound_data, [0, 5], 0)
    reshape_sound = mfunc.reshape_sound_data(sound_data, w_sampling_rate, INTERVAL_TIME, NEED_TIME, START_TIME,
                                             DESTINY_FREQUENCY_LIST)
    '''
    cut = [deg(1), MIC(4), Freq(3), Time(44100/0.1s)]
    '''
    cut = np.reshape(reshape_sound, (1, reshape_sound.shape[0], len(DESTINY_FREQUENCY_LIST), -1))
    rms_array = np.zeros((1, cut.shape[1], cut.shape[2]), dtype=np.float)
    for k in range(cut.shape[2]):
        for j in range(cut.shape[1]):
            rms_array[0, j, k] = mfunc.rms(cut[0, j, k, :].tolist())
      
    bpf_rms_array = np.zeros((1, cut.shape[1], cut.shape[2]), dtype=np.float)
    freq_list = np.fft.fftfreq(FREQ_SIZE, 1/w_sampling_rate)
    tf = bm.steering_vector(freq_list, 1, cut.shape[1])  # tf = [deg(360), freq(1024?), MIC(4)]
    bm_array = np.zeros((1, cut.shape[1]))
    for k, freq in enumerate(DESTINY_FREQUENCY_LIST):
        fft_array = np.zeros((cut.shape[1], FREQ_SIZE), dtype=np.complex)
        for j in range(cut.shape[1]):
            # bpf
            fft_array[j, :], ifft_bpf = mfunc.band_pass_filter(FREQ_SIZE, w_sampling_rate, 0, cut[0, j, k, :],
                                                               freq - 50, freq + 50)
            bpf_rms_array[0, j, k] = mfunc.rms(ifft_bpf.real.tolist())
            # fft_array[j, :] = fft_bpf
        # bm_array[k, :, :] = bm.beam_forming_localization(fft_array, tf, freq_list)
        # beam_data = bm.beam_forming_localization(fft_array, tf, freq_list)[0]
        bm_array[0, k] = mfunc.rms(bm.beam_forming_localization(fft_array, tf, freq_list)[0].sum(0))
    return cut, rms_array, bpf_rms_array, bm_array


if __name__ == '__main__':
    beam = BeamForming("./config_rsj2019.ini")
    DELECTION_LIST = list(range(-50, 50))
    exp_data, rms_data, bpf_rms_data, bm_data = cut_data(DELECTION_LIST[0], beam)
    for i in DELECTION_LIST[1:]:
        exp, exp_rms, exp_bpf_rms, exp_bm = cut_data(i, beam)
        # print("OK")
        print("NUM:", str(i))
        # exp_data = np.append(exp_data, exp, axis=0)
        rms_data = np.append(rms_data, exp_rms, axis=0)
        bpf_rms_data = np.append(bpf_rms_data, exp_bpf_rms, axis=0)
        bm_data = np.append(bm_data, exp_bm, axis=0)
    # a = np.average(exp_data, axis=3)
    # print(bm_data.shape)
    '''
    フィルタも何もかけずにマイクで取得した音の振幅の比
    rms_data = [deg(100), MIC(4), Freq(3(500, 1000, 2000))]
    '''
    TITLE = "1000Hz, 2000Hz rate using data that no filter, only amp(RMS result) "
    Y_LABEL = "$e_1000$/$e_2000$"
    X_LABEL = "deg"
    MIC = 1
    e_rms_rate = rms_data[:, MIC, 0]/rms_data[:, MIC, 2]
    # mfunc.data_plot(deg_list, e_rms_rate, TITLE, X_LABEL, Y_LABEL)
    # plt.show()
    
    '''
    次はフィルタをかけて、
    '''
    TITLE2 = "1000Hz, 2000Hz rate using data that BPF filter, only amp(RMS result) "
    e_bpf_rms_rate = bpf_rms_data[:, MIC, 0]/bpf_rms_data[:, MIC, 2]
    # mfunc.data_plot(deg_list, e_bpf_rms_rate, TITLE2, X_LABEL, Y_LABEL)
    # plt.show()
    
    '''
    ビームフォーマを用いて、到達音源方向を制限する
    '''
    TITLE3 = "1000Hz, 2000Hz rate using data that Beam ,BPF and amp(RMS result) "
    e_bm_rate = bm_data[:, 0]/bm_data[:, 2]
    mfunc.data_plot(DELECTION_LIST, e_bm_rate, TITLE3, X_LABEL, Y_LABEL)
    plt.show()