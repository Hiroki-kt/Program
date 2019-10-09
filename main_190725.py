import os
import numpy as np
from scipy import signal
from _beamforming import BeamForming
# from simulation_envs import Id
from matplotlib import pyplot as plt


def my_makedirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def reshape_sound_data(data, rate, interval_time, need_time, start_time, des_freq_list, time_range=0.1,
                       not_reshape=False):
    if not_reshape:
        return data
    # print('Sound Data:', data.shape)  # (Mic, Freq, Time)
    start_time = int(start_time * rate)
    time_range = int(time_range * rate / 2)
    use_sound_data = data[:, start_time + int(need_time * rate / 2) - time_range:
                             start_time + int(need_time * rate / 2) + time_range]
    start_time = start_time + int(need_time * rate) + int(interval_time * rate)
    for k in range(len(des_freq_list)):
        use_sound_data = np.append(use_sound_data,
                                   data[:, start_time + int(need_time * rate / 2) - time_range:
                                           start_time + int(need_time * rate / 2) + time_range],
                                   axis=1)
        start_time += int(need_time * rate) + int(interval_time * rate)
    print('#Complete reshape', use_sound_data.shape)
    return use_sound_data


def down_sampling(conversion_rate, data, fs):
    """
    ダウンサンプリングを行う．
    入力として，変換レートとデータとサンプリング周波数．
    ダウンサンプリング後のデータとサンプリング周波数を返す．
    """
    # 間引くサンプル数を決める
    decimationSampleNum = conversion_rate-1

    # FIRフィルタの用意をする
    nyqF = fs/2.0             # 変換後のナイキスト周波数
    cF = (fs/conversion_rate/2.0-500.)/nyqF     # カットオフ周波数を設定（変換前のナイキスト周波数より少し下を設定）
    taps = 511                                  # フィルタ係数（奇数じゃないとだめ）
    b = signal.firwin(taps, cF)           # LPFを用意
    
    # フィルタリング
    data = signal.lfilter(b, 1, data)

    # 間引き処理
    downData = []
    for i in range(0, len(data), decimationSampleNum+1):
        downData.append(data[i])
    return downData


if __name__ == '__main__':
    '''パラメータ初期値'''
    bm = BeamForming("./190722_config.ini")
    file_path = "../_exp/190714/recode_data/0_2"
    wave_path = file_path + ".wav"
    print("*****************************************")
    print("#Load sound data:", wave_path)
    INTERVAL_TIME = 3
    NEED_TIME = 0.2
    START_TIME = 3.2
    DESTINY_FREQUENCY_LIST = [250, 500, 1000, 2000, 3000]
    FFT_SAMPLING_FREQUENCY = 1
    CONVERSION_RATE = 4
    
    '''main'''
    sound_data, w_channel, w_sampling_rate, w_frames_num = bm.wave_read_func(wave_path)
    sound_data = np.delete(sound_data, [0, 5], 0)
    reshape_sound = reshape_sound_data(sound_data, w_sampling_rate, INTERVAL_TIME, NEED_TIME, START_TIME,
                                       DESTINY_FREQUENCY_LIST)
    plt.specgram(reshape_sound[0, :], Fs=w_sampling_rate)
    plt.show()
    down_data = np.zeros((sound_data.shape[0], int(reshape_sound.shape[1]/CONVERSION_RATE) + 1))
    for m in range(sound_data.shape[0]):
        down_d = down_sampling(CONVERSION_RATE, reshape_sound[m, :], w_sampling_rate)
        down_data[m, :] = down_d
    print("#Complete down sampling:", down_data.shape)
    frq, time, Pxx = signal.stft(down_data, fs=w_sampling_rate/CONVERSION_RATE, nfft=1024)
    power_spec = 10 * np.log(np.abs(Pxx))  # 対数表示に直す
    print("#Complete STFT")
    print("frq:", frq.shape, "time:", time.shape, "Pxx:", Pxx.shape)
    X, Y = np.meshgrid(time, range(360))
    plt.pcolormesh(time, frq, power_spec[0, ], cmap='jet')
    plt.colorbar()
    plt.show()
    tf = bm.steering_vector(frq, 1, sound_data.shape[0])
    bm_result = np.zeros((tf.shape[0], tf.shape[1], len(time)))
    for t in range(len(time)):
        beam_power, beam_sum = bm.beam_forming_localization(Pxx[:, :, t], tf, frq)
        # plt.plot(beam_power.sum(axis=1))
        # plt.legend()
        # plt.show()
        bm_result[:, :, t] = beam_power
    result_theta = bm_result.sum(axis=1)
    print(result_theta.shape)
    X, Y = np.meshgrid(time, range(360))
    plt.contourf(X, Y, result_theta, cmap='jet', levels=np.arange(5000, 20000, 100), extend="both")
    plt.colorbar()
    plt.show()
