import numpy as np
import wave
import struct
import os
from datetime import datetime
from matplotlib import pyplot as plt
from _function import MyFunc


def my_makedirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def single_tone(length, interval_length, fs, freq_list):
    samples = int(length * fs)  # サンプル数
    samples_rest = int(interval_length * fs)
    # sound_samples = int((1 + 2 * len(f_list)) * samples)
    # sound_samples = int((1 + len(freq_list)) * samples + (len(freq_list)) * samples_rest)
    sound_samples = int((len(freq_list)) * samples + (len(freq_list)) * samples_rest)
    print('sound sample', sound_samples)
    print('sound second', sound_samples / fs)
    
    t = np.linspace(0, length, samples + 1)
    t_rest = np.linspace(0, interval_length, samples_rest + 1)
    # s = 32767 * np.sin(2 * np.pi * 800 * t)  # テスト音
    s = np.empty((1, ))
    for i, name in enumerate(freq_list):
        s_1 = 0.7 * 32767 * np.sin(2 * np.pi * name * t)
        rest = np.zeros((len(t_rest),))
        s = np.append(s, rest)
        s = np.append(s, s_1)
    s = np.rint(s)
    s = s.astype(np.int16)
    print(s.shape)
    
    s = s[0:sound_samples]
    single_data = struct.pack("h" * sound_samples, *s)  # ndarrayからbytesオブジェクトに変換
    
    return single_data

def create_chord(amp, freq_list, fs, length):
    """freqListの正弦波を合成した波を返す"""
    chord_data = []
    amp = float(amp) / len(freq_list)
    # [-1.0, 1.0]の小数値が入った波を作成
    for n in range(int(length * fs)):  # nはサンプルインデックス
        s = 0.0
        for f in freq_list:
            s += amp * np.sin(2 * np.pi * f * n / fs)
        # 振幅が大きい時はクリッピング
        if s > 1.0:  s = 1.0
        if s < -1.0: s = -1.0
        chord_data.append(s)
    # [-32768, 32767]の整数値に変換
    chord_data = [int(x * 32767.0) for x in chord_data]
    print(len(chord_data))
    # バイナリに変換
    chord_data = struct.pack("h" * len(chord_data), *chord_data)  # listに*をつけると引数展開される
    
    return chord_data

def tsp_signal(tsp_data, sampling, plus_time, rest_time):
    print(tsp_data.shape)
    plus_frames = plus_time * sampling
    print(int(tsp_data.shape[1] % plus_frames))
    data_frames = int(tsp_data.shape[1] - (tsp_data.shape[1] % plus_frames) + plus_frames)
    print(data_frames)
    tsp_data = np.append(tsp_data, np.zeros((1, int(plus_frames - (tsp_data.shape[1] % plus_frames)))))
    print(tsp_data.shape)
    tsp_data = np.reshape(tsp_data, (int(tsp_data.shape[0]/plus_frames), -1))
    print(tsp_data.shape)
    rest_data = np.zeros((tsp_data.shape[0], int(rest_time * sampling)))
    print(rest_data.shape)
    data = np.c_[tsp_data, rest_data]
    print(data.shape)
    data = np.reshape(data, (1, -1))
    print(data.shape)

    s = np.rint(data[0])
    s = s.astype(np.int16)
    print(s.shape)

    s = s[0:s.shape[0]]
    single_data = struct.pack("h" * s.shape[0], *s)  # ndarrayからbytesオブジェクトに変換

    return single_data

if __name__ == '__main__':
    file_path = '../_exp/19' + datetime.today().strftime("%m%d") + '/origin_data/'
    my_makedirs(file_path)
    FRAME = file_path + datetime.today().strftime("%H%M%S") + ".wav"
    CHANNELS = 1  # チャンネル数
    WIDTH = 2  # 量子化精度：2byte=16bit=65,536段階=±32,767
    SAMPLING_RATE = 44100  # サンプリング周波数
    AMPLITUDE = 0.8  # 振幅

    # f_list = (500, 1000, 5000)  # 欲しい音の高さ
    f_list = np.arange(500, 4500, 100)  # だんだん高くしていくバージョン
    time = 0.05  # 録音時間
    time_rest = 0.05  # 単音を連続で流す場合のインターバル
    data = single_tone(time, time_rest, SAMPLING_RATE, f_list)

    # tsp_origin_file = "./tsp_origin.wav"
    # tsp_origin_data, channels, samlpling, frames = MyFunc().wave_read_func(tsp_origin_file)
    # tsp_origin_data = tsp_origin_data[::-1]
    # data = tsp_signal(tsp_origin_data, samlpling, 0.05, 0.05)
    # plt.figure()
    # plt.plot(data)
    # plt.show()
    # data = create_chord(AMPLITUDE, f_list, SAMPLING_RATE, time)
    
    wf = wave.open(FRAME, 'w')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(WIDTH)
    wf.setframerate(SAMPLING_RATE)
    wf.writeframes(data)
    wf.close()
    print("Saved.")