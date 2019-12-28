# coding:utf-8

import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from test_wave_file import WaveFft
import math

if __name__ == '__main__':
    '''
    パラメータ
    '''
    size = 1024
    # 窓関数はハミング
    hammingWindow = np.hamming(size)
    fs = 44100
    d = 1 / fs
    freqList = np.fft.fftfreq(size, d)
    second_list = [0, 4, 8, 12, 16, 20, 24, 28]
    direction_list = [0, 30, 50, 70]
    second = 2

    '''
    初期値
    '''
    test_array = np.zeros((len(direction_list), len(second_list), int(4 * fs * second / size), size),
                          dtype=np.complex64)
    count = 0
    maxid2_array = np.zeros((len(direction_list), len(second_list)), dtype=np.complex64)
    maxid1_array = np.zeros_like(maxid2_array)
    '''
    main
    '''
    direct_sound_file_name = '../Experiment/190509/Test_wav/only_direct2.wav'
    direct_fft = WaveFft(direct_sound_file_name)
    direct_stft_data = direct_fft.stft(size, hammingWindow)

    for j in second_list:
        direct_x = direct_fft.separateSound(direct_stft_data, second, j, 1, False)
        print(direct_x.shape)

    for i in direction_list:
        file_name = '../Experiment/190509/Test_wav/ref_25_' + str(i) + '.wav'
        fft = WaveFft(file_name)

        stft_data = fft.stft(size, hammingWindow)
        for j in second_list:
            x = fft.separateSound(stft_data, second, j, 1, False)
            # x = x - direct_x
            test_array[count, int(j / 4), :, :] = x
            avg = np.average(abs(x), axis=0)

            ### 一旦、先に調べておいた音がなったいる周波数で切った。
            maxid1 = signal.argrelmax(avg[15:23], order=100)[0]  # 低い方
            maxid2 = signal.argrelmax(avg[27:36], order=100)[0]  # 高い方
            # print(i, j/4, maxid1, maxid2)
            maxid1_array[count, int(j / 4)] = avg[15:23][maxid1]
            maxid2_array[count, int(j / 4)] = avg[27:36][maxid2]
        count += 1

    print("===================================")
    print("success make array")
    print("===================================")

    '''
    プロット
    '''
    ### 正解値 cos(r - 30)
    y = [math.cos(math.radians(k) - math.radians(30)) for k in direction_list]
    print(y)
    ### 計測値
    for i in second_list:
        plt.plot(direction_list, np.float64(maxid2_array[:, int(i / 4)]), label="Experiment")
        z = [(maxid2_array[1, int(i / 4)]) * k for k in y]
        plt.plot(direction_list, z, label="Prediction")
        plt.title("Test result_" + str(i) + "s")
        plt.ylabel("Amplitude")
        plt.xlabel("Direction[°]")
        plt.ylim(0, 6000)
        plt.legend()
        plt.show()
        # save_name = "save_" + str(i) + "s.png"
        # plt.savefig(save_name)
