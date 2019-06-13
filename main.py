#coding:utf-8
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from test_wave_file import WaveFft
import scipy.signal

from beamforming import beamforming

def executeBeamForming():
    bm = beamforming('./config.ini')
    use_data = makeUsingSound(bm.sound_data, bm.w_sampling_rate, 10, 1, 4, 2)
    frq, t, Pxx = scipy.signal.stft(bm.sound_data, bm.w_sampling_rate)
    # print(frq.shape, t.shape, Pxx.shape)
    direction_data = np.zeros((360, int(t.shape[0])), dtype=np.complex) #  角度×時間のデータ
    # Pxx = 10 * np.log(np.abs(Pxx)) #対数表示に直す
    bm.steering_vector(frq)
    for i in range(500):
        bm_result_array, bms = bm.beamformer_localization(Pxx[:, :, i])
        direction_data[:, i] = bm_result_array.sum(axis=1)

    print(direction_data.shape)
    # plot
    '''
    X, Y = np.meshgrid(t,range(360))
    print(X.shape, Y.shape)
    plt.contourf(X, Y, direction_data, cmap="jet")
    plt.xlim(0, 1.5)
    #plt.plot(t, bm_result.T)
    plt.show()
    '''

    return direction_data

def makeUsingSound(sound_data, rate, interval_time, want_data_time, combine_num, start_time):
    print("Frame Rate : " , rate)
    print("Sound Data : ", sound_data.shape)

    use_sound_data = np.zeros((combine_num, sound_data.shape[0], want_data_time * rate), dtype=np.complex)
    for i in range(combine_num):
        use_sound_data[i, :, :] = sound_data[:, start_time: start_time + (want_data_time * rate)]

        start_time += interval_time * rate

    print('sccsess make use data :', use_sound_data.shape)
    return  use_sound_data

if __name__ == '__main__':
    executeBeamForming()