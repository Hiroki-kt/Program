import numpy as np
import wave
from _function import MyFunc
from datetime import datetime
import struct
from matplotlib import pyplot as plt


def tsp_signal(time, fs, m=None):
    N = time * fs
    if m is None:
        m = N/4
    k_1 = np.arange(0, N/2)
    w_1 = 4j * np.pi * m * (k_1/N) ** 2
    
    k_2 = np.arange((N/2 - 1), (N-1))
    w_2 = -1 * 4j * np.pi * m * ((N - k_2)/N) ** 2
    
    w = np.array([w_1, w_2]).reshape((1, -1))
    H = np.exp(w)
    tsp_data = np.fft.ifft(H)
    tsp_data = tsp_data.real
    print(tsp_data)
    plt.figure()
    plt.plot(tsp_data[0], Fs=fs)
    plt.show()
    tsp_data = np.r_[tsp_data[0][int(m):], tsp_data[0][0:int(m)]]
    tsp_data = tsp_data[::-1]
    plt.figure()
    plt.specgram(tsp_data, Fs=fs)
    plt.show()
    plt.plot(tsp_data)
    plt.show()
    
    return tsp_data

    
def signal_data(length, fs, freq_list, interval_length, tsp=None):
    if tsp is not None:
        sound_samples = tsp.shape[0]
        s = np.rint(tsp)
        s = s.astype(np.int16)
        print(s.shape)
        s = s[0:sound_samples]
        single_data = struct.pack("h" * sound_samples, *s)  # ndarrayからbytesオブジェクトに変換
        single_data = 1
        print("TSP signal")
        return single_data
    else:
        samples = int(length * fs)  # サンプル数
        samples_rest = int(interval_length * fs)
        # sound_samples = int((1 + 2 * len(f_list)) * samples)
        sound_samples = int((1 + len(freq_list)) * samples + (len(freq_list)) * samples_rest)
        print('sound sample', sound_samples)
        print('sound second', sound_samples / fs)
        
        t = np.linspace(0, length, samples + 1)
        t_rest = np.linspace(0, interval_length, samples_rest + 1)
        s = 32767 * np.sin(2 * np.pi * 800 * t)  # テスト音
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

def pink_tsp():
    """
    pinktspGenerator.py
    Generate pinktsp signal.
    This code was originaly written by Atsushi Marui in Matlab,
    and ported to Python by Hidetaka Imamura.
    
    Copyright 2009 MARUI Atsushi
    Copyright 2015 Hidetaka Imamura
    """
    
    import primes
    import cmath
    import numpy as np
    from scipy.io import wavfile
    from utility import float2pcm
    
    # User settings
    dur = 5  # length of signal (seconds)
    fs = 48000  # number of samples per second (Hz)
    nbits = 16  # number of bits per sample (bit)
    reps = 4  # number of repeated measurements (times)
    
    N = 2 ** (nextpow2(dur * fs))
    m = primes.primes_below(N / 4)
    m = m[-1]
    
    a = 2 * m * np.pi / ((N / 2) * np.log(N / 2))
    j = cmath.sqrt(-1)
    pi = (cmath.log(-1)).imag
    
    H = np.array([1])
    H = np.hstack(
        [H, np.exp(j * a * np.arange(1, N / 2 + 1) * np.log(np.arange(1, N / 2 + 1))) / np.sqrt(np.arange(1, N / 2 + 1))])
    H = np.hstack([H, np.conj(H[int((N / 2 - 1)):0:-1])])
    h = (np.fft.ifft(H)).real
    mvBefore = np.abs(h)
    mv = min(mvBefore)
    mi = np.where(mvBefore == mvBefore.min())
    mi = int(mi[0])
    h = np.hstack([h[mi:len(h)], h[0:mi]])
    h = h[::-1]
    
    Hinv = np.array([1])
    Hinv = np.hstack([Hinv, np.exp(j * a * np.arange(1, N / 2 + 1) * np.log(np.arange(1, N / 2 + 1))) * np.sqrt(
        np.arange(1, N / 2 + 1))])
    Hinv = np.hstack([Hinv, np.conj(Hinv[int((N / 2 - 1)):0:-1])])
    hinv = (np.fft.ifft(Hinv)).real
    
    hh = np.hstack((np.tile(h, (reps, 1)).flatten(), np.zeros(len(h))))
    out = hh / max(np.abs(hh)) / np.sqrt(2)
    
    wavfile.write('pinktsp.wav', fs, float2pcm(out, 'int16'))
    
    plt.specgram(out, Fs=fs)
    plt.show()


def nextpow2(n):
    m_f = np.log2(n)
    m_i = np.ceil(m_f)
    return int(np.log2(2**m_i))

if __name__ == '__main__':
    # test()
    time = 1
    sampling = 44100
    tsp_signal(time, 44100)
    # file_path = '../_exp/19' + datetime.today().strftime("%m%d") + '/origin_data/'
    # MyFunc.my_makedirs(file_path)
    # FRAME = file_path + datetime.today().strftime("%H%M%S") + ".wav"
    # CHANNELS = 1  # チャンネル数
    # WIDTH = 2  # 量子化精度：2byte=16bit=65,536段階=±32,767
    # SAMPLING_RATE = 44100  # サンプリング周波数
    # AMPLITUDE = 0.8  # 振幅
    #
    # wf = wave.open(FRAME, 'w')
    # wf.setnchannels(CHANNELS)
    # wf.setsampwidth(WIDTH)
    # wf.setframerate(SAMPLING_RATE)
    # wf.writeframes(data)
    # wf.close()
    # print("Saved.")