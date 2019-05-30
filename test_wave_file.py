#coding:utf-8
#!/usr/bin/env python

'''
WavファイルのデータをFFTして、周波数ごとの音の大きさをプロットするプログラム

→したいこと
* 元データを自動で区切りたい。 OK
* 正解値を表示したい
* 直接音のみの音を引く
'''

import wave
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

class WaveFft():
    def __init__(self, filename):
        self.filename = filename
        pass

    # fileをほしいwaveデータを１つにしたデータにする？
    def waveLoad(self):

        '''
        wavファイルを読み込む
        :return:音データ
        '''
        # waveファイルオープン
        # print(self.filename)
        waveFile = wave.open(self.filename, 'r')
        # print(waveFile.getparams())

        # パラメータ確認
        channels = waveFile.getnchannels()
        width = waveFile.getsampwidth()
        self.rate = waveFile.getframerate()
        self.frame = waveFile.getnframes()
        self.time = self.frame/self.rate

        print("****************************")
        print("ファイル名:", self.filename)
        print("チャンネル:", channels)
        print("総フレーム数:", self.frame)
        print("時間:", self.time)

        # バイナリ読み込み
        buf = waveFile.readframes(self.frame)
        # print(buf)

        # バイナリデータを整数型に変換
        data = np.frombuffer(buf, dtype="int16")
        # print(data)

        # 振幅で正規化しないほうがいいかな？笑 全然違う音で比べる場合はしたほうがいいが今回は同じ音
        # しかも、この振幅自体の差が知りたい。
        # チャンネル数がある場合にチャンネルごとに分ける。
        amp = (2 ** 8) ** width / 2
        # data = data / amp
        data = data[::channels]

        return data

    def fft(self, start, size):

        '''
        単純な高速フーリエ変換
        count = グラフに入れたい数
        size = FFTのサンプル数(2**n)
        :return: FFTデータ
        '''

        st = start # サンプリングする位置
        hammingWindow = np.hamming(size)
        fs = 44100 # サンプリングレート
        d = 1.0 / fs
        freqList = np.fft.fftfreq(size, d)
        # print(freqList)
        wave = self.waveLoad()
        windowedData = hammingWindow + wave[st:st + size] # 切り出し波形データ(窓関数)
        data = np.fft.fft(windowedData)
        # data = data / max(abs(data))
        '''
        plt.plot(freqList, abs(data))
        # plt.axis([0, fs/16, 0, 1])
        # plt.xlim(0, fs/16)
        plt.xlim(0, 1500)
        plt.ylim(200, 3000)
        plt.xlabel("Frequency[Hz]")
        plt.ylabel("amplitude spectrum")
        plt.show()
        '''

    def stft(self, size, win):

        '''
        短区間フーリエ変換
        :param size: 区切り幅
        :return:
        '''

        wave = self.waveLoad()
        l = len(wave) # 入力信号の長さ

        # step：シフト幅は切り出す幅の半分とすることが多い
        self.step = size / 4

        self.N = len(win) # 窓幅
        self.M = int(np.ceil(float(l - self.N + self.step)/ self.step)) #スペクトログラムの時間フレーム数
        self.frame_per_time = self.M/self.time # 1sのフレーム数

        print("l:" + str(l))
        print("N:" + str(self.N))
        print("M:" + str(self.M))
        print("step:" + str(self.step))
        print("frame/s:" + str(self.frame_per_time))
        print("****************************")

        new_x = np.zeros(int(self.N + ((self.M-1) * self.step)), dtype = np.float64)
        new_x[:l] = wave # 区切った後に長さがちょうど良くなるように

        X = np.zeros((self.M, self.N), dtype = np.complex64) # スペクトログラムの初期化（複素数）
        sum_list = []
        threshold = 10000
        for m in range(self.M):
            start = int(self.step * m)
            X[m, :] = np.fft.fft(new_x[start : start + self.N] * win)
            sum_list.append(abs(X[m, 10:40]).sum())


        # スペクトログラム表示
        '''
        # plt.imshow(abs(X[:, : int(size / 2) + 1].T), aspect="auto", origin="lower")
        # 実験時の音は大体、10 < N < 40　で音が周期的になっている。↓↓
        plt.imshow(abs(X[:, 27:36].T), aspect="auto", origin="lower")
        # plt.imshow(abs(X[:, 15:23].T), aspect="auto", origin="lower")
        plt.title("Spectrogram", fontsize=20)
        '''

        # スペクトログラム合計表示
        '''
        plt.plot(np.linspace(0, self.M, self.M), sum_list)
        '''

        #plt.show()

        return X

    def separateSound(self, data, second, start_second, num, save):
        '''
        waveファイルとして出力したい場合は save = True ※ただし、きちんとちゃんとした音に治すまでは行けていない。
        (一度stftしているため、それを直せていない、もしかしたら、もともとのデータを使えばいけるかもしれない)
        :param data: fft data (M,N)
        :param second: 切り取りたいデータの長さ(秒)
        :param start_second: 切り取りたい始まりの時間
        :param num: ファイルとして出力する際に、つける番号
        :param save: save == True のときにwavファイルを出力
        :return:if save != True のとき分かれたデータを返す
        '''

        start = int(start_second * self.frame_per_time)
        x = data[start : int(start + second * self.frame_per_time), : ] # データからほしいデータだけ切り抜く

        # 新しいwavファイルを作成する場合
        if save == True :
            file_name = "test" + str(num) + ".wav"

            x = b''.join(x)

            out = wave.open(file_name, 'w')
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(self.rate)
            out.writeframes(x)
            out.close()

            print("saved.")

        else:
            return x

    def istft(self, data, win):

        '''
        :param data:
        :param win:
        :return:
        '''
        M, N = data.shape

        assert (len(win) == N), "FFT length and window length are different"

        l = int((M - 1) * self.step + N)
        x = np.zeros(l, dtype = np.float64)
        wsum = np.zeros(l, dtype = np.float64)
        for m in range(M):
            start = int(self.step * m)

            ### 滑らかな接続
            x[start : start + N] = x[start : start + N] = np.fft.ifft(data[m, :]).real * win
            wsum[start : start + N] += win ** 2

        pos = (wsum != 0)
        x_pre = x.copy()

        ### 窓分のスケール合わせ
        x[pos] /= wsum[pos]

        return x

if __name__ == '__main__':
    '''
    frame = "wavファイルのパス"
    '''
    # frame = './test.wav'
    # frame = '../1.0/wh_1_45.wav'
    # frame = '../Experiment/190509/Test_wav/only_direct.wav'
    frame = '../Experiment/190509/Test_wav/ref_25_70.wav'
    # frame = '../Experiment/190509/environment/origin_sound.wav'

    '''
    パラメータ
    '''
    fft = WaveFft(frame)
    size = 1024
    # 窓関数はハミング
    hammingWindow = np.hamming(size)
    fs = 44100
    d = 1 / fs
    freqList = np.fft.fftfreq(size, d)

    '''
    関数使用
    '''
    # fft.fft(0, size)
    stft_data = fft.stft(size, hammingWindow)
    resyn_data = fft.istft(stft_data, hammingWindow)  # shape = (frame, )
    line = [0,4,8,12,16,20,24,28]
    second = 2
    X = np.zeros((len(line), int(second * fft.frame_per_time), fft.N), dtype = np.complex64)
    for i in line:
        x = fft.separateSound(stft_data, second, i, 1, False)
        # print(x.shape)
        X[int(i/4), :, :] = x
    print("データサイズ：", X.shape)

    '''
    プロット
    '''
    for i in range(8):
        ### スペクトログラム
        plt.imshow(abs(X[i, :, 15:23].T), aspect="auto", origin="lower")
        plt.title("Spectrogram", fontsize=20)

        ### 振幅の大きさ平均計算、プロット
        avg = np.average(abs(X[i, :, :]), axis = 0)
        # plt.plot(freqList, avg)
        ### 極値検出, プロット
        maxid = signal.argrelmax(avg[15:23], order=100) # 全体としてタプル形ではあるが、中身はarrayになってる。
        print("max", avg[15:23][maxid[0]])
        print(avg[15:23])
        # plt.plot(freqList[maxid], avg[maxid], 'ro')

        ### プロット設定
        # plt.axis([0, fs/16, 0, 1])
        # plt.xlim(0, fs/16)
        # plt.xlim(100, 1600)
        # plt.ylim(200, 3000)
        # plt.xlabel("Frequency[Hz]")
        # plt.ylabel("amplitude spectrum")
        plt.show()
