#coding:utf-8
#!/usr/bin/env python

# プロット関係のライブラリ
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import sys

# 音声関係のライブラリ
import pyaudio
import struct

class PlotWindow:
    def __init__(self):
        # マイクインプット設定
        self.CHUNK = 1024
        self.RATE = 44100
        self.update_seconds = 50
        self.CHANNEL = 1
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format = pyaudio.paInt16,
                                      channels = self.CHANNEL,
                                      rate = self.RATE,
                                      input = True,
                                      frames_per_buffer = self.CHUNK
                                      )
        # 音声デーラの格納場所(プロットデータ)
        self.data = np.zeros(self.CHUNK)
        self.axis = np.fft.fftfreq(len(self.data), d = 1.0/self.RATE)

        # プロット初期設定
        self.win = pg.GraphicsWindow()
        self.win.setWindowTitle("SpectrumAnalyzer")
        self.plt = self.win.addPlot()
        self.plt.setYRange(0, 100)

        # アップデート時間の設定
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(self.update_seconds) # 10msごとにupdateを呼び出し

    def update(self):
        self.data = np.append(self.data, self.AudioInput())
        if len(self.data)/1024 > 10:
            self.data = self.data[1024:]
        self.fft_data = self.FFT_AMP(self.data)
        self.axis = np.fft.fftfreq(len(self.data), d = 1.0/self.RATE)
        self.plt.plot(x = self.axis, y = self.fft_data, clear = True, pen = "y")

    def AudioInput(self):
        ret = self.stream.read(self.CHUNK)
        # 音声の読み取り（バイナリ）CHUNKが大きい所で時間がかかる
        # バイナリ → 数値(int16)に変換
        # 32768.0 = 2^16で割るのは正規化（絶対値を1以下にするため）
        ret =np.frombuffer(ret, dtype="int16")/32768.0
        return ret

    def FFT_AMP(self, data):
        data = np.hamming(len(data)) * data
        data = np.fft.fft(data)
        data = np.abs(data)
        return data


if __name__ == '__main__':
    plotwin = PlotWindow()
    if (sys.flags.interactive!=1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QGuiApplication.instance().exec_()
