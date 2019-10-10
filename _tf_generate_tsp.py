# conding:utf-8
import numpy as np
import os
import configparser
import sys
from _function import MyFunc
from matplotlib import pyplot as plt
from datetime import datetime


class TSP(MyFunc):
    def __init__(self, config_path):
        if os.path.exists(config_path):
            config = configparser.ConfigParser()
            config.read(config_path)
            # Mic Array
            self.MIC = config['MicArray']['Mic']
            self.MIC_NUM = int(config['MicArray']['Mic_Num'])
            # Sound Data Parameter
            self.SOUND_SPEED = float(config['SoundParam']['Sound_Speed'])
            self.SOUND_PATH = config['SoundParam']['File_Path']
            self.UP_TSP = bool(config['SoundParam']['UP_TSP'])
            # Sound Reshape Parameter
            self.NEED_NUM = int(config['ReshapeSound']['Need_Num'])
            self.TSP_PATH = config['ReshapeSound']['TSP_Origin_Path']
            self.CROSS0_SIZE = int(config['ReshapeSound']['0Cross_Size'])
            self.CROSS0_STEP = int(config['ReshapeSound']['0Cross_Step'])
            self.CROSS0_THRESHOLD = int(config['ReshapeSound']['0Cross_Threshold'])
            # uptspの場合は逆
            if self.UP_TSP:
                itsp, self.TSP_CHANNELS, self.TSP_SAMPLING, self.TSP_FRAMES = self.wave_read_func(self.TSP_PATH)
                self.ITSP = itsp[0]
                self.TSP = self.ITSP[::-1]
            else:
                Tsp, self.TSP_CHANNELS, self.TSP_SAMPLING, self.TSP_FRAMES = self.wave_read_func(self.TSP_PATH)
                self.TSP = Tsp[0]
                self.ITSP = self.TSP[::-1]
            print(self.TSP.shape, self.ITSP.shape)
            self.FIG_PATH = '../_img/19' + datetime.today().strftime("%m%d") + '/' + \
                        datetime.today().strftime("%H%M%S") + "/"
            # self.my_makedirs(self.FIG_PATH)
                
        else:
            print("#couldn't find", config_path)
            sys.exit()
    
    def cut_tsp_data(self, num):
        # Initialization
        file = self.SOUND_PATH + str(num) + '.wav'
        # Main
        origin_sound, channels, sampling, frames = self.wave_read_func(file)
        origin_sound = np.delete(origin_sound, [0, 5], 0)
        # Zero Cross
        START_TIME = self.zero_cross(origin_sound, self.CROSS0_STEP, sampling, self.CROSS0_SIZE)
        if START_TIME != 0:
            # img_file = self.FIG_PATH + str(num) + '.png'
            use_sound = origin_sound[:, START_TIME: int(START_TIME + self.TSP_FRAMES * self.NEED_NUM)]
            # plt.specgram(use_sound[0], Fs=44100)
            # plt.savefig(img_file)
            use_sound = np.reshape(use_sound, (self.MIC_NUM, self.NEED_NUM, -1))
            return use_sound
        else:
            return -1

    def generate_tf(self, tsp_res, plot=False):
        tsp_res = np.average(tsp_res, axis=0)
        # plt.specgram(tsp_res, Fs=44100)
        # plt.show()
        # 今回は同じ大きさになるはずだが、一応itsp信号と同じ大きさなのか確認
        N = self.ITSP.shape[0]
        residual = tsp_res.shape[0] - N
        if residual >= N:
            tsp_res[:N] = tsp_res[:N] + tsp_res[N:2 * N]
        else:
            tsp_res[:residual] = tsp_res[:residual] + tsp_res[N:N + residual]
        # fft
        fft_tsp_res = np.fft.rfft(tsp_res[:N])
        fft_itsp = np.fft.rfft(self.ITSP)
        # 畳み込み
        fft_ir = fft_tsp_res * fft_itsp
        ir = np.fft.irfft(fft_ir)
        # 正規化
        # ir = stats.zscore(ir)
        # ir = ir / np.max(ir)
        # ir = (ir - np.min(ir))/ (np.max(ir) - np.min(ir))
        if plot:
            time_list = [k / 44100 for k in range(N)]
            plt.plot(time_list, ir)
            plt.show()
            plt.specgram(ir, Fs=44100, cmap='jet')
            plt.ylim(0, 8000)
            plt.clim(-100, 0)
            plt.colorbar()
            plt.show()
        return ir, fft_ir

        
if __name__ == '__main__':
    CONFIG_PATH = "./config_tf.ini"
    tsp = TSP(CONFIG_PATH)
    DIRECTIONS = list(range(-50, 50))
    error_num = 0
    mic = 0
    max_array = np.zeros((tsp.MIC_NUM, 100))
    # data = tsp.cut_tsp_data(22)
    # tf, fft_tf = tsp.generate_tf(data[mic, :, :])
    max_list = []
    for i, deg in enumerate(DIRECTIONS):
        print(deg)
        data = tsp.cut_tsp_data(deg)
        for mic in range(tsp.MIC_NUM):
            tf, fft_tf = tsp.generate_tf(data[mic, :, :])
            max_array[mic, i] = np.max(tf)

    for i in range(tsp.MIC_NUM):
        title = "Impulse Response Mic " + str(i+1)
        X_Label = "Azimuth [deg]"
        Y_Label = "2k-1k ?? [Hz], Max num"
        tsp.data_plot(DIRECTIONS, max_array[i, :], title=title, xlabel=X_Label, ylabel=Y_Label)
        plt.ylim(10**11, 2.0*10**11)
        plt.show()