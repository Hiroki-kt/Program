# conding:utf-8
import numpy as np
import os
import configparser
import sys
from _function import MyFunc
from matplotlib import pyplot as plt
from scipy import stats


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
            # Sound Reshape Parameter
            self.NEED_NUM = int(config['ReshapeSound']['Need_Num'])
            self.TSP_PATH = config['ReshapeSound']['TSP_Origin_Path']
            self.CROSS0_SIZE = int(config['ReshapeSound']['0Cross_Size'])
            self.CROSS0_STEP = int(config['ReshapeSound']['0Cross_Step'])
            self.CROSS0_THRESHOLD = int(config['ReshapeSound']['0Cross_Threshold'])
            self.TSP, self.TSP_CHANNELS, self.TSP_SAMPLING, self.TSP_FRAMES = self.wave_read_func(self.TSP_PATH)
            self.ITSP = self.TSP[0][::-1]
        else:
            print("#couldn't find", config_path)
            sys.exit()
    
    def cut_tsp_data(self, num):
        # Initialization
        start = count = START_TIME = 0
        file = self.SOUND_PATH + str(num) + '.wav'
        # Main
        origin_sound, channels, sampling, frames = self.wave_read_func(file)
        origin_sound = np.delete(origin_sound, [0, 5], 0)
        # Zero Cross
        while count < sampling/self.CROSS0_STEP:
            sign = np.diff(np.sign(origin_sound[:, start: start + self.CROSS0_SIZE]))
            zero_cross = np.where(sign)[1].shape[0]
            if zero_cross > self.CROSS0_THRESHOLD:
                START_TIME = start
                break
            start = start + self.CROSS0_STEP
            count += 1
        if START_TIME != 0:
            use_sound = origin_sound[:, START_TIME: int(START_TIME + self.TSP_FRAMES * self.NEED_NUM)]
            use_sound = np.reshape(use_sound, (self.MIC_NUM, self.NEED_NUM, -1))
            return use_sound
        else:
            return -1

    def generate_tf(self, tsp_res):
        tsp_res = np.average(tsp_res, axis=0)
        # 今回は同じ大きさになるはずだが、一応itsp信号と同じ大きさなのか確認
        N = self.ITSP.shape[0]
        residual = tsp_res.shape[0] - N
        if residual >= N:
            tsp_res[:N] = tsp_res[:N] + tsp_res[N:2 * N]
        else:
            tsp_res[:residual] = tsp_res[:residual] + tsp_res[N:N + residual]
        # fft
        fft_tsp_res = np.fft.rfft(tsp_res[:N])
        fft_itsp = np.fft.rfft(self.TSP[0])
        # 畳み込み
        fft_ir = fft_tsp_res * fft_itsp
        ir = np.fft.irfft(fft_ir)
        # 正規化
        ir = stats.zscore(ir)
        # ir = ir / np.max(ir)
        # ir = (ir - np.min(ir))/ (np.max(ir) - np.min(ir))
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
    CONFIG_PATH = "./config_1003.ini"
    tsp = TSP(CONFIG_PATH)
    DIRECTIONS = list(range(-45, -44))
    error_num = 0
    mic = 0
    max_array = np.zeros((tsp.MIC_NUM, 100))
    data = tsp.cut_tsp_data(22)
    tf, fft_tf = tsp.generate_tf(data[mic, :, :])
    # max_list = []
    # for i, deg in enumerate(DIRECTIONS):
    #     try:
    #         data = tsp.cut_tsp_data(deg)
    #         # for mic in range(tsp.MIC_NUM):
    #             # tf, fft_tf = tsp.generate_tf(data[mic, :, :])
    #             # max_array[mic, i] = np.max(tf)
    #     except:
    #         print(deg, "Error")
    #         error_num += 1
    #         if error_num >= 20:
    #             print("Error 20times")
    #             print("Program stop")
    #             sys.exit()
    #         pass

    # for i in range(tsp.MIC_NUM):
    #     title = "Impulse Response Mic " + str(i+1)
    #     X_Label = "Azimuth [deg]"
    #     Y_Label = "2k-1k ?? [Hz], Max num"
    #     tsp.data_plot(DIRECTIONS, max_array[i, :], title=title, xlabel=X_Label, ylabel=Y_Label)
    #     plt.ylim(10**11, 2.0*10**11)
    #     plt.show()