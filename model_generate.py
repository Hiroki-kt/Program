# coding:utf-8
from matplotlib import pyplot as plt
import numpy as np
import math
from scipy import signal
# from datetime import datetime
import os


class Model():
    def model(self, freq_mid, freq_hig, psi_deg=15, theta_deg=90, phi_deg=90):
        # パラメータ
        gamma_dm, gamma_sm = self.gamma(freq_mid)
        gamma_dh, gamma_sh = self.gamma(freq_hig)
        rho = 0.5
        alpha = 2
        
        s_q = np.array([math.cos(math.radians(theta_deg)), math.sin(math.radians(theta_deg))])
        q_m = -1 * np.array([math.cos(math.radians(phi_deg)), math.sin(math.radians(phi_deg))])
        psi_rad = math.radians(psi_deg)
        n = np.array([- 1 * math.sin(psi_rad), -1 * math.cos(psi_rad)])
        r = s_q + 2 * ((-1 * s_q) @ n) * n
        
        # 答えの計算
        specular = (r @ q_m) / (np.sqrt(r[0] ** 2 + r[1] ** 2) * np.sqrt(q_m[0] ** 2 + q_m[1] ** 2))
        diffuse = ((-1 * s_q) @ n) / (np.sqrt(n[0] ** 2 + n[1] ** 2) * np.sqrt(s_q[0] ** 2 + s_q[1] ** 2))
        middle = gamma_sm * specular ** alpha + gamma_dm * diffuse * rho
        high = gamma_sh * specular ** alpha + gamma_dh * diffuse * rho
        
        return middle / high, middle, high
    
    @staticmethod
    def gamma(freq):
        e = math.e
        # gamma_l = 1 - 1 / (1 + e ** (-1 * (freq/15 - 35/3)))
        gamma_d = 0.8 * 1 / (1 + e ** (-1 * (freq / 150 - 25 / 3)))
        gamma_s = 1 - gamma_d
        
        return gamma_d, gamma_s

    @staticmethod
    def speaker_chara(synthetic_data, direct_data, sampling, mic, d, plot=False):
        fft_d = np.fft.rfft(direct_data)
        fft_s = np.fft.rfft(synthetic_data)
        
        reflection_data = synthetic_data - direct_data
        
        fft_r = np.fft.rfft(reflection_data)
        
        '''正規化'''
        # fft_d = stats.zscore(fft_d)
        # fft_d_normal = (fft_d - np.min(fft_d)) / (np.max(fft_d) - np.min(fft_d))
        # fft_r = stats.zscore(fft_r)
        # fft_r_normal = (fft_r - np.min(fft_r)) / (np.max(fft_r) - np.min(fft_r))
        
        time_list = np.arange(0, synthetic_data.shape[1]) / sampling
        fft_list = np.fft.rfftfreq(synthetic_data.shape[1], 1 / sampling)
        
        '''極致データのみにする'''
        peak = signal.argrelmax(fft_d[mic], order=30)
        
        func = np.poly1d(np.polyfit(peak[0], fft_d[mic][peak], d))
        
        if plot:
            plt.figure()
            plt.plot(fft_list[peak[0]], fft_d[mic][peak])
            plt.plot(fft_list[peak[0]], func(peak[0]), label='n=' + str(n))
            plt.legend()
            # plt.ylim(0, 700)
            plt.xlim(0, 2000)
            plt.show()
            # print(func)

        else:
            return func