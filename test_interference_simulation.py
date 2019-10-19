import wave
from _function import MyFunc
import numpy as np
from matplotlib import pyplot as plt
import math

def reflection_interference(wave_file, mic2wall_dis, sound_speed=340):
    func = MyFunc()
    data, channels, samlpling, frames = func.wave_read_func(wave_file)
    delay = mic2wall_dis * 2 / sound_speed
    print(delay, delay * samlpling)
    delay_list = np.array([0] * int(delay * samlpling))
    reflect_wave = np.append(delay_list, data)
    direct_wave = np.append(data[0], delay_list)
    
    synthetic_wave = reflect_wave + direct_wave
    
    print(direct_wave.shape, reflect_wave.shape, synthetic_wave.shape)
    
    time_list = np.arange(0, direct_wave.shape[0])/44100
    
    plt.figure()
    # plt.plot(time_list, reflect_wave, label='R')
    # plt.plot(time_list, direct_wave, label='D')
    plt.plot(time_list, synthetic_wave)
    # plt.legend()
    # plt.xlim(0.2, 0.201)
    plt.show()
    
def separation_reflection_sound(synthetic, direct):
    func = MyFunc()
    s_data, s_channels, s_samlpling, s_frames = func.wave_read_func(synthetic)
    d_data, d_channels, d_samlpling, d_frames = func.wave_read_func(direct)
    
    r_data = s_data - d_data

    time_list = np.arange(0, s_data.shape[1])/44100
    
    plt.figure()
    plt.plot(time_list, r_data[0])
    plt.show()
    
if __name__ == '__main__':
    wave_file = "./tsp_origin.wav"
    # m2w_d = 0.2  # [m]
    # reflection_interference(wave_file, m2w_d)
    # mic_1 = m2w_d - 0.03/math.sqrt(2)
    # mic_3 = m2w_d + 0.03/math.sqrt(2)
    # reflection_interference(wave_file, mic_1)
    # reflection_interference(wave_file, mic_3)
    
    s_file = '../../../../OneDrive/Research/Recode_Data/up_tsp/0.wav'
    d_file = '../../../../OneDrive/Research/Recode_Data/only_speaker/10.wav'
    
    separation_reflection_sound(s_file, d_file)