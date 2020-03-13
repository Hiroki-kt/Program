# coding:utf-8
import configparser
from _function import MyFunc
import numpy as np
from matplotlib import pyplot as plt
import math
from scipy import stats
from scipy import signal
from _tf_generate_tsp import TSP
from scipy import optimize
from model_generate import Model
from test_ICA import ICA
from _beamforming import BeamForming


def fit_func(x, a, b, c, d):
    return a * np.sin(x) ** b


def reflection_interference(wave_file, mic2wall_dis, mic, sound_speed=340):
    data, channels, samlpling, frames = MyFunc().wave_read_func(wave_file)
    delay = mic2wall_dis * 2 / sound_speed
    # print(delay, delay * samlpling)
    delay_list = np.array([0] * int(delay * samlpling))
    reflect_wave = np.append(delay_list, data)
    direct_wave = np.append(data[0], delay_list)
    
    synthetic_wave = reflect_wave + direct_wave
    time_list = np.arange(0, direct_wave.shape[0])/44100
    fft_list = np.fft.rfftfreq(synthetic_wave.shape[0], 1/44100)
    fft_s = np.fft.rfft(synthetic_wave)
    
    freq_min = MyFunc().freq_ids(fft_list, 500)
    freq_max = MyFunc().freq_ids(fft_list, 2200)
    use_fft_list = fft_list[freq_min:freq_max]
    use_fft_s = fft_s[freq_min:freq_max]

    fft_s_normal = (use_fft_s - np.min(use_fft_s)) / (np.max(use_fft_s) - np.min(use_fft_s))
    
    peak_fft = signal.argrelmax(use_fft_s, order=1)
    peak_normal = signal.argrelmax(fft_s_normal, order=1)
    peak = signal.argrelmax(synthetic_wave, order=10)
    
    '''fitting'''
    d = 20
    synthetic_func = np.poly1d(np.polyfit(peak_fft[0], use_fft_s[peak_fft], d))
    normal_func = np.poly1d(np.polyfit(use_fft_list[peak_normal], fft_s_normal[peak_normal], d))
    
    # plt.figure()
    # plt.plot(use_fft_list[peak_normal], fft_s_normal[peak_normal])
    # plt.plot(use_fft_list, normal_func(use_fft_list))
    # plt.ylim(0.5, 1)
    # plt.show()
    
    plt.figure()
    plt.plot(time_list, synthetic_wave)
    # plt.plot(time_list[peak], synthetic_wave[peak])
    # plt.plot(time_list[peak], Y)
    # plt.plot(time_list, y)
    plt.title('Peak of Synthetic Wave' + mic)
    # plt.ylim(0, 70000)
    plt.legend()
    plt.show()
    
    # plt.figure()
    # plt.plot(use_fft_list[peak_fft], use_fft_s[peak_fft])
    # plt.plot(use_fft_list[peak_fft], synthetic_func(peak_fft[0]))
    # # plt.plot(time_list[peak], Y)
    # # plt.plot(time_list, y)
    # plt.title('Peak of Synthetic Wave()Freq' + mic)
    # # plt.ylim(0, 70000)
    # # plt.xlim(500, 2000)
    # plt.legend()
    # plt.show()
    
    # plt.figure()
    # plt.plot(time_list, synthetic_wave)
    # plt.title("Synthetic wave - direct" + mic)
    # plt.ylim(0, 70000)
    # plt.show()
    
    # plt.figure()
    # plt.plot(time_list, reflect_wave)
    # plt.title("Reflect wave" + mic)
    # plt.show()
    
    return normal_func, peak_normal


def separation_reflection_sound(synthetic, direct, mic, num, normal_func=None, peak_normal=None, plot=False):
    CONFIG_PATH = "../config/config_tf.ini"
    
    s_data, s_channels, s_samlpling, s_frames = MyFunc().wave_read_func(synthetic)
    d_data, d_channels, d_samlpling, d_frames = MyFunc().wave_read_func(direct)
    r_data = s_data - d_data

    s2_data = TSP(CONFIG_PATH).cut_up_tsp_data(use_data=s_data)
    d2_data = TSP(CONFIG_PATH).cut_up_tsp_data(use_data=d_data)
    # r2_data = TSP(CONFIG_PATH).cut_tsp_data(use_data=r_data)

    '''ICA'''
    ica_reflections_data = ICA().ica(wave_data=s2_data).T[0]
    # print(ica_reflections_data.shape)

    fft_d = np.fft.rfft(d2_data)
    fft_s = np.fft.rfft(s2_data)
    # fft_r = np.fft.rfft(r2_data)
    fft_ica_r = np.fft.rfft(ica_reflections_data)
    
    '''normalize'''
    # fft_d = stats.zscore(fft_d)
    # fft_d_normal = (fft_d - np.min(fft_d))/ (np.max(fft_d) - np.min(fft_d))
    # fft_r = stats.zscore(fft_r)
    # fft_r_normal = (fft_r - np.min(fft_r))/ (np.max(fft_r) - np.min(fft_r))
    
    '''graph x-axis'''
    fft_list = np.fft.rfftfreq(s2_data.shape[1], 1/44100)
    time_list = np.arange(0, s_data.shape[1])/44100
    time2_list = np.arange(0, d2_data.shape[1])/44100
    
    '''use freq'''
    freq_min = MyFunc().freq_ids(fft_list, 1000)
    freq_max = MyFunc().freq_ids(fft_list, 2200)
    freq_min1 = MyFunc().freq_ids(fft_list, 1000)
    freq_min2 = MyFunc().freq_ids(fft_list, 1200)
    freq_max1 = MyFunc().freq_ids(fft_list, 1900)
    freq_max2 = MyFunc().freq_ids(fft_list, 2100)
    
    '''use data'''
    use_fft_d = fft_d[mic][freq_min:freq_max]
    use_fft_s = fft_s[mic][freq_min:freq_max]
    # use_fft_r = fft_r[mic][freq_min:freq_max]
    use_fft_ica = fft_ica_r[freq_min:freq_max]
    # use_freq_list = fft_list[freq_min:freq_max]
    
    '''beam forming'''
    # config = './config_1015.ini'
    # bm = BeamForming(config)
    # tf = bm.steering_vector(fft_list, 1, 4)
    # print(tf.shape)

    '''smoothing'''
    n = 50
    v = np.ones(n)/float(n)
    s_fft_d = np.convolve(np.abs(use_fft_d), v, mode='same')
    s_fft_s = np.convolve(np.abs(use_fft_s), v, mode='same')
    # s_fft_r = np.convolve(np.abs(use_fft_r), v, mode='same')
    
    # mdl = Model()
    # mdl.speaker_chara(s2_data, d2_data, 44100, mic)
    
    '''extremun(極値)'''
    peak = signal.argrelmax(use_fft_d, order=10)
    # peak_r = signal.argrelmax((use_fft_r/2), order=10)
    peak_s = signal.argrelmax(use_fft_s, order=10)
    # print(len(peak[0]), len(peak_r[0]), len(peak_s[0]))
    peak_ica = signal.argrelmax(use_fft_ica, order=10)

    if not plot:
        '''test'''
        min_avg = np.average(np.abs(fft_s[mic][freq_min1:freq_min2]))
        max_avg = np.average(np.abs(fft_s[mic][freq_max1:freq_max2]))
        # min_avg = np.average(np.abs(fft_ica_r[freq_min1:freq_min2]))
        # max_avg = np.average(np.abs(fft_ica_r[freq_max1:freq_max2]))
        return [min_avg, max_avg]

    else:
        '''fitting'''
        # d = 30
        # direct_func = np.poly1d(np.polyfit(use_freq_list[peak], use_fft_d[peak], d))
        # d_s = 3
        # synthetic_func = np.poly1d(np.polyfit(use_freq_list[peak_s], use_fft_s[peak_s], d_s))
        
        '''plot'''
        # plt.figure()
        # plt.plot(use_freq_list, use_fft_s, 'y.', label='synthetic')
        # plt.plot(use_freq_list, np.abs(use_fft_s), 'y.', label='synthetic')
        # plt.plot(use_freq_list[peak_s[0]], use_fft_s[peak_s], label='$\psi$ = ' + str(num))
        # plt.plot(use_freq_list, s_fft_s, label='$\psi$ = ' + str(num))
        # plt.plot(use_freq_list[peak_ica[0]], use_fft_ica[peak_ica], label='$\psi$ = ' + str(num))
        # plt.plot(use_freq_list[peak_s[0]], use_fft_s[peak_s], 'y.', label='synthetic')
        # plt.plot(use_freq_list[peak[0]], use_fft_d[peak], 'g-', label='direct')
        # plt.plot(use_freq_list[peak[0]], use_fft_d[peak], 'g.', label='direct')
        # plt.plot(use_freq_list, direct_func(use_freq_list), 'r-', label='fitting')
        # plt.plot(use_freq_list, synthetic_func(use_freq_list), 'c-', label='d=' + str(d_s))
        # plt.plot(use_freq_list[peak_r[0]], use_fft_r[peak_r], 'b.', label="reflection")
        # plt.plot(use_freq_list[peak_r[0]], use_fft_r[peak_r], label="reflection")
        # plt.plot(use_freq_list, use_fft_r[mic], '-', label="reflection")
        if normal_func is not None and peak_normal is not None:
            # plt.plot(use_freq_list, np.max(use_fft_s) * normal_func(use_freq_list), label="model")
            # plt.plot(use_freq_list, direct_func(use_freq_list) * normal_func(use_freq_list), label="model")
            print("model")
        #
        # plt.xlim(1000, 2000)
        # plt.legend()
        # plt.xlabel('Frequency [Hz]')
        # plt.ylabel('Amplitude')
        # plt.ylim(0, 180000)
        # # plt.show()
        # plt.savefig('../_img/191105_2/' + str(num) + '.png')
        
        # plt.figure()
        # # plt.plot(time2_list, ica_reflections_data)
        # plt.plot(use_freq_list[peak_ica[0]], use_fft_ica[peak_ica])
        # plt.title(str(num))
        # plt.ylim(0, 3.0)
        # plt.show()
        
        # plt.figure()
        # plt.plot(time_list, np.abs(s_data[mic]), '.')
        # plt.plot(time_list, np.convolve(np.abs(s_data[mic]), v, mode='same'))
        # plt.xlim(0, 1.0)
        # plt.title("Synthetic wave [mic" + str(mic-1) + "]")
        # plt.show()
        #
        # plt.figure()
        # plt.plot(time2_list, d2_data[mic])
        # plt.ylim(0, 700)
        # plt.title("Direct wave [mic" + str(mic-1) + "]")
        # plt.show()
        #
        # plt.figure()
        # plt.plot(time_list, r_data[mic])
        # plt.xlim(0, 1.0)
        # plt.title("Reflection wave [mic" + str(mic-1) + "]")
        # plt.show()
    
        # test = np.abs(fft_r)/np.abs(fft_d)
        # #
        # plt.figure()
        # plt.xlim(500, 2000)
        # # plt.ylim(0, 1)
        # plt.title("reflection")
        # plt.show()
        #
        # plt.figure()
        # plt.plot(fft_list, fft_r[mic-1])
        # plt.xlim(1500, 2000)
        # plt.title("reflection")
        # plt.show()
        
        # plt.figure()
        # plt.plot(fft_list, fft_d[mic-1])
        # plt.xlim(500, 2000)
        # plt.title("direct")
        # # plt.ylim(0, 1)
        # plt.show()
        #
        # plt.figure()
        # plt.plot(fft_list, fft_s[mic])
        # plt.xlim(500, 2000)
        # plt.title("synthetic")
        # plt.show()
    
        # plt.figure()
        # plt.plot(fft_list, test[mic-1])
        # plt.xlim(500, 2000)
        # plt.ylim(0, 3)
        # plt.title("test")
        # plt.show()
        return -1
    
    
if __name__ == '__main__':
    # wave_file = "./tsp_origin.wav"
    wave_file = "../_exp/191029/origin_data/test0001_5.wav"
    m2w_d = 0.19  # [m]
    mic_1 = m2w_d - 0.03/math.sqrt(2)
    mic_3 = m2w_d + 0.03/math.sqrt(2)
    # reflection_interference(wave_file, m2w_d, " [mic center]")
    # a, b = reflection_interference(wave_file, mic_1, " [mic1, 2]")
    # reflection_interference(wave_file, mic_3, " [mic3, 4]")
    
    d_file = '../../../../OneDrive/Research/Recode_Data/only_speaker/10.wav'
    
    dir_name = MyFunc().data_search(200214, 'Ts', '10', None, plane_wave=True)
    path = MyFunc().recode_data_path + dir_name
    print(path)
    
    d_path = MyFunc().recode_data_path + MyFunc().data_search(191015, 'Ts', '01', None, plane_wave=False, calibration=True)
    print(d_path)
    
    # directions = [-50, -40, -30, -21, -10, 0, 9, 21, 30, 40, 50]
    directions = np.arange(-50, 51)
    # directions = [0, 20]
    min_list = []
    max_list = []
    for s in directions:
        file = path + str(s) + '.wav'
        print(str(s))
        avg = separation_reflection_sound(file, d_path, 0, s)
        min_list.append(avg[0])
        max_list.append(avg[1])
        # plt.figure()
        # separation_reflection_sound(file, d_file, 1, s, plot=True)
    # plt.legend()
    # plt.xlabel('Frequency [Hz]')
    # plt.ylabel('Amplitude')
    # plt.show()
    
    plt.figure()
    plt.plot(directions, min_list, label="1000Hz")
    plt.plot(directions, max_list, label="2000Hz")
    plt.xlabel('Azimuth [deg]', fontsize=15)
    plt.ylabel('Amplitude', fontsize=15)
    plt.legend(fontsize=15)
    plt.show()
    #
    plt.plot(directions, np.array(min_list)/np.array(max_list), label='D')
    plt.ylim(0, 3.0)
    plt.xlabel('Azimuth [deg]')
    plt.ylabel('$e_m$/$e_h$')
    plt.show()