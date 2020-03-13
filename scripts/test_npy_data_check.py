import numpy as np
from matplotlib import pyplot as plt
from _function import MyFunc

mf =MyFunc()

data_set_file_path = "_array/200225/"
data_name = '200214_PTs10'
data_set_path = mf.onedrive_path + data_set_file_path + data_name + '.npy'
mic = 0
data_id = 0
origin_path = mf.speaker_sound_path + '2_up_tsp_1num.wav'
smooth_step = 50
freq_list = np.fft.rfftfreq(mf.get_frames(origin_path), 1 / 44100)
freq_max_id = mf.freq_ids(freq_list, 7000)
freq_min_id = mf.freq_ids(freq_list, 1000)
freq_list = freq_list[freq_min_id + int(smooth_step / 2) - 1:freq_max_id - int(smooth_step / 2)]

data_set_freq_len = freq_max_id - freq_min_id - (smooth_step - 1)


freq_1000 = mf.freq_ids(freq_list, 1000)
freq_2000 = mf.freq_ids(freq_list, 2000)
print(freq_list)
print(freq_1000, freq_2000)

data_set = np.load(data_set_path)
DIRECTIONS = np.arange(data_set.shape[0]/2 * (-1), data_set.shape[0]/2)

plt.figure()
plt.plot(DIRECTIONS, data_set[:, mic, data_id, freq_1000], label="1000Hz")
plt.plot(DIRECTIONS, data_set[:, mic, data_id, freq_2000], label="2000Hz")
plt.xlabel('Azimuth [deg]', fontsize=15)
plt.ylabel('Amplitude', fontsize=15)
plt.legend(fontsize=15)
plt.show()