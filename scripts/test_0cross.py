from _function import MyFunc
from matplotlib import pyplot as plt
import numpy as np

mf = MyFunc()
wave_path = mf.recode_data_path + mf.data_search("191205", "Ts", "05", plane_wave=True)
target = 'glass_plate'
data_id = 1
data_dir = 0
sound_data, channel, sampling, frames = \
    mf.wave_read_func(wave_path + target + '_' + str(data_id+1) + '/' + str(data_dir) + '.wav')
start = mf.zero_cross(sound_data,  128, sampling, 512, up=True)

print(start)

plt.figure()
plt.specgram(sound_data[0], Fs=sampling, vmin=0, cmap='jet')
plt.ylim(0, 8000)
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
plt.colorbar()
plt.show()
