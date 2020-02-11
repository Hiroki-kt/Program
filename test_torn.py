import numpy as np
import wave
from _function import MyFunc

if __name__ == '__main__':
    mf = MyFunc()
    wave_path = '../../exp'
    data_set = np.zeros((len(DIRECTIONS), smic_num, data_num, data_set_freq_len), dtype=np.float)
    for data_dir in DIRECTIONS:
        sound_data, channel, sampling, frames = mf.wave_read_func(wave_path + str(data_dir) + '.wav')
        sound_data = np.delete(sound_data, [0, 5], 0)
        cut_data = self.reshape_sound_data(sound_data, sampling, 0.95, self.speaker_time, 0, [800, 1000, 2000])
        cut_data = np.reshape(cut_data, (self.mic_num, self.data_num, -1))
        fft_data = np.fft.rfft(cut_data)