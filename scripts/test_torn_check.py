import numpy as np
from matplotlib import pyplot as plt
from _function import MyFunc


class CheckTorn(MyFunc):
    def __init__(self):
        super().__init__()
        pass
    
    def main(self, name, directions, mic_id):
        wave_path = self.recode_data_path + name
        # wave_path = '/Users/hiroki-kt/OneDrive/Research/Recode_Data/200208_PTo08/torn_1000_1/'
        # use_freq_list = np.arange(500, 8000, 100)
        use_freq_list = [2000]
        for freq in use_freq_list:
            max_list = []
            for dire in directions:
                sound_data, channels, sampling, frames = self.wave_read_func(wave_path + str(dire) + '.wav')
                sound_data = np.delete(sound_data, [0, 5], 0)
                
                # print(sound_data.shape[1])
                start_time = self.zero_cross(sound_data, 128, sampling, 512, 45178, up=True)
                # print(start_time)
                if start_time < 0:
                    start_time = 0
                cut_data = sound_data[:, start_time: int(start_time + 45178 * 7)]
                cut_data = np.reshape(cut_data, (4, 7, -1))
                cut_data = cut_data[mic_id, 3, :]
                fft_data = np.fft.rfft(cut_data)
                fft_data = np.abs(fft_data)
                freq_list = np.fft.rfftfreq(cut_data.shape[0], 1/sampling)
                # print(freq_list)
                freq_id = self.freq_ids(freq_list, freq)
                max_list.append(fft_data[freq_id])
            
            self.my_makedirs('../../_img/200210/mic_' + str(mic_id+1) + '/')
            test_list = [(i - min(max_list))/(max(max_list)-min(max_list)) for i in max_list]
            plt.figure()
            plt.plot(directions, test_list)
            # plt.ylim(0, 500000)
            plt.xlabel('Azimuth [deg]')
            plt.xlabel('Amplitude spectrum')
            plt.show()
            # plt.savefig('../../_img/200210/mic_' + str(mic_id+1) + '/' + str(freq) + '_' + str(mic_id+1) + '.png')
    
    
if __name__ == '__main__':
    # NAME = '191205_Pts05/glass_plate_1/'
    NAME = '200208_PTo08/torn_1000_1/'
    # NAME = '191015_PTs01/'
    # DIRECTIONS = [0, 10]
    DIRECTIONS = np.arange(-45, 45)
    MIC_ID = 3
    ct = CheckTorn()
    for i in range(4):
        ct.main(NAME, DIRECTIONS, i)
