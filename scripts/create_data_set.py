import numpy as np
from _function import MyFunc
from _tf_generate_tsp import TSP
from scipy import stats
from configparser import ConfigParser
from distutils.util import strtobool
from mic_setting import MicSet
import sys
from matplotlib import pyplot as plt


class CreateDataSet(MyFunc):
    def __init__(self, config_path):
        super().__init__()
        config = ConfigParser()
        config.read(config_path)
        # Data param
        self.freq_max = int(config['Data']['Freq_Max'])
        self.freq_min = int(config['Data']['Freq_Min'])
        self.smooth_step = int(config['Data']['Smooth_Step'])
        self.date = config['Data']['Date']
        self.sound_kind = config['Data']['Sound_Kind']
        self.geometric = config['Data']['Geometric']
        self.target = config['Data']['Target']
        self.plane_wave = bool(strtobool(config['Data']['Plane_Wave']))
        self.data_num = int(config['Data']['Data_Num'])
        # Label param
        self.label_max = int(config['Label']['Label_Max'])
        label_min = int(config['Label']['Label_Min'])
        self.label_step = int(config['Label']['Label_Step'])
        self.DIRECTIONS = np.arange(label_min, self.label_max)
        # Speaker sound
        file_name = config['Speaker_Sound']['File']
        if file_name != 'None':
            origin_sound_path = self.speaker_sound_path + file_name
            origin_data, channel, origin_sampling, self.origin_frames = self.wave_read_func(origin_sound_path)
            self.fft_list = np.fft.rfftfreq(self.origin_frames, 1 / origin_sampling)
        else:
            origin_sampling = 44100
            self.speaker_time = float(config['Speaker_Sound']['Time'])
            self.fft_list = np.fft.rfftfreq(int(self.speaker_time * origin_sampling), 1 / origin_sampling)
        self.freq_min_id = self.freq_ids(self.fft_list, self.freq_min)
        self.freq_max_id = self.freq_ids(self.fft_list, self.freq_max)
        self.data_set_freq_len = self.freq_max_id - self.freq_min_id - (self.smooth_step - 1)
        self.use_fft_list = self.fft_list[self.freq_min_id + int(self.smooth_step/2) - 1:
                                          self.freq_max_id - int(self.smooth_step/2)]
        self.beam = bool(strtobool(config['Data']['Beam']))
        self.mic_name = config['Tool']['Mic']
        self.mic_num = int(config['Tool']['Mic_Num'])
        if self.beam:
            self.ms = MicSet(self.mic_name)
            self.tf = self.ms.steering_vector_azimuth(self.origin_frames, 1/origin_sampling, self.use_fft_list)
            print("make stearing vector for beamforming")
            
    def __call__(self, impulse=False):
        if self.sound_kind == 'To':
            self.tone_sigal()
        elif self.sound_kind == 'Ts':
            if self.target == 'None':
                self.tsp_signal(impulse=impulse)
            else:
                self.tsp_signal_individually()
        else:
            print("Error")
            sys.exit()

    def tsp_signal(self, impulse=False):
        tsp = TSP('../config_tf.ini')
        recode_data_directory = self.data_search(self.date, self.sound_kind, self.geometric,
                                                             plane_wave=self.plane_wave)
        wave_path = self.recode_data_path + recode_data_directory
        print(wave_path)
        print("Making data set....")
        # impulse = Trueでインパルス応答計算用に平均したデータを作成
        if impulse:
            data_set = np.empty((0, self.data_set_freq_len), dtype=np.float)
            for data_dir in self.DIRECTIONS:
                sound_data, channel, sampling, frames = self.wave_read_func(wave_path + str(data_dir) + '.wav')
                cut_data = tsp.cut_up_tsp_data(use_data=sound_data)
                fft_data = np.fft.rfft(cut_data)[0]
                fft_data = fft_data[self.freq_min_id:self.freq_max_id]
                smooth_data = np.convolve(fft_data, np.ones(self.smooth_step) / float(self.smooth_step), mode='valid')
                smooth_data = np.reshape(smooth_data, (1, -1))
                # print(smooth_data.shape)
                data_set = np.append(data_set, smooth_data, axis=0)
                print('finish: ', data_dir + 50 + 1, '/', len(self.DIRECTIONS))
            print('Made data set: ', data_set.shape)
            return np.real(data_set)
        # impluse = FalseでSVRやPCA用の一個一個のデータに分割
        else:
            data_set = np.zeros((len(self.DIRECTIONS), self.mic_num, self.data_num, self.data_set_freq_len),
                                dtype=np.float)
            print('Need Number', self.origin_frames * 1.2)
            for data_dir in self.DIRECTIONS:
                sound_data, channel, sampling, frames = self.wave_read_func(wave_path + str(data_dir) + '.wav')
                sound_data = np.delete(sound_data, [0, 5], 0)
                start_time = self.zero_cross(sound_data, 128, sampling, 512, self.origin_frames, up=True)
                if start_time < 0:
                    start_time = 0
                cut_data = sound_data[:, start_time: int(start_time + self.origin_frames * self.data_num)]
                cut_data = np.reshape(cut_data, (self.mic_num, self.data_num, -1))
                fft_data = np.fft.rfft(cut_data)
                
                if data_dir == self.DIRECTIONS[0]:
                    print("#######################")
                    print("This mic is ", self.mic_name)
                    print("Channel ", channel)
                    print("Frames ", frames)
                    print("Data Set ", fft_data.shape)
                    # print("0Cross point ", start_time)
                    print("Object ", self.target)
                    print("Rate ", sampling)
                    print("#######################")
                    
                for data_id in range(cut_data.shape[1]):
                    for mic in range(cut_data.shape[0]):
                        smooth_data = np.convolve(np.abs(fft_data[mic, data_id, self.freq_min_id:self.freq_max_id]),
                                                  np.ones(self.smooth_step) / float(self.smooth_step), mode='valid')
                        smooth_data = np.real(np.reshape(smooth_data, (1, -1)))[0]
                        normalize_data = stats.zscore(smooth_data, axis=0)
                        # normalize_data = (smooth_data - smooth_data.mean()) / smooth_data.std()
                        # normalize_data = (smooth_data - min(smooth_data))/(max(smooth_data) - min(smooth_data))
                        data_set[data_dir, mic, data_id, :] = normalize_data
                        # print(normalize_data)
                print('finish: ', data_dir + self.label_max + 1, '/', len(self.DIRECTIONS))
            print('Made data set: ', data_set.shape)
            output_path = self.make_dir_path(array=True)
            np.save(output_path + recode_data_directory.strip('/') + '_freq_7000', data_set)

    def tsp_signal_individually(self):
        wave_path = self.recode_data_path + self.data_search(self.date, self.sound_kind, self.geometric,
                                                             plane_wave=self.plane_wave)
        print(wave_path)
        if self.beam:
            data_set = np.zeros((len(self.DIRECTIONS), len(self.ms.ss_list), self.data_num, self.data_set_freq_len),
                                dtype=np.float)
        else:
            data_set = np.zeros((len(self.DIRECTIONS), self.mic_num, self.data_num, self.data_set_freq_len),
                                dtype=np.float)
        for data_id in range(self.data_num):
            for data_dir in self.DIRECTIONS:
                sound_data, channel, sampling, frames = \
                    self.wave_read_func(wave_path + self.target + '_' + str(data_id + 1) + '/' + str(data_dir) + '.wav')
                if self.mic_name == 'Respeaker':
                    sound_data = np.delete(sound_data, [0, 5], 0)
                elif self.mic_name == 'Matrix':
                    sound_data = np.delete(sound_data, 0, 0)
                else:
                    print("Error")
                    sys.exit()
                start_time = self.zero_cross(sound_data, 128, sampling, 512, self.origin_frames, up=True)
                if start_time < 0:
                    start_time = 0
                sound_data = sound_data[:, start_time: int(start_time + self.origin_frames)]
                if data_dir == self.DIRECTIONS[0]:
                    print("#######################")
                    print("This mic is ", self.mic_name)
                    print("Channel ", channel)
                    print("Frames ", frames)
                    print("Data Set ", sound_data.shape)
                    print("0Cross point ", start_time)
                    print("Object ", self.target)
                    print("Rate ", sampling)
                    print("#######################")
                fft_data = np.fft.rfft(sound_data)
                if self.beam:
                    bmp = self.ms.beam_forming_localization(fft_data[:, self.freq_min_id:self.freq_max_id],
                                                            self.tf, self.fft_list)
                    for dir in range(bmp.shape[0]):
                        smooth_data = np.convolve(np.abs(bmp[dir, :]),
                                                  np.ones(self.smooth_step) / float(self.smooth_step), mode='valid')
                        smooth_data = np.real(np.reshape(smooth_data, (1, -1)))[0]
                        normalize_data = stats.zscore(smooth_data, axis=0)  # 平均0 分散1
                        data_set[data_dir, dir, data_id, :] = normalize_data
                else:
                    for mic in range(fft_data.shape[0]):
                        smooth_data = np.convolve(np.abs(fft_data[mic, self.freq_min_id:self.freq_max_id]),
                                                  np.ones(self.smooth_step) / float(self.smooth_step), mode='valid')
                        smooth_data = np.real(np.reshape(smooth_data, (1, -1)))[0]
                        normalize_data = stats.zscore(smooth_data, axis=0)  # 平均0 分散1
                        # normalize_data = (smooth_data - smooth_data.mean()) / smooth_data.std()
                        # normalize_data = (smooth_data - min(smooth_data))/(max(smooth_data) - min(smooth_data))
                        data_set[data_dir, mic, data_id, :] = normalize_data
            
                # print('finish: ', data_dir + 50 + 1, '/', len(self.DIRECTIONS))
            print('finish: ', data_id + 1, '/', self.data_num)
            print('***********************************************')
        print('Made data set: ', data_set.shape)
        output_path = self.make_dir_path(array=True)
        np.save(output_path + self.target + '_beam.npy', data_set)
        
    def tone_sigal(self):
        wave_path = self.recode_data_path + self.data_search(self.date, self.sound_kind, self.geometric,
                                                             plane_wave=self.plane_wave)
        print(wave_path)
        print("Making data set....")
        # data_set_freq_len =
        data_set = np.zeros((len(self.DIRECTIONS), self.mic_num, self.data_num, self.data_set_freq_len), dtype=np.float)
        for data_dir in self.DIRECTIONS:
            sound_data, channel, sampling, frames = self.wave_read_func(wave_path + str(data_dir) + '.wav')
            sound_data = np.delete(sound_data, [0, 5], 0)
            cut_data = self.reshape_sound_data(sound_data, sampling, 0.95, self.speaker_time, 0, [800, 1000, 2000])
            cut_data = np.reshape(cut_data, (self.mic_num, self.data_num, -1))
            fft_data = np.fft.rfft(cut_data)

            if data_dir == self.DIRECTIONS[0]:
                print("#######################")
                print("This mic is ", self.mic_name)
                print("Channel ", channel)
                print("Frames ", frames)
                print("Data Set ", fft_data.shape)
                print("Object ", self.target)
                print("Rate ", sampling)
                print("#######################")
                
            for data_id in range(cut_data.shape[1]):
                for mic in range(cut_data.shape[0]):
                    smooth_data = np.convolve(np.abs(fft_data[mic, data_id, self.freq_min_id:self.freq_max_id]),
                                              np.ones(self.smooth_step) / float(self.smooth_step), mode='valid')
                    smooth_data = np.real(np.reshape(smooth_data, (1, -1)))[0]
                    normalize_data = stats.zscore(smooth_data, axis=0)
                    # normalize_data = (smooth_data - smooth_data.mean()) / smooth_data.std()
                    # normalize_data = (smooth_data - min(smooth_data))/(max(smooth_data) - min(smooth_data))
                    data_set[data_dir, mic, data_id, :] = normalize_data
                    # print(normalize_data)
            print('finish: ', data_dir + self.label_max + 1, '/', len(self.DIRECTIONS))
        print('Made data set: ', data_set.shape)
        output_path = self.make_dir_path(array=True)
        np.save(output_path + self.sound_kind + '.npy', data_set)


if __name__ == '__main__':
    config_ini = '../config_191015_PTs01_freq_7000.ini'
    cd = CreateDataSet(config_ini)
    cd()
