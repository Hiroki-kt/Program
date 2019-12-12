# coding:utf-8
from matplotlib import pyplot as plt
import numpy as np
from _function import MyFunc
from _tf_generate_tsp import TSP
from sklearn.decomposition import PCA
from sklearn.svm import SVC, SVR
from matplotlib import colors
from matplotlib import cm
from sklearn.metrics import accuracy_score
from scipy import stats
from configparser import ConfigParser
from distutils.util import strtobool
import sys
from mpl_toolkits.mplot3d import Axes3D
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import confusion_matrix
# from sklearn.preprocessing import StandardScaler


class PrametricEigenspace(MyFunc):
    def __init__(self, config_path='config_nonpara.ini'):
        super().__init__()
        config = ConfigParser()
        config.read(config_path)
        # Data param
        freq_max = int(config['Data']['Freq_Max'])
        freq_min = int(config['Data']['Freq_Min'])
        self.smooth_step = int(config['Data']['Smooth_Step'])
        self.date = config['Data']['Date']
        self.sound_kind = config['Data']['Sound_Kind']
        self.geometric = config['Data']['Geometric']
        self.plane_wave = bool(strtobool(config['Data']['Plane_Wave']))
        # Label param
        label_max = int(config['Label']['Label_Max'])
        label_min = int(config['Label']['Label_Min'])
        self.label_step = int(config['Label']['Label_Step'])
        self.DIRECTIONS = np.arange(label_min, label_max)
        # Speaker sound
        file_name = config['Speaker_Sound']['File']
        origin_sound_path = self.speaker_sound_path + file_name
        origin_data, channel, origin_sampling, origin_frames = self.wave_read_func(origin_sound_path)
        self.fft_list = np.fft.rfftfreq(origin_frames, 1 / origin_sampling)
        self.freq_min_id = self.freq_ids(self.fft_list, freq_min)
        self.freq_max_id = self.freq_ids(self.fft_list, freq_max)
        self.data_set_freq_len = self.freq_max_id - self.freq_min_id - (self.smooth_step - 1)
        
    def make_data_set(self, individual=False):
        tsp = TSP('./config_tf.ini')
        wave_path = self.recode_data_path + self.data_search(self.date, self.sound_kind, self.geometric,
                                                             plane_wave=self.plane_wave)
        print(wave_path)
        print("Making data set....")
        if not individual:
            data_set = np.empty((0, self.data_set_freq_len), dtype=np.float)
            for data_dir in self.DIRECTIONS:
                sound_data, channel, sampling, frames = self.wave_read_func(wave_path + str(data_dir) + '.wav')
                cut_data = tsp.cut_tsp_data(use_data=sound_data)
                fft_data = np.fft.rfft(cut_data)[0]
                fft_data = fft_data[self.freq_min_id:self.freq_max_id]
                smooth_data = np.convolve(fft_data, np.ones(self.smooth_step) / float(self.smooth_step), mode='valid')
                smooth_data = np.reshape(smooth_data, (1, -1))
                # print(smooth_data.shape)
                data_set = np.append(data_set, smooth_data, axis=0)
                print('finish: ', data_dir + 50 + 1, '/', len(self.DIRECTIONS))
            print('Made data set: ', data_set.shape)
            return np.real(data_set)
        
        else:
            data_set = np.zeros((len(self.DIRECTIONS), tsp.MIC_NUM, tsp.NEED_NUM, self.data_set_freq_len),
                                dtype=np.float)
            for data_dir in self.DIRECTIONS:
                sound_data, channel, sampling, frames = self.wave_read_func(wave_path + str(data_dir) + '.wav')
                cut_data = tsp.cut_tsp_data(use_data=sound_data, individual=True)
                fft_data = np.fft.rfft(cut_data)
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
                print('finish: ', data_dir + 50 + 1, '/', len(self.DIRECTIONS))
            print('Made data set: ', data_set.shape)
            return np.real(data_set)
    
    def make_data_set_recode_individually(self, recode_num, mic_num, target='glass_plate', mic_name='Respeaker'):
        wave_path = self.recode_data_path + self.data_search(self.date, self.sound_kind, self.geometric,
                                                             plane_wave=self.plane_wave)
        data_set = np.zeros((len(self.DIRECTIONS), mic_num, recode_num, self.data_set_freq_len), dtype=np.float)
        for data_id in range(recode_num):
            for data_dir in self.DIRECTIONS:
                sound_data, channel, sampling, frames = \
                    self.wave_read_func(wave_path + target + '_' + str(data_id+1) + '/' + str(data_dir) + '.wav')
                if mic_name == 'Respeaker':
                    sound_data = np.delete(sound_data, [0, 5], 0)
                elif mic_name == 'Matrix':
                    sound_data = np.delete(sound_data, 0, 0)
                else:
                    print("Error")
                    sys.exit()
                if data_dir == -50:
                    print("#######################")
                    print("This mic is ", mic_name)
                    print("Channel ", channel)
                    print("Frames ", frames)
                    print("Object", target)
                    print("#######################")
                fft_data = np.fft.rfft(sound_data)
                for mic in range(fft_data.shape[0]):
                    smooth_data = np.convolve(np.abs(fft_data[mic, self.freq_min_id:self.freq_max_id]),
                                              np.ones(self.smooth_step) / float(self.smooth_step), mode='valid')
                    smooth_data = np.real(np.reshape(smooth_data, (1, -1)))[0]
                    normalize_data = stats.zscore(smooth_data, axis=0)
                    # normalize_data = (smooth_data - smooth_data.mean()) / smooth_data.std()
                    # normalize_data = (smooth_data - min(smooth_data))/(max(smooth_data) - min(smooth_data))
                    data_set[data_dir, mic, data_id, :] = normalize_data

                # print('finish: ', data_dir + 50 + 1, '/', len(self.DIRECTIONS))
            print('finish: ', data_id + 1, '/', recode_num)
            print('***********************************************')
        print('Made data set: ', data_set.shape)
        output_path = self.make_dir_path(array=True)
        np.save(output_path + target + '.npy', data_set)
        
    def pca(self, data_set):
        pca = PCA()
        pca.fit(data_set)
        feature = pca.transform(data_set)
        print(feature.shape)
        # print(feature[:, 0])
        '''寄与率'''
        ev_ratio = pca.explained_variance_ratio_
        ev_ratio = np.hstack([0, ev_ratio.cumsum()])
        print(ev_ratio[0:3])
        '''作図'''
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(feature[:, 0], feature[:, 1], feature[:, 2], marker='o', c=self.DIRECTIONS, cmap='winter')
        '''作図設定'''
        colormap = plt.get_cmap('winter')
        norm = colors.Normalize(vmin=min(self.DIRECTIONS), vmax=max(self.DIRECTIONS))
        mappable = cm.ScalarMappable(cmap=colormap, norm=norm)
        mappable._A = []
        # ax.set_xlim(-29500, -27000)
        # ax.set_ylim(-14000, -11000)
        plt.colorbar(mappable)
        plt.show()
    
    def label(self, label_len, step=None):
        if step is None:
            label_step = self.label_step
        else:
            label_step = step
        if label_len % label_step == 0:
            label = np.zeros((label_len,))
            label_list = np.arange(0, label_len + 1, label_step)[1:]
            post_id = 0
            for label_num, label_id in enumerate(label_list):
                if label_num == 0:
                    label[:label_id] = label_num
                elif label_num == len(label_list):
                    label[label_id:] = label_num
                else:
                    label[post_id:label_id] = label_num
                post_id = label_id
            return label
        else:
            print('Error: Indivisible shape data set', label_len, 'and label step(config)', label_step)
            sys.exit()
    
    @staticmethod
    def support_vector_machine(data_set, svm_label):  # data_set=Noneの時、ラベル用のテスト
        # 線形SVMのインスタンス作成
        svm_model = SVC(kernel='linear', random_state=None)
        # svm_label = self.label(data_set.shape[0])
        
        # モデルの学習, fit関数でおこなう。(ロジスティクス回帰を使うこともできます。)
        svm_model.fit(data_set[:, 0, 0, :], svm_label)
        return svm_model
    
    @staticmethod
    def support_vector_regression(data_set, svr_label):
        model = SVR(kernel='rbf')
        # print(data_set.shape)
        model.fit(data_set[:, 0, 0, :], svr_label)
        return model


if __name__ == '__main__':
    pe = PrametricEigenspace()
    '''データセットを作成するため'''
    # data = pe.make_data_set(individual=True)
    # np.save('normalize.npy', data)
    # pe.pca(data)
    ''''''
    data = np.load('./normalize.npy')
    label_all = pe.label(data.shape[0])
    svm = pe.support_vector_machine(data, label_all)
    svr = pe.support_vector_regression(data, pe.DIRECTIONS)
    for i in range(data.shape[2] - 1):
        print('******************************')
        pred_train = svm.predict(data[:, 0, i + 1, :])
        # print(np.reshape(pred_train, (-1, deg_step)))
        accuracy_train = accuracy_score(label_all, pred_train)
        print(accuracy_train)
        # print(confusion_matrix(label, pred_train))

    pred_train_svr = svr.predict(data[:, 0, 0, :])
    plt.figure()
    plt.plot(pe.DIRECTIONS, pred_train_svr)
    plt.ylim(-50, 50)
    plt.show()
