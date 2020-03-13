# coding:utf-8
from matplotlib import pyplot as plt
import numpy as np
from _function import MyFunc
# from _tf_generate_tsp import TSP
from sklearn.decomposition import PCA
from matplotlib import colors
from matplotlib import cm
# from scipy import stats
from configparser import ConfigParser
# from distutils.util import strtobool
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import confusion_matrix
# from sklearn.preprocessing import StandardScaler
# from mic_setting import MicSet


class PrametricEigenspace(MyFunc):
    def __init__(self, data_set_file_path, data, mic_id, use_data_id):
        super().__init__()
        self.data_set_path = self.onedrive_path + data_set_file_path + data + '.npy'
        self.data_name = data
        self.mic = mic_id
        self.data_id = use_data_id
        origin_path = self.speaker_sound_path + '2_up_tsp_1num.wav'
        smooth_step = 50
        self.freq_list = np.fft.rfftfreq(self.get_frames(origin_path), 1/44100)
        self.freq_max_id = self.freq_ids(self.freq_list, 7000)
        self.freq_min_id = self.freq_ids(self.freq_list, 1000)
        self.freq_list = self.freq_list[self.freq_min_id + int(smooth_step/2) - 1:self.freq_max_id - int(smooth_step/2)]

        self.data_set_freq_len = self.freq_max_id - self.freq_min_id - (smooth_step - 1)
        self.use_fft_list = self.freq_list[self.freq_min_id + int(smooth_step/2) - 1:self.freq_max_id
                                                                                         - int(smooth_step/2)]
        
    def pca_check(self):
        data_set = np.load(self.data_set_path)
        ss = StandardScaler()
        DIRECTIONS = np.arange(data_set.shape[0]/2 * (-1), data_set.shape[0]/2)
        print(DIRECTIONS)
        print(data_set.shape, data_set[:, 0, 0, :].shape)
        data_set = ss.fit_transform(data_set[:, 0, 0, :])
        pca = PCA()
        pca.fit(data_set)
        feature = pca.transform(data_set)
        pc1 = pca.components_[0]
        pc2 = pca.components_[1]
        pc3 = pca.components_[2]
        # print(feature.shape)
        # print(feature[:, 0])
        '''寄与率'''
        ev_ratio = pca.explained_variance_ratio_
        ev_ratio = np.hstack([0, ev_ratio.cumsum()])
        print('### Cumulative contribution: ', ev_ratio[0:3])
        
        '''作図'''
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(feature[:, 0], feature[:, 1], feature[:, 2], marker='o', c=DIRECTIONS, cmap='winter')
        '''作図設定'''
        colormap = plt.get_cmap('winter')
        norm = colors.Normalize(vmin=min(DIRECTIONS), vmax=max(DIRECTIONS))
        mappable = cm.ScalarMappable(cmap=colormap, norm=norm)
        mappable._A = []
        plt.colorbar(mappable)
        ax.set_xlabel("PC1", fontsize=12)
        ax.set_ylabel("PC2", fontsize=12)
        ax.set_zlabel("PC3", fontsize=12)
        plt.show()
        
        # plt.figure()
        # plt.scatter(feature[:, 0], feature[:, 1], marker='o', c=label, cmap='winter')
        # colormap = plt.get_cmap('winter')
        # norm = colors.Normalize(vmin=min(self.DIRECTIONS), vmax=max(self.DIRECTIONS))
        # mappable = cm.ScalarMappable(cmap=colormap, norm=norm)
        # mappable._A = []
        # plt.colorbar(mappable)
        # plt.show()
        
        # plt.figure()
        # plt.scatter(feature[630:, 0], feature[630:, 1], label='Glass')
        # plt.scatter(feature[:630, 0], feature[:630, 1], label='Card')
        # plt.xlabel('PC1')
        # plt.ylabel('PC2')
        # plt.legend()
        # plt.show()
        #
        # fig = plt.figure()
        # ax1 = fig.add_subplot(3, 1, 1)
        # ax2 = fig.add_subplot(3, 1, 2)
        # ax3 = fig.add_subplot(3, 1, 3)
        # ax1.plot(self.use_fft_list, np.abs(pc1), label='PC1', c='c')
        # ax2.plot(self.use_fft_list, np.abs(pc2), label='PC2', c='m')
        # ax3.plot(self.use_fft_list, np.abs(pc3), label='PC3', c='y')
        # axs = plt.gcf().get_axes()

        # 軸毎にループ
        # for ax in axs:
        #     # 現在の軸を変更
        #     plt.axes(ax)
        #
        #     # 凡例を表示
        #     plt.legend(loc='upper left')
        #
        #     # 軸の範囲
        #     plt.ylim([0, max([np.max(np.abs(pc1)), np.max(np.abs(pc2)), np.max(np.abs(pc3))])])
        # plt.xlabel('Frequency [Hz]')
        # # plt.show()
        # img_path = self.make_dir_path(img=True)
        # plt.savefig(img_path + 'pca_' + self.data_name + '.png')
        
        # plt.figure()
        # # plt.scatter(feature[:, 0], feature[:, 1], marker='o', c=label, cmap='winter')
        # arrow_mul = 15
        # text_mul = 1.1
        # use_fft_list =self.fft_list[self.freq_min_id:self.freq_max_id]
        # feature_names = ["{0}".format(int(i)) for i in use_fft_list]
        # test = range(0, pc1.shape[0], 50)
        # pc1_list = []
        # pc2_list = []
        # for i in test:
        #     plt.arrow(0, 0,
        #               pc1[i] * arrow_mul, pc2[i] * arrow_mul, alpha=0.5)
        #     plt.text(pc1[i] * arrow_mul * text_mul,
        #              pc2[i] * arrow_mul * text_mul,
        #              feature_names[i],
        #              color='r')
        #     pc1_list.append(pc1[i] * arrow_mul)
        #     pc2_list.append(pc2[i] * arrow_mul)
        # plt.scatter(pc1_list, pc2_list, marker='o', c=test, cmap='spring')
        # colormap = plt.get_cmap('spring')
        # norm = colors.Normalize(vmin=min(use_fft_list), vmax=max(use_fft_list))
        # mappable = cm.ScalarMappable(cmap=colormap, norm=norm)
        # mappable._A = []
        # plt.colorbar(mappable)
        # plt.xlim(-1, 1)
        # plt.ylim(-1, 1)
        # plt.show()


if __name__ == '__main__':
    data_path = 'Model/191125_anechonic/'
    data_name = 'anechoic'
    mic = 0
    use_data = 0
    pe = PrametricEigenspace(data_path, data_name, mic, use_data)
    pe.pca_check()
