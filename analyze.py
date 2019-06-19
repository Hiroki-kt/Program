import numpy as np
from matplotlib import pyplot as plt
import math
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn import datasets

def main(normal=False, intensity=False, amp=False, freq=False, pca=False):
    theta_array = np.load('../_array/190618_fix_s/theta_array.npy')
    normal_array = np.load('../_array/190618_fix_s/normal_array.npy')
    rho_array = np.load('../_array/190618_fix_s/rho_array.npy')
    intensity_array = np.load('../_array/190618_fix_s/intensity_array.npy')
    amp_array = np.load('../_array/amp_data.npy')

    print(theta_array.shape)
    print(normal_array.shape)
    print(rho_array.shape)
    print(intensity_array.shape)
    print(amp_array.shape)

    origin_frq_list = [3000, 5000, 7000, 2000]
    ss_list = ['s4', 's3', 's2', 's1']
    mic_list = ['mic1', 'mic2', 'mic3', 'mic4']
    true_list = [-40, -30, -20, -10, 0, 10, 20, 30, 40]
    theta_array = theta_array[:, [8, 7, 6, 5, 0, 1, 2, 3, 4]]
    intensity_array = intensity_array[:, :, [8, 7, 6, 5, 0, 1, 2, 3, 4]]
    amp_array = amp_array[[8, 7, 6, 5, 0, 1, 2, 3, 4], :, :, :]
    # print(intensity_array)
    # (true direction(9), ss_list(4), origin_freq_list(4), mic_list(4))

    if normal:
        for i, name in enumerate(origin_frq_list):
            plt.scatter(true_list, theta_array[i, :], label='predict')
            plt.plot(true_list, true_list, label='truth')
            plt.title('surface normal direction(' + str(name) + 'Hz)')
            plt.ylim(-50, 60)
            plt.xlabel('direction[°]')
            plt.ylabel('direction[°]')
            plt.legend()
            plt.show()

    if intensity:
        for i, name in enumerate(origin_frq_list):
            for j, dis in enumerate(ss_list):
                plt.plot(true_list, intensity_array[j, i, :])
                if dis == 's4':
                    theta = math.atan(1 / 2) / 2
                elif dis == 's3':
                    theta = math.atan(3 / 10) / 2
                elif dis == 's2':
                    theta = -1 * math.atan(3 / 10) / 2
                else:
                    theta = -1 * math.atan(1 / 2) / 2
                y = np.linspace(0, max(intensity_array[j, i, :]))
                x = 0 * y + math.degrees(theta)
                plt.plot(x, y)
                plt.title('intensity(' + str(name) + 'Hz , ' + dis + ')')
                plt.ylim(0, 700)
                plt.xlabel('direction[°]')
                plt.ylabel('intensity')
                # plt.legend()
                plt.savefig('../_img/190618/intensity_' + str(name) + 'Hz_' + dis +'.png')
                plt.show()

    if amp:
        dis = 's1'
        for j, mic in enumerate(mic_list):
            for i, name in enumerate(origin_frq_list):
                plt.plot(true_list, amp_array[:, 0, i, j])
                # print(amp_array[:, j, i, 0])
                if dis == 's4':
                    theta = math.atan(1 / 2) / 2
                elif dis == 's3':
                    theta = math.atan(3 / 10) / 2
                elif dis == 's2':
                    theta = -1 * math.atan(3 / 10) / 2
                else:
                    theta = -1 * math.atan(1 / 2) / 2
                y = np.linspace(0, max(amp_array[:, 0, i, j]))
                x = 0 * y + math.degrees(theta)
                plt.plot(x, y)
                plt.title('amp(' + str(name) + 'Hz , ' + mic + ')')
                plt.ylim(0, 700000)
                plt.xlabel('direction[°]')
                plt.ylabel('amp')
                # plt.legend()
                plt.savefig('../_img/190618/amp' + str(name) + 'Hz_' + mic + dis +'.png')
                plt.show()

    if freq:
        for j, mic in enumerate(mic_list):
            for i, name in enumerate(origin_frq_list):
                plt.plot(true_list, amp_array[:, 0, i, j], label=name)
                # print(amp_array[:, j, i, 0])
            plt.title('amp(' + mic + ')')
            plt.ylim(0, np.max(amp_array[:, 0, :, :]))
            plt.xlabel('direction[°]')
            plt.ylabel('amp')
            plt.legend()
            plt.savefig('../_img/190618/amp' + mic + '.png')
            plt.show()

    if pca:
        #print(amp_array[:,:,:,2])
        a = np.arange(-40, 41, 10)
        #print(a)
        X = np.reshape(amp_array[:, :, :, 2],(9, 16))
        Y = np.insert(X,16, a.T, axis=1)
        #print(Y)
        #print(Y.shape)
        pca = PCA(n_components=4)
        pca.fit(X)
        print('寄与率:', pca.explained_variance_ratio_)
        transformed = pca.fit_transform(Y)
        print(transformed.shape)
        # data = datasets.load_iris()
        # print(data)
        # 主成分をプロットする
        for i in range(9):
            plt.scatter(transformed[i, 0],transformed[i, 1], label=a[i])
        plt.title('principal component')
        plt.xlabel('pc1')
        plt.ylabel('pc2')
        plt.legend()
        plt.savefig('../_img/190619/pca.png')
        plt.show()

        for i in range(4):
            plt.plot(a, transformed[:, i])
            plt.title('pc' + str(i+1))
            plt.ylim(-200000, 300000)
            plt.xlabel('direction [°]')
            plt.savefig('../_img/190619/pca_pc' + str(i+1) + '.png')
            plt.show()


if __name__ == '__main__':
    main(pca=True)
