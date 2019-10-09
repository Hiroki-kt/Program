# coding:utf-8
import numpy as np
from matplotlib import pyplot as plt
import math
from sklearn.decomposition import PCA
import scipy.signal


def main(normal=False, intensity=False, amp=False, freq_model=False, pca=False, freq_time=False, freq_time_alone=None,
         model_diff_psi=None, beam_time=None):

    ss_list = ['s4', 's3', 's2', 's1']
    mic_list = ['mic1', 'mic2', 'mic3', 'mic4']
    true_list = [-40, -30, -20, -10, 0, 10, 20, 30, 40]
    origin_frq_list = [3000, 5000, 7000, 2000]
    # print(freq_list_2)
    # print(intensity_array)
    # (true direction(9), ss_list(4), origin_freq_list(4), mic_list(4))

    if normal:
        theta_array = np.load('../_array/190618_fix_s/theta_array.npy')
        print("theta array shape:", theta_array.shape)
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
        intensity_array = np.load('../_array/190618_fix_s/intensity_array.npy')
        print("intensity array", intensity_array.shape)
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
                plt.savefig('../_img/190618/intensity_' + str(name) + 'Hz_' + dis + '.png')
                plt.show()

    if amp:
        amp_array = np.load('../_array/amp_data.npy')
        print("amp array:", amp_array.shape)
        amp_array = amp_array[[8, 7, 6, 5, 0, 1, 2, 3, 4], :, :, :]
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
                plt.savefig('../_img/190618/amp' + str(name) + 'Hz_' + mic + dis + '.png')
                plt.show()

    if freq_model:
        amp_array = np.load('../_array/amp_data.npy')
        print("amp array:", amp_array.shape)
        amp_array = amp_array[[8, 7, 6, 5, 0, 1, 2, 3, 4], :, :, :]
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
        amp_array = np.load('../_array/amp_data.npy')
        print("amp array:", amp_array.shape)
        amp_array = amp_array[[8, 7, 6, 5, 0, 1, 2, 3, 4], :, :, :]
        a = np.arange(-40, 41, 10)
        pca_x = np.reshape(amp_array[:, :, :, 2], (9, 16))
        pca_y = np.insert(pca_x, 16, a.T, axis=1)
        pca = PCA(n_components=4)
        pca.fit(pca_x)
        print('寄与率:', pca.explained_variance_ratio_)
        transformed = pca.fit_transform(pca_y)
        print(transformed.shape)
        # 主成分をプロットする
        for i in range(9):
            plt.scatter(transformed[i, 0], transformed[i, 1], label=a[i])
        plt.title('principal component')
        plt.xlabel('pc1')
        plt.ylabel('pc2')
        plt.legend()
        plt.savefig('../_img/190619/pca.png')
        plt.show()

        for i in range(4):
            plt.plot(a, transformed[:, i])
            plt.title('pc' + str(i + 1))
            plt.ylim(-200000, 300000)
            plt.xlabel('direction [°]')
            plt.savefig('../_img/190619/pca_pc' + str(i + 1) + '.png')
            plt.show()

    if freq_time:
        fi_array = np.load('../_array/fi_array.npy')
        print("filter array:", fi_array.shape)
        fi_array = fi_array[[8, 7, 6, 5, 0, 1, 2, 3, 4], :, :, :]
        time_list = np.arange(0, 14, 1 / 44100)
        for i, name in enumerate(mic_list):
            for k, ss in enumerate(ss_list):
                for j, theta in enumerate(true_list):
                    max_id = scipy.signal.argrelmax(fi_array[j, k, i, :], order=1000)
                    plt.plot(time_list[max_id[0]], abs(fi_array[j, k, i, :][max_id[0]]), label=str(theta))
                    plt.title('freq time_' + str(ss) + '_' + str(name))
                    plt.xlabel('time[s]')
                    plt.ylabel('amp')
                    plt.ylim(0, 1500)
                    plt.legend()
                    # plt.show()
                plt.show()

    if freq_time_alone:
        '''
        freq_time_alone = [ss_id, mic_id]
        '''
        fi_array = np.load('../_array/fi_array.npy')
        print("filter array:", fi_array.shape)
        fi_array = fi_array[[8, 7, 6, 5, 0, 1, 2, 3, 4], :, :, :]
        time_list = np.arange(0, 14, 1 / 44100)
        freq_list_2 = (2000 - 100) / (14 - 0) * time_list + 100
        model_array = np.load('../_array/model_array.npy')
        global axis
        freq_model = np.arange(100, 2000)
        print(freq_time_alone[0], freq_time_alone[1])
        fig = plt.figure(figsize=(7, 6))
        for j, theta in enumerate(true_list):
            max_id = scipy.signal.argrelmax(fi_array[j, freq_time_alone[0], freq_time_alone[1], :], order=100)
            print("success get peak")
            axis = fig.add_subplot(len(true_list), 1, j + 1)
            axis.plot(freq_list_2[max_id[0]],
                      abs(fi_array[j, freq_time_alone[0], freq_time_alone[0], :][max_id[0]]) / max(
                          abs(fi_array[j, freq_time_alone[0], freq_time_alone[0], :][max_id[0]])),
                      label='Exp data')
            axis.plot(freq_model, model_array[j, :] / max(model_array[j, :]), label='Model')
            plt.ylim(0, 1)
            plt.ylabel('$\psi$=' + str(theta) + '°', rotation=0)
            axis.yaxis.set_label_coords(-0.15, 0.5)
            plt.setp(axis.get_xticklabels(), visible=False)
        plt.setp(axis.get_xticklabels(), visible=True)
        fig.tight_layout()
        plt.xlabel('freq[Hz]')
        plt.legend(bbox_to_anchor=(-0.2, -2), loc='lower left')
        plt.show()

    if model_diff_psi:
        freq_model = np.arange(100, 2000)
        max_freq = 2000
        min_freq = 100
        fi_array = np.load('../_array/fi_array.npy')
        model_array = np.load('../_array/model_array.npy')
        print("filter array:", fi_array.shape)
        fi_array = fi_array[[8, 7, 6, 5, 0, 1, 2, 3, 4], :, :, :]
        time_list = np.arange(0, 14, 1 / 44100)
        freq_list_2 = (2000 - 100) / (14 - 0) * time_list + 100
        id_min, id_max = freq_id(freq_list_2, min_freq, max_freq)
        fig = plt.figure(figsize=(5, 7))
        for j, theta in enumerate(true_list):
            exp_data = fi_array[j, model_diff_psi[0], model_diff_psi[1], id_min:id_max]
            # print("exp-data", exp_data.shape)
            max_id = scipy.signal.argrelmax(exp_data, order=200)
            # print("success get peak", max_id[0].shape)
            axis = fig.add_subplot(len(true_list), 1, j + 1)
            peak_data = exp_data[max_id[0]]
            model_data = model_array[j, min_freq-100:max_freq-100]
            normalize_exp_data = (peak_data - min(peak_data)) / (max(peak_data) - min(peak_data))
            normalize_model_data = (model_data - min(model_data)) / (max(model_data) - min(model_data))
            # print("normalized data:", normalize_model_data.shape, normalize_exp_data.shape)
            axis.plot(freq_list_2[id_min:id_max][max_id[0]], normalize_exp_data, label='exp-data')
            plt.plot(freq_model[min_freq-100:max_freq-100], normalize_model_data, label='model-data')
            plt.xlim(100, 2000)
            plt.ylim(0, 1)
            plt.ylabel('$\psi$=' + str(theta) + '°', rotation=0)
            axis.yaxis.set_label_coords(-0.15, 0.5)
            plt.setp(axis.get_xticklabels(), visible=False)
        plt.setp(axis.get_xticklabels(), visible=True)
        plt.xlabel('freq[Hz]')
        plt.legend(bbox_to_anchor=(-0.25, -1.7), loc='lower left')
        plt.show()

    if beam_time:
        path = '../_array/' + beam_time
        bm_time = np.load(path + '_bm_time.npy')
        time = np.load(path + '_time.npy')
        model_array = np.load('../_array/190708/030633_model_array.npy')
        freq_exp = (2500 - 100) / (50 - 0) * time + 100
        freq_model = np.arange(100, 2000)
        max_freq = 2000
        min_freq = 100
        id_min = np.abs(freq_exp - min_freq).argmin()
        id_max = np.abs(freq_exp - max_freq).argmin()
        # print("exp_data:", bm_time.shape)
        # print("model_data:", model_array.shape)
        fig = plt.figure(figsize=(5, 7))
        for j, theta in enumerate(true_list):
            axis = fig.add_subplot(len(true_list), 1, j + 1)
            model_data = model_array[j, min_freq-100:max_freq-100]
            plot_freq_exp = freq_exp[id_min:id_max]
            exp_data = bm_time[j, id_min:id_max]
            peak_id = scipy.signal.argrelmax(exp_data, order=50)
            peak_data = exp_data[peak_id[0]]
            # print(plot_freq_exp.shape)
            # print(peak_data.shape)
            normalize_exp_data = (peak_data - min(peak_data)) / (max(peak_data) - min(peak_data))
            normalize_model_data = (model_data - min(model_data)) / (max(model_data) - min(model_data))
            plt.plot(plot_freq_exp[peak_id], normalize_exp_data, label='Measured')
            plt.plot(freq_model[min_freq-100:max_freq-100], normalize_model_data, label='Model')
            plt.xlim(100, 2000)
            # plt.ylim(0, 1)
            # plt.ylabel('$\psi$=' + str(theta) + '°', rotation=0)
            # axis.yaxis.set_label_coords(-0.15, 0.5)
            # plt.setp(axis.get_xticklabels(), visible=True)
            plt.ylabel('Amp ($\psi$=' + str(theta) + '°)')
            plt.xlabel('Frequency[Hz]')
            axis.yaxis.set_label_coords(-0.15, 1)
            print('OK', theta)
            plt.show()
        plt.setp(axis.get_xticklabels(), visible=True)
        plt.xlabel('Freq[Hz]')
        plt.legend(bbox_to_anchor=(-0.25, -1.7), loc='lower left')
        plt.savefig('../_img/190708/result.png')
        # plt.show()
        
        
def difference_mid_hig(target_freq_mid, target_freq_hig, directory, file_name):
    # モデルの比較
    true_list = [-40, -30, -20, -10, 0, 10, 20, 30, 40]
    model_array = np.load('../_array/190708/030633_model_array.npy')
    # 500Hz付近でのパワーの大きさの中での最大値（中域の代表値）
    model_power_mid = np.average(model_array[:, target_freq_mid-200:target_freq_mid], axis=1)
    model_power_hig = np.average(model_array[:, target_freq_hig-200:target_freq_hig], axis=1)
    model_mid_hig_rate = 1.1 * model_power_mid/model_power_hig
    
    # 計測値の比較
    path = '../_array/' + str(directory) + str(file_name)
    bm_time = np.load(path + '_bm_time.npy')
    time = np.load(path + '_time.npy')
    freq_exp = (2500 - 100) / (50 - 0) * time + 100
    id_mid, id_hig = freq_id(freq_exp, target_freq_mid, target_freq_hig)
    id_500, id_600 = freq_id(freq_exp, 500, 600) # 50Hzの差のフレーム数
    interval = id_600 - id_500
    power_mid = np.average(bm_time[:, id_mid-interval:id_mid+interval], axis=1)
    power_hig = np.average(bm_time[:, id_hig-interval:id_hig+interval], axis=1)
    exp_mid_hig_rate = power_mid/power_hig

    plt.figure(figsize=(5, 7))
    plt.plot(true_list, exp_mid_hig_rate, label='計測')
    plt.plot(true_list, model_mid_hig_rate, label='モデル')
    plt.xlabel('$\psi$[deg]')
    plt.ylabel('$e_m$/$e_h$')
    # plt.title(path)
    plt.savefig('../_img/190708/_rate.png')
    plt.show()
    

def freq_id(freq_list, freq_min, freq_max):
    id_min = np.abs(freq_list - freq_min).argmin()
    id_max = np.abs(freq_list - freq_max).argmin()
    return id_min, id_max


if __name__ == '__main__':
    dire = '190704/'
    file_name = '0704_dis20_0'
    difference_mid_hig(500, 1900, dire, file_name)
    main(beam_time=dire + file_name)
    # true_list = [-40, -30, -20, -10, 0, 10, 20, 30, 40]
    # bm_time = np.load('../_array/190704/bm_time.npy')
    # time = np.load('../_array/190704/time.npy')
    # freq_list = (2500 - 100) / (50 - 0) * time + 100
    # max_freq = 2000
    # min_freq = 1000
    # id_min = np.abs(freq_list - min_freq).argmin()
    # id_max = np.abs(freq_list - max_freq).argmin()
    # a = bm_time[:, id_min: id_max]
    # normalize_exp_data = np.zeros_like(a)
    # for i in range(9):
    #     normalize_exp_data[i, :] = (a[i, :] - min(a[i, :])) / (max(a[i, :]) - min(a[i, :]))
    # print(normalize_exp_data.shape)
    # normalize_exp_data = normalize_exp_data.sum(axis=1)
    # print(normalize_exp_data.shape)
    # plt.plot(true_list, normalize_exp_data)
    # plt.show()