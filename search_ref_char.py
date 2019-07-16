import numpy as np
from matplotlib import pyplot as plt
from beamforming import BeamForming
from shape_from_sound import ShapeFromSound


def create_amp_data(self, true_direction, origin_freq_list, sign='plus', check_spectrogram=False):
    """

    :param true_direction:
    :param origin_freq_list:
    :param sign:
    :param check_spectrogram:
    :return: (true direction(9), ss_list(4), origin_freq_list(4), mic_list(4))
    """
    sfs = ShapeFromSound('./config.ini')
    file_path = '../_exp/190612/test_data/'
    mic_list = ['m1', 'm2', 'm3', 'm4']
    amp_array = np.zeros((len(sfs.ss_list), len(origin_freq_list), len(mic_list)), dtype=np.float)
    for k, ss_name in enumerate(sfs.ss_list):
        # print(ss_name)
        if ss_name == -0.25:
            data_path = 'left_25'
        elif ss_name == -0.15:
            data_path = 'left_15'
        elif ss_name == 0.15:
            data_path = 'right_15'
        elif ss_name == 0.25:
            data_path = 'right_25'
        else:
            data_path = 0
            print("ERROR")
        if sign == 'plus':
            wave_data = file_path + data_path + '_' + str(true_direction) + '.wav'
        else:
            wave_data = file_path + data_path + '_m' + str(true_direction) + '.wav'
        sound_data, w_channel, w_sampling_rate, w_frames_num = self.wave_read_func(wave_data)
        print(sound_data.shape)
        plt.specgram(sound_data[0, :], Fs=w_sampling_rate)
        plt.title('origin' + data_path + '_' + str(true_direction) + '.wav')
        plt.clim(0, 20)
        plt.colorbar()
        plt.show()
        print("##########################################")
        print('wave data:' + wave_data)
        use_data = reshape_sound_data(sound_data, w_sampling_rate, 10, 1, 4, 2)
        if check_spectrogram:
            for m, freq in enumerate(origin_freq_list):
                for j, ss in enumerate(mic_list):
                    plt.specgram(use_data[m, j, :], Fs=w_sampling_rate)
                    plt.title('origin' + str(freq) + '_' + ss + '_' + str(true_direction))
                    plt.clim(-20, 20)
                    plt.colorbar()
                    plt.show()
        size = 512
        fs = 44100  # サンプリングレート
        start = int(fs / 2)
        freq_min = 1000
        freq_max = 9000
        fft_data, ifft_data = band_pass_filter(size, fs, start, use_data, freq_min, freq_max)
        amp_list = np.zeros((len(origin_freq_list), len(mic_list)))
        for m, freq in enumerate(origin_freq_list):
            for j, ss in enumerate(mic_list):
                amp = max([np.sqrt(c.real ** 2 + c.imag ** 2) for c in fft_data[m, j, :]])
                # amp_id = np.argmax([np.sqrt(c.real ** 2 + c.imag ** 2) for c in fft_data[m, j, :]]) #確認用
                amp_list[m, j] = amp
                # print(amp_id)
        amp_array[k, :, :] = amp_list
    return amp_array


def reshape_sound_data(sound_data, rate, interval_time, want_data_time, combine_num, start_time):
    print("Frame Rate : ", rate)
    print("Sound Data : ", sound_data.shape)

    use_sound_data = np.zeros((combine_num, sound_data.shape[0], want_data_time * rate), dtype=np.complex)
    start_time = start_time * rate
    for k in range(combine_num):
        use_sound_data[k, :, :] = sound_data[:, start_time: start_time + (want_data_time * rate)]

        start_time += interval_time * rate

    # print('sccsess make use data :', use_sound_data.shape)
    return use_sound_data


def all_sound_source_data_set(self, true_direction, sign='plus', spec=False):
    file_path = '../_exp/190619/test_data/'
    ss_list = [-0.15, -0.10, 0.10, 0.15]
    freq_list = np.zeros((len(ss_list), 4, int(44100 * 14)), dtype=np.complex)
    for ss_id, ss_name in enumerate(ss_list):
        print(ss_name)
        if ss_name == -0.15:
            data_path = 'left_15'
        elif ss_name == -0.10:
            data_path = 'left_10'
        elif ss_name == 0.10:
            data_path = 'right_10'
        elif ss_name == 0.15:
            data_path = 'right_15'
        else:
            data_path = 0
            print("ERROR")
        if sign == 'plus':
            wave_data = file_path + data_path + '_20_' + str(true_direction) + '_2000.wav'
        else:
            wave_data = file_path + data_path + '_20_m' + str(true_direction) + '_2000.wav'
        print('===============================================')
        print(wave_data)
        print('===============================================')
        sound_data, w_channel, w_sampling_rate, w_frames_num = self.wave_read_func(wave_data)
        print(sound_data.shape)
        if spec:
            plt.specgram(sound_data[0, :], Fs=w_sampling_rate)
            plt.title('origin' + data_path + '_' + str(true_direction) + '.wav')
            # plt.clim(0, 20)
            plt.colorbar()
            plt.show()
        use_data = reshape_sound_data(sound_data, w_sampling_rate, 0, 14, 1, 0)
        # print('use_data shape', use_data.shape)
        fft_data, ifft_data = band_pass_filter(14 * w_sampling_rate, w_sampling_rate, 0, use_data,
                                               freq_min=100, freq_max=2000)
        print("filter data:", ifft_data.shape)
        freq_list[ss_id, :, :] = ifft_data
        # np.save('../_array/ifft_data', ifft_data)
    return freq_list


def band_pass_filter(size, fs, start, data, freq_min, freq_max):
    d = 1.0 / fs
    hammingWindow = np.hamming(size)
    freq_list = np.fft.fftfreq(size, d)
    # print(freq_list)
    print("filFft data :", data.shape)
    print('size', size)
    windowedData = hammingWindow + data[:, :, start:start + size]  # 切り出し波形データ(窓関数)
    data = np.fft.fft(windowedData)
    id_min = np.abs(freq_list - freq_min).argmin()
    id_max = np.abs(freq_list - freq_max).argmin()
    bpf_distribution = np.ones((size,), dtype=np.complex)
    bpf_distribution[int(-1 * id_min):] = 0
    bpf_distribution[id_max:int(-1 * id_max)] = 0
    bpf_distribution[:id_min] = 0
    fft_bpf_data = data * bpf_distribution
    ifft_bpf_data = np.fft.ifft(fft_bpf_data)
    return fft_bpf_data, ifft_bpf_data


def plot_data(data, freq_list, origin_freq_list, mic_list, w_sampling_rate,
              before_bpf=False, after_bpf=False, after_bpf_spectrogram=False):
    if before_bpf:
        for k, freq in enumerate(origin_freq_list):
            for j, ss in enumerate(mic_list):
                plt.plot(freq_list, abs(data[k, j, :]))
                # plt.ylim(0, 2.0 * 10**7)
                plt.xlim([0, w_sampling_rate / 4])
                plt.title('before filter' + str(freq) + ss)
                plt.show()

    if after_bpf:
        for k, freq in enumerate(origin_freq_list):
            for j, ss in enumerate(mic_list):
                plt.plot(freq_list, abs(data[k, j, :]))
                # plt.ylim(0, 2.0 * 10**7)
                # plt.xlim([0, fs/4])
                plt.title('after filter' + str(freq) + ss)
                plt.savefig('../Image/190618/after_filter' + str(freq) + ss + '.png')
                plt.show()

    if after_bpf_spectrogram:
        for k, freq in enumerate(origin_freq_list):
            for j, ss in enumerate(mic_list):
                plt.specgram(data[k, j, :], Fs=w_sampling_rate)
                plt.title('ifft' + str(freq) + ss)
                plt.clim(0, 20)
                plt.colorbar()
                plt.show()


if __name__ == '__main__':
    bm = BeamForming('./config.ini')
    direction_list = [0, 10, 20, 30, 40]
    amp_data = np.zeros((9, 4, 4, 4))
    fi_array = np.zeros((9, 4, 4, int(14 * 44100)), dtype=np.float)
    for i, name in enumerate(direction_list):
        # amp_data[i, :, :, :] = create_amp_data(bm, name, bm.odigin_freq_list)
        fi_array[i, :, :, :] = all_sound_source_data_set(bm, name)
    for i, name in enumerate(direction_list[1:]):
        # amp_data[i+5, :, :, :] = create_amp_data(bm, name, bm.odigin_freq_list, sign='minus')
        fi_array[i+5, :, :, :] = all_sound_source_data_set(bm, name, sign='minus')
    # np.save('../_array/amp_data', amp_data)
    np.save('../_array/fi_array', fi_array)
    print("success save")