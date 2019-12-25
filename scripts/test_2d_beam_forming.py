import numpy as np
import math
from _function import MyFunc
from scipy import signal
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class TransferFunction2d:
    def __init__(self, grid_size, cell_size, z_dis, mic_r, mic_num):
        self.mic_position = self.mic_positions(mic_r, mic_num)
        self.ss_position = self.ss_positions(grid_size, cell_size, z_dis)
        
    def ss_vector(self, cell_num, dis):
        s = math.sqrt(self.ss_position[0, cell_num] ** 2 +
                      self.ss_position[1, cell_num] ** 2 +
                      self.ss_position[2, cell_num] ** 2)
        x = dis * self.ss_position[0, cell_num]/s
        y = dis * self.ss_position[1, cell_num]/s
        z = dis * self.ss_position[2, cell_num]/s
        return x, y, z
    
    @staticmethod
    def mic_positions(mic_r, mic_num):
        # return mic positions of mic array
        mic_position = np.zeros((mic_num, 3))
        for mic_id in range(mic_num):
            theta = mic_id / mic_num * 360
            mic_position[mic_id] = Position(mic_r, theta).pos()
        print('#Create circular microphone array position')
        # print(mic_positions)
        return mic_position

    @staticmethod
    def ss_positions(grid_size, cell_size, z_dis):
        # 単位 cm
        ss_pos_list = []
        ss_position = np.ones((3, int(grid_size[0]/cell_size[0] + 1), int(grid_size[1]/cell_size[1] + 1))) * z_dis
        x = np.tile(np.arange(-grid_size[0]/2, grid_size[0]/2 + 1, cell_size[0]),
                    (int(grid_size[1]/cell_size[1] + 1), 1))
        y = np.tile(np.arange(grid_size[1]/2, -grid_size[1]/2 - 1, -cell_size[1]),
                    (int(grid_size[0]/cell_size[0] + 1), 1)).T
        ss_position[0, :, :] = x
        ss_position[1, :, :] = y
        ss_position = np.reshape(ss_position, (3, -1))
        # print(ss_position)
        print('#Create temporal sound source position list')
        return ss_position
    
    def steering_vector(self, n, d, w_channel):
        freq_array = np.fft.fftfreq(n, d)
        freq_num = freq_array.shape[0]
        # freq_array = self.freq_list
        # freq_num = freq_array.shape[0]
        tf = np.zeros((self.ss_position.shape[1], freq_num, w_channel), dtype=np.complex)
        # beam_conf = np.zeros((temp_ss_num, freq_num, w_channel), dtype=np.complex)
    
        # create tf
        # l_w = math.pi * freq_array / self.sound_speed
        freq_repeat_array = np.ones((freq_num, w_channel), dtype=np.complex) * freq_array.reshape(
            (freq_num, -1)) * -1j * 2 * np.pi  # フーリエ変換
        size = 10
        sound_speed = 340 * 100
        for k in range(int(self.ss_position.shape[1])):
            sx, sy, sz = self.ss_vector(k, size)
            center2ss_dis = math.sqrt(sx ** 2 + sy ** 2 + sz**2)
            mic2ss_dis = np.sqrt((self.mic_position[:, 0] - sx) ** 2 +
                                 (self.mic_position[:, 1] - sy) ** 2 +
                                 (self.mic_position[:, 2] - sz) ** 2)
            # print(center2ss_dis)
            dis_diff = (mic2ss_dis - center2ss_dis) / sound_speed  # * self.w_sampling_rate 打消
            # print(dis_diff)
            dis_diff_repeat_array = np.ones((freq_num, w_channel)) * dis_diff.reshape((-1, w_channel))
            tf[k, :, :] = np.exp(freq_repeat_array * dis_diff_repeat_array)
            # beam_conf[k,:,:] = tf[k,:,:]/ ()
        print('#Create transfer funtion', tf.shape)
        tf = tf.conj()
        return tf
        

class Position(object):
    def __init__(self, r, theta):
        # r[m], theta[deg]
        theta = theta * math.pi / 180
        self.x = r * math.cos(theta)
        self.y = r * math.sin(theta)
        self.z = 0

    def pos(self):
        return [self.x, self.y, self.z]
    
    
if __name__ == '__main__':
    # 単位はcm
    grid_size = [100, 100]
    cell_size = [5, 5]
    z_dis = 10
    mic_r = 3
    freq_size = 1024
    window = np.hamming(freq_size)
    
    x1 = list(range(21))
    x2 = list(range(21))
    X1, X2 = np.meshgrid(x1, x2)
    
    mf = MyFunc()
    data, w_channels, w_samplings, w_frames = mf.wave_read_func('../_exp/test/test_beam_2.wav')
    
    beam = TransferFunction2d(grid_size, cell_size, z_dis, mic_r, w_channels)
    tf = beam.steering_vector(int(freq_size/2 + 1), 1/w_samplings, 4)
    
    img_path = '../img/test/'
    mf.my_makedirs(img_path)
    frq, t, stft_data = signal.stft(data, fs=w_samplings, nperseg=freq_size)
    print(stft_data.shape)
    for i, time in enumerate(t):
        beam = tf * stft_data[:, :, i].T
        beam = beam.sum(axis=2) # mic sum
        beam = np.sqrt(beam.real ** 2 + beam.imag ** 2)
        beam = beam.sum(axis=1)
        beam = np.reshape(beam, (21, -1))
        fig, ax = plt.subplots(figsize=(7, 7))
        cs = ax.pcolormesh(X1, X2, beam)
        plt.title(str(time))
        plt.savefig(img_path + str(time) + '.png')
        # print(beam.shape)
    # print(mic_positions)
    # print(ss_positions)