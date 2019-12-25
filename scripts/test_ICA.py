from sklearn.decomposition import FastICA
from sklearn.datasets import load_digits
from _function import MyFunc
from _tf_generate_tsp import TSP
from matplotlib import pyplot as plt

class ICA:
    @staticmethod
    def ica(wave_path=None, wave_data=None):
        if wave_path is not None:
            CONFIG_PATH = "./config_tf.ini"
            data, channels, samlpling, frames = MyFunc().wave_read_func(wave_path)
            data = TSP(CONFIG_PATH).cut_tsp_data(use_data=data)
        elif wave_data is not None:
            data = wave_data
        else:
            print("ERROR")
            return -1
        transformer = FastICA(n_components=2, random_state=0)
        X_transformed = transformer.fit_transform(data.T)
        
        # plt.figure()
        # for i in range(4):
        #     plt.plot(X_transformed[:, i], label='%d' % (i+1))
        #     plt.specgram(X_transformed[:, i], Fs=44100)
        #     plt.show()
    
        return X_transformed


if __name__ == '__main__':
    s_file = '../../../../OneDrive/Research/Recode_Data/up_tsp_heimen/0.wav'
    a = ICA().ica(wave_path=s_file)
    print(a.shape)