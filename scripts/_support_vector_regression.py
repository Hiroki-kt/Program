# coding:utf-8
from matplotlib import pyplot as plt
import numpy as np
from sklearn.svm import SVR
import joblib
from sklearn.model_selection import GridSearchCV
from parametric_eigenspace import PrametricEigenspace
from sklearn.metrics import mean_squared_error
# from sklearn.linear_model import LinearRegression
import random


class ExecuteSVR(PrametricEigenspace):
    def __init__(self, data_set_file_path, use_mic_id=0, use_test_num=3, use_model_file=None,
                 output_file_name='test.pkl', config_name='../config_191015_PTs01_freq_all.ini'):
        super().__init__(config_path=config_name)
        self.data_set = np.load(data_set_file_path)
        print("### Data Name: ", self.data_name)
        print("### Data Set Shape: ", self.data_set.shape)
        if use_mic_id >= 9 and use_mic_id <= 12:
            self.x, self.x_test, self.y, self.y_test = self.split_train_test_only_one_data(use_mic_id-9, use_test_num)
        elif use_mic_id == -1:
            x, x_test, self.y, self.y_test = self.split_train_test_only_one_data(0, use_test_num)
            for i in range(2):
                x_i, x_test_i, y_i, y_test_i = self.split_train_test_only_one_data(i, use_test_num)
                x = np.c_[x, x_i]
                x_test = np.c_[x_test, x_test_i]
            print(x.shape, x_test.shape)
            self.x = x
            self.x_test = x_test
        else:
            self.x, self.x_test, self.y, self.y_test = self.split_train_test(use_mic_id, use_test_num)
        if use_model_file is None:
            model = self.svr(output_file_name)
        else:
            model = joblib.load(use_model_file)
        self.model_check(model)
        # self.pca_check()
    
    def split_train_test(self, mic, test_num):
        x = np.empty((0, self.data_set.shape[3]), dtype=np.float)
        x_test = np.empty_like(x)
        for i in range(int(self.data_set.shape[2] - test_num)):
            x = np.append(x, self.data_set[:, mic, i, :], axis=0)
        for i in range(test_num):
            x_test = np.append(x_test, self.data_set[:, mic, int(-1 * test_num), :], axis=0)
        y = np.array(self.DIRECTIONS.tolist() * int(self.data_set.shape[2] - test_num))
        y_test = np.array(self.DIRECTIONS.tolist() * test_num)
        print('### Test & Traing data shape: ', x.shape, y.shape, x_test.shape, y_test.shape)
        return x, x_test, y, y_test
    
    def split_train_test_only_one_data(self, freq, test_num):
        x = np.empty((0, self.data_set.shape[3]), dtype=np.float)
        x_test = np.empty_like(x)
        for i in range(int(self.data_set.shape[1] - test_num)):
            x = np.append(x, self.data_set[:, i, freq, :], axis=0)
        for i in range(test_num):
            x_test = np.append(x_test, self.data_set[:, int(-1 * test_num), freq, :], axis=0)
        y = np.array(self.DIRECTIONS.tolist() * int(self.data_set.shape[1] - test_num))
        y_test = np.array(self.DIRECTIONS.tolist() * test_num)
        # print()
        # print(x.shape, y.shape, x_test.shape, y_test.shape)
        return x, x_test, y, y_test
    
    def gen_cv(self):
        m_train = np.floor(len(self.y) * 0.75).astype(int)  # このキャストをintにしないと後にハマる
        train_indices = np.arange(m_train)
        test_indices = np.arange(m_train, len(self.y))
        yield (train_indices, test_indices)
    
    def svr(self, file_name):
        print()
        print("*** Now fitting ...  ***")
        print()
        params_cnt = 20
        params = {"C": np.logspace(0, 2, params_cnt), "epsilon": np.logspace(-1, 1, params_cnt)}
        gridsearch = GridSearchCV(SVR(), params, cv=self.gen_cv(), scoring="r2", return_train_score=True)
        gridsearch.fit(self.x, self.y)
        print("C, εのチューニング")
        print("最適なパラメーター =", gridsearch.best_params_)
        print("精度 =", gridsearch.best_score_)
        print()
        svr_model = SVR(C=gridsearch.best_params_["C"], epsilon=gridsearch.best_params_["epsilon"])
        train_indices = next(self.gen_cv())[0]
        valid_indices = next(self.gen_cv())[1]
        svr_model.fit(self.x[train_indices, :], self.y[train_indices])
        path = self.make_dir_path(array=True)
        joblib.dump(svr_model, path + file_name)
        # テストデータの精度を計算
        # print("テストデータにフィット")
        # print("テストデータの精度 =", model.score(self.x_test, self.y_test))
        # print()
        # print("※参考")
        # print("訓練データの精度 =", model.score(self.x[train_indices, :], self.y[train_indices]))
        # print("交差検証データの精度 =", model.score(self.x[valid_indices, :], self.y[valid_indices]))

        # print()
        # print("結果")
        # print(model.predict(self.x[0:90]))
        return svr_model
    
    def model_check(self, model, num=90):
        plt.figure()
        plt.plot(self.DIRECTIONS, model.predict(self.x_test[:num]), '.', label="Estimated (SVR)")
        plt.plot(self.DIRECTIONS, self.y_test[:num], label="True")
        plt.xlabel('True azimuth angle [deg]')
        plt.ylabel('Estimate azimuth angle [deg]')
        plt.legend()
        # plt.show()
        img_path = self.make_dir_path(img=True)
        plt.savefig(img_path + 'svr_' + self.data_name + '.png')

        # plt.figure()
        # plt.plot(self.DIRECTIONS, abs(model.predict(self.x_test[0:num]) - self.y_test[0:num]), 'o')
        # plt.xlabel('True azimuth angle [deg]')
        # plt.ylabel('Absolute value of error [deg]')
        # plt.ylim(0, 40)
        # # plt.legend()
        # # plt.show()
        # img_path = self.make_dir_path(img=True)
        # plt.savefig(img_path + 'svr_error_' + self.data_name + '.png')
        
        # test_num = random.randint(-45, 45)
        # print()
        # print("3つのデータの平均を出力")
        # print(test_num)
        # print(np.average(model.predict(self.x[test_num + 49:test_num + 52])))
        
        print('### RMSE', np.sqrt(mean_squared_error(self.y_test[0:num], model.predict(self.x_test[0:num]))))
        print("### R2 score", model.score(self.x_test, self.y_test))
    
    def pca_check(self):
        self.pca(self.x, self.y)

    def estimate_azimuth(self, model, test_num=random.randint(-45, 45)):
        train_indices = next(self.gen_cv())[0]
        model.fit(self.x[train_indices, :], self.y[train_indices])
        print()
        print("3つのデータの平均を出力")
        print(test_num)
        print(np.average(model.predict(self.x[test_num + 49:test_num + 52])))


if __name__ == '__main__':
    data_set_file_path = '../../_array/200123/'
    config_path = '../config_'
    model_file = '../../_array/200125/svr_200121_PTs06_cardboard_200_400.pkl'
    
    data_name = '200121_PTs06_glass_200_400'

    data_set_file = data_set_file_path + data_name + '.npy'
    output_file = 'svr_' + data_name + '.pkl'
    config_file = config_path + data_name + '.ini'
    
    # if use mic id is '9-11' make test data for torn using mic data, use test num 800 = '9', 1000 = '10', 2000 = '11'
    # if use mic id is '-1' make test data for torn useing three data
    # else select 0 ~ 8, you can make data set using id's mic
    # if use beamforming data use_mic_id is direction of data
    es = ExecuteSVR(data_set_file,
                    use_mic_id=0,
                    use_test_num=2,
                    use_model_file=model_file,
                    output_file_name=output_file,
                    config_name=config_file)
