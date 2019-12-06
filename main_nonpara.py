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
from sklearn.model_selection import GridSearchCV
from parametric_eigenspace import PrametricEigenspace


class ExecuteSVR(PrametricEigenspace):
    def __init__(self, data_set_file, use_mic_id=0, use_test_num=2):
        super().__init__()
        self.data_set = np.load(data_set_file)
        print()
        print("### Data Set Shape: ", self.data_set.shape)
        self.x, self.x_test, self.y, self.y_test = self.split_train_test(use_mic_id, use_test_num)
        
    def split_train_test(self, mic, test_num):
        x = np.empty((0, self.data_set.shape[3]), dtype=np.float)
        x_test = np.empty_like(x)
        for i in range(int(self.data_set.shape[2] - test_num)):
            x = np.append(x, self.data_set[:, mic, i, :], axis=0)
        for i in range(test_num):
            x_test = np.append(x_test, self.data_set[:, mic, int(-1 * test_num), :], axis=0)
        y = np.array(self.DIRECTIONS.tolist() * int(self.data_set.shape[2] - test_num))
        y_test = np.array(self.DIRECTIONS.tolist() * test_num)
        print()
        print(x.shape, y.shape, x_test.shape, y_test.shape)
        return x, x_test, y, y_test

    def gen_cv(self):
        m_train = np.floor(len(self.y) * 0.75).astype(int)  # このキャストをintにしないと後にハマる
        train_indices = np.arange(m_train)
        test_indices = np.arange(m_train, len(self.y))
        yield (train_indices, test_indices)
        
    def svr(self):
        print("Now fitting ... ")
        params_cnt = 20
        params = {"C": np.logspace(0, 2, params_cnt), "epsilon": np.logspace(-1, 1, params_cnt)}
        gridsearch = GridSearchCV(SVR(), params, cv=self.gen_cv(), scoring="r2", return_train_score=True)
        gridsearch.fit(self.x, self.y)
        print("C, εのチューニング")
        print("最適なパラメーター =", gridsearch.best_params_)
        print("精度 =", gridsearch.best_score_)
        print()
        regr = SVR(C=gridsearch.best_params_["C"], epsilon=gridsearch.best_params_["epsilon"])
        train_indices = next(self.gen_cv())[0]
        valid_indices = next(self.gen_cv())[1]
        regr.fit(self.x[train_indices, :], self.y[train_indices])
        # テストデータの精度を計算
        print("テストデータにフィット")
        print("テストデータの精度 =", regr.score(self.x_test, self.y_test))
        print()
        print("※参考")
        print("訓練データの精度 =", regr.score(self.x[train_indices, :], self.y[train_indices]))
        print("交差検証データの精度 =", regr.score(self.x[valid_indices, :], self.y[valid_indices]))
        
        
if __name__ == '__main__':
    data_set_file = '../_array/191206/glass_plate.npy'
    es = ExecuteSVR(data_set_file)
    es.svr()