# coding:utf-8
from matplotlib import pyplot as plt
import numpy as np
from sklearn.svm import SVR
import joblib
from sklearn.model_selection import GridSearchCV
from parametric_eigenspace import PrametricEigenspace
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import random


class ExecuteSVR(PrametricEigenspace):
    def __init__(self, data_set_file, use_mic_id=0, use_test_num=3, use_model_file=None):
        super().__init__()
        self.data_set = np.load(data_set_file)
        print()
        print("### Data Set Shape: ", self.data_set.shape)
        self.x, self.x_test, self.y, self.y_test = self.split_train_test(use_mic_id, use_test_num)
        if use_model_file is None:
            model = self.svr()
        else:
            model = joblib.load(use_model_file)
        self.model_check(model)
        self.pca_check()
        
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
        
    def svr(self, file_name='svr_model.pkl'):
        print("Now fitting ... ")
        params_cnt = 20
        params = {"C": np.logspace(0, 2, params_cnt), "epsilon": np.logspace(-1, 1, params_cnt)}
        gridsearch = GridSearchCV(SVR(), params, cv=self.gen_cv(), scoring="r2", return_train_score=True)
        gridsearch.fit(self.x, self.y)
        print("C, εのチューニング")
        print("最適なパラメーター =", gridsearch.best_params_)
        print("精度 =", gridsearch.best_score_)
        print()
        svr_model = SVR(C=gridsearch.best_params_["C"], epsilon=gridsearch.best_params_["epsilon"])
        path = self.make_dir_path(array=True)
        joblib.dump(svr_model, path + file_name)
        return svr_model
        
    def model_check(self, model):
        train_indices = next(self.gen_cv())[0]
        valid_indices = next(self.gen_cv())[1]
        model.fit(self.x[train_indices, :], self.y[train_indices])
        # テストデータの精度を計算
        print("テストデータにフィット")
        print("テストデータの精度 =", model.score(self.x_test, self.y_test))
        print()
        print("※参考")
        print("訓練データの精度 =", model.score(self.x[train_indices, :], self.y[train_indices]))
        print("交差検証データの精度 =", model.score(self.x[valid_indices, :], self.y[valid_indices]))
        
        print()
        print("結果")
        print(model.predict(self.x[100:200]))
        
        plt.figure()
        plt.plot(self.DIRECTIONS, model.predict(self.x[100:200]), '.', label="estimated (SVR)")
        plt.plot(self.DIRECTIONS, self.y[100:200], label="True")
        plt.legend()
        plt.show()
        
        test_num = random.randint(-45, 45)
        print()
        print("3つのデータの平均を出力")
        print(test_num)
        print(np.average(model.predict(self.x[test_num+49:test_num+52])))
        
        print()
        print('RMSE')
        print(np.sqrt(mean_squared_error(self.y[10:90], model.predict(self.x[10:90]))))
        
    def pca_check(self):
        self.pca(self.x, self.y)

        
if __name__ == '__main__':
    data_set_file = '../_array/191217/1205_glass_plate_0cross.npy'
    model_file = '../_array/191217/1025_svr_model.pkl'
    es = ExecuteSVR(data_set_file, use_mic_id=0, use_test_num=2, use_model_file=model_file)
