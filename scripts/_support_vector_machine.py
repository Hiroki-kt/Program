# coding:utf-8
from matplotlib import pyplot as plt
import numpy as np
from sklearn.svm import SVR, SVC
import joblib
from sklearn.model_selection import GridSearchCV
from parametric_eigenspace import PrametricEigenspace
from sklearn.metrics import mean_squared_error
# from sklearn.linear_model import LinearRegression
import random
import sys


class ExecuteSVR(PrametricEigenspace):
    def __init__(self,
                 data_set,
                 label_list,
                 use_mic_id=0,
                 use_test_num=3,
                 use_model=None,
                 output='test',
                 config='../config_191015_PTs01_freq_all.ini'
                 ):
        
        super().__init__(config_path=config)
        self.data_set = np.load(data_set)
        self.output_name = output
        print("### Data Name: ", self.output_name)
        print("### Data Set Shape: ", self.data_set.shape)
        
        self.x, self.x_test, self.y, self.y_test = self.split_train_test(use_mic_id, use_test_num, label_list)
        
        # if use_model_file is None:
        #     model = self.svm(output_file_name + '.pkl)
        # else:
        #     model = joblib.load(use_model_file)
        # self.model_check(model)
        # self.pca_check()
    
    def split_train_test(self, mic, test_num, label_list):
        x = np.empty((0, self.data_set.shape[3]), dtype=np.float)
        x_test = np.empty_like(x)
        for i in range(int(self.data_set.shape[2] - test_num)):
            x = np.append(x, self.data_set[:, mic, i, :], axis=0)
        for i in range(test_num):
            x_test = np.append(x_test, self.data_set[:, mic, int(-1 * test_num), :], axis=0)
        labeling_directions = self.labeling(label_list)
        y = np.array(self.DIRECTIONS.tolist() * int(self.data_set.shape[2] - test_num))
        y_test = np.array(self.DIRECTIONS.tolist() * test_num)
        print('### Test & Traing data shape: ', x.shape, y.shape, x_test.shape, y_test.shape)
        return x, x_test, y, y_test
    
    def labeling(self, label_list):
        if len(self.DIRECTIONS) % len(label_list) != 0:
            print('Error con not split label, Now directions is :', self.DIRECTIONS.shape)
            sys.exit()
        else:
            label = np.zeros_like(self.DIRECTIONS)
            range_num = len(self.DIRECTIONS)/len(label_list)
            for i in label_list:
                target_id = np.abs(self.DIRECTIONS - i).argmin()
                print(target_id)
                label[target_id-range_num:target_id+range_num+1] = i
                
            return label
        
    def gen_cv(self):
        m_train = np.floor(len(self.y) * 0.75).astype(int)  # このキャストをintにしないと後にハマる
        train_indices = np.arange(m_train)
        test_indices = np.arange(m_train, len(self.y))
        yield (train_indices, test_indices)
    
    def svm(self, file_name):
        print()
        print("*** Now fitting ...  ***")
        print()
        params_cnt = 20
        params = {"C": np.logspace(0, 2, params_cnt), "epsilon": np.logspace(-1, 1, params_cnt)}
        gridsearch = GridSearchCV(SVC(), params, cv=self.gen_cv(), scoring="r2", return_train_score=True)
        gridsearch.fit(self.x, self.y)
        print("C, εのチューニング")
        print("最適なパラメーター =", gridsearch.best_params_)
        print("精度 =", gridsearch.best_score_)
        print()
        svr_model = SVR(C=gridsearch.best_params_["C"], epsilon=gridsearch.best_params_["epsilon"])
        train_indices = next(self.gen_cv())[0]
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
        plt.savefig(img_path + self.output_name + '.png')

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
    data_set_file_path = '/Users/hiroki-kt/OneDrive/Research/_array/200128/'
    config_path = '../config_'
    model_file = '../../_array/200125/svr_200121_PTs06_cardboard_200_400.pkl'
    
    data_name = '200128_PTs07_kuka_distance_250'

    data_set_file = data_set_file_path + data_name + '.npy'
    output_file = 'svr_' + data_name
    config_file = config_path + data_name + '.ini'
    
    LABEL = np.arange(-45, 46, 10)
    print(LABEL)
    
    # else select 0 ~ 8, you can make data set using id's mic
    # if use beamforming data use_mic_id is direction of data
    es = ExecuteSVR(data_set_file,
                    LABEL,
                    use_mic_id=0,
                    use_test_num=2,
                    use_model=model_file,
                    output=output_file,
                    config=config_file)
