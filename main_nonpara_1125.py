from parametric_eigenspace import PrametricEigenspace
import numpy as np
import sys
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.externals import joblib


class NonParametric(PrametricEigenspace):
    def __init__(self, use_data_npy=None, use_model_file=None, config='config_nonpara.ini',
                 output_pkl_file_name='svr_gridsearch.pkl', recode_inddividual=None, use_mic=1, test_data_num=2):
        super().__init__(config_path=config)
        if use_data_npy is None:
            if recode_inddividual is not None:
                self.data = self.make_data_set_recode_individually(recode_inddividual[0], recode_inddividual[1])
            else:
                self.data = self.make_data_set(individual=True)
                np.save('normalize.npy', self.data)
        else:
            self.data = np.load(use_data_npy)
            
        '''Make data set'''
        self.x, self.y, self.x_test, self.y_test = self.make_data(use_mic, test_data_num)
        '''Make model'''
        # self.svm_model, self.svr_model = self.svm_svr()
        if use_model_file is None:
            grid = self.gridresearch(output_pkl_file_name)
        else:
            grid = joblib.load(use_model_file)
        print(grid)
        self.svr_model_ver2 = self.svr_ver2(grid)
        
    def make_data(self, mic, test_num):
        # 3つで区切る
        # 訓練データ、テストデータに分割
        x = np.empty((0, self.data.shape[3]))
        y = []
        x_test = np.empty_like(x)
        y_test = []
        for i in range(self.data.shape[2]-test_num):
            x = np.append(x, self.data[:, mic-1, i, :], axis=0)
            y += self.DIRECTIONS.tolist()
        for i in range(test_num):
            x_test = np.append(x_test, self.data[:, mic-1, i, :], axis=0)
            y_test += self.DIRECTIONS.tolist()
        print("Make test data", x.shape, len(y), x_test.shape, len(y_test))
        return x, y, x_test, y_test
        
    def gen_cv(self):
        # 6:2:2に分割にするため、訓練データのうちの後ろ1/4を交差検証データとする
        # 交差検証データのジェネレーター
        m_train = np.floor(len(self.y) * 0.75).astype(int)  # このキャストをintにしないと後にハマる
        train_indices = np.arange(m_train)
        test_indices = np.arange(m_train, len(self.y))
        yield (train_indices, test_indices)
    
    def gridresearch(self, output_name):
        # ハイパーパラメータのチューニング
        params_cnt = 20
        params = {"C": np.logspace(0, 2, params_cnt), "epsilon": np.logspace(-1, 1, params_cnt)}
        gridsearch = GridSearchCV(SVR(), params, cv=self.gen_cv(), scoring="r2", return_train_score=True)
        print("Now fitting ...")
        gridsearch.fit(self.x, self.y)
        print("C, εのチューニング")
        print("最適なパラメーター =", gridsearch.best_params_)
        print("精度 =", gridsearch.best_score_)
        output_path = '../array' + self.make_dir_path()
        self.my_makedirs(output_path)
        joblib.dump(gridsearch.best_estimator_, output_path + output_name)
        return gridsearch
        
    def svm_svr(self, svm_step=20):
        step_all = svm_step
        post = 0
        label_svm = self.label(self.data.shape[0], step=svm_step)
        svm_model = self.support_vector_machine(self.data, label_svm)
        svr_model_list = []
        for i in range(int(self.data.shape[0]/svm_step)):
            model = self.support_vector_regression(self.data[post:svm_step, :, :, :], self.DIRECTIONS[post:svm_step])
            # print(pe.DIRECTIONS[post:step])
            post = svm_step
            svm_step += step_all
            svr_model_list.append(model)
        return svm_model, svr_model_list

    def svr_ver2(self, gridsearch):
        if isinstance(gridsearch, SVR):
            print("OK")
            regr = gridsearch
        elif isinstance(gridsearch, GridSearchCV):
            print("OK!!!")
            regr = SVR(C=gridsearch.best_params_["C"], epsilon=gridsearch.best_params_["epsilon"])
        else:
            print("Error")
            sys.exit()
        train_indices = next(self.gen_cv())[0]
        # print(train_indices)
        valid_indices = next(self.gen_cv())[1]
        regr.fit(self.x[train_indices, :], np.array(self.y)[train_indices])
        # テストデータの精度を計算
        print("テストデータにフィット")
        print("テストデータの精度 =", regr.score(self.x_test, self.y_test))
        print()
        print("※参考")
        print("訓練データの精度 =", regr.score(self.x[train_indices, :], np.array(self.y)[train_indices]))
        print("交差検証データの精度 =", regr.score(self.x[valid_indices, :], np.array(self.y)[valid_indices]))
        print()
        return regr
    
    def estimate(self, test_data):
        pred_svm = self.svm_model.predict(test_data)
        # print(pred_svm)
        if len(pred_svm) == 1:
            pred_svr = self.svr_model[int(pred_svm[0])].predict(test_data)
            # print(pred_svr)
            return pred_svr
        else:
            print('ERROR')
            sys.exit()
    

if __name__ == '__main__':
    npm = NonParametric(use_data_npy='normalize.npy', use_model_file='test.pkl')
    estimate_list = []
    estimate_list_2 = []
    for test_deg in range(100):
        estimate = npm.estimate(npm.data[test_deg:test_deg+1, 0, 6, :])
        estimate_ver2 = npm.svr_model_ver2.predict(npm.x_test[test_deg:test_deg+1, :])
        # print(test_deg - 50)
        estimate_list.append(estimate[0])
        estimate_list_2.append((estimate_ver2[0]))
        
    plt.figure()
    plt.plot(npm.DIRECTIONS, estimate_list, '.', label='SVM+SVR estimate')
    # plt.plot(npm.DIRECTIONS, estimate_list_2, '.', label='SVR(Using Grid Search) estimate')
    plt.plot(npm.DIRECTIONS, npm.DIRECTIONS, label='True')
    plt.ylim(-50, 50)
    plt.xlabel("Estimated Azimuth [deg]")
    plt.ylabel("True Azimuth [deg]")
    plt.legend()
    plt.show()
