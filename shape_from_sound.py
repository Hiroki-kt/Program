# coding:utf-8
import numpy as np
import configparser
import os
import scipy
# from matplotlib import pyplot as plt



class ShapeFromSound:
    def __init__(self, config_path):
        super().__init__()
        if os.path.exists(config_path):
            config = configparser.ConfigParser()
            config.read(config_path)
            # Environment
            object_pos = np.zeros((3,), dtype=float)
            ss_list = []
            object_pos[1] = float(config['Environment']['object_distance'])
            self.object_pos = object_pos
            spearker_num = int(config['Environment']['speaker_num'])
            for i in range(spearker_num):
                id = 'speaker' + str(i + 1) + '_distance'
                ss_list.append(float(config['Environment'][id]))
            self.ss_list = ss_list
            self.true_direction = int(config['Environment']['object_direction'])
            print("make envs data")
            print("make sound source vector")
            ss_pos = np.zeros((len(self.ss_list), 3), np.float)
            for i, name in enumerate(self.ss_list):
                ss_pos[i, 0] = float(name)
            self.ss_pos = ss_pos
            s = self.object_pos + ss_pos
            s[:, 0] = s[:, 1] / s[:, 0]
            s[:, 1] = 1
            s[:, 2] = 0
            self.s = s
            print("make s data set", s)

    def shapeFromSound(self, r):

        if self.ss_pos.shape[0] == 3:
            # 音源数が三の場合は等式で連立方程式で解ける
            surface_normal = np.dot(np.linalg.inv(self.s[:2, :2]), r[:2]) #検証の際には次元が二次元のみであったため、次元を落としている。
            return surface_normal

        elif self.ss_pos.shape[0] < 3:
            print("ERROR: sound source number is less")

        else:
            # 音源数が3以上の場合は解析的に解く
            # Xtil = np.c_[np.ones(s.shape[0]), s[:, :2]]
            Xtil = self.s[:, :2]
            A = np.dot(Xtil.T, Xtil)
            b = np.dot(Xtil.T, r)
            print('Xtil:', Xtil, 'A:', A, 'b:', b)
            surface_normal = np.dot(scipy.linalg.pinv(A), b)
            return surface_normal
