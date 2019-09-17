import numpy as np
import math


class TransferFunction2d:
    def __init__(self):
        pass
    
    @staticmethod
    def mic_positions(mic_r, mic_num):
        # return mic positions of mic array
        mic_pos_list = []
        for mic_id in range(mic_num):
            theta = mic_id / mic_num * 360
            mic_pos_list.append(Position(mic_r, theta))
        print('#Create circular microphone array position')
        return mic_pos_list

    @staticmethod
    def ss_positions(radius, min_theta, max_theta, theta_interval):
        # return sound source theta list , and sound source position class list
        theta_list = np.arange(min_theta, max_theta + theta_interval, theta_interval)
        ss_pos_list = []
        for theta in theta_list:
            ss_pos_list.append(Position(radius, theta))
        print('#Create temporal sound source position list')
        # print("theta 0 's direction is", ss_pos_list[0].pos())
        return theta_list, ss_pos_list


class Position(object):
    def __init__(self, r, theta):
        # r[m], theta[deg]
        theta = theta * math.pi / 180
        self.x = r * math.cos(theta)
        self.y = r * math.sin(theta)

    def pos(self):
        return self.x, self.y
