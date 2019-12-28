# coding:utf-8
import numpy as np
import os
import sys
import configparser
import math
import _function as func


class SoundDirections:
    def __init__(self):
        sd_config = './config_sd.ini'
        if os.path.exists(sd_config):
            config = configparser.ConfigParser()
            config.read(sd_config)
            self.radius = int(config['Sound']['Radius'])
            self.min_theta = int(config['Sound']['Mini_Theta'])
            self.max_theta = int(config['Sound']['Max_Theta'])
            self.theta_interval = int(config['Sound']['Theta_Interval'])
            y_grid = int(config['Sound']['Y_Grid'])
            x_grid = int(config['Sound']['X_Grid'])
            self.grid_size = [y_grid, x_grid]
            y_cell = int(config['Sound']['Y_Cell'])
            x_cell = int(config['Sound']['X_Cell'])
            self.cell_size = [y_cell, x_cell]
            self.z_distance = int(config['Sound']['Z_Distance'])
        else:
            print("#couldn't find", sd_config)
            sys.exit()
            
    def __call__(self, bm_azimuth=False, bm_image=False):
        if bm_azimuth:
            return self.directions_azimuth()
        elif bm_image:
            return self.directions_2d()
        else:
            sys.exit()
    
    def directions_azimuth(self):
        # return sound source theta list , and sound source position class list
        theta_list = np.arange(self.min_theta, self.max_theta + self.theta_interval, self.theta_interval)
        ss_pos_list = []
        for theta in theta_list:
            ss_pos_list.append(func.Position(self.radius, theta))
        # print('#Create temporal sound source position list')
        # print("theta 0 's direction is", ss_pos_list[0].pos())
        
        # return theta_list, ss_pos_list
        return np.array(ss_pos_list)
        
    def directions_2d(self):
        # 単位 cm
        # ss_pos_list = []
        ss_position = \
            np.ones((3, int(self.grid_size[0] / self.cell_size[0] + 1),
                     int(self.grid_size[1] / self.cell_size[1] + 1))) * self.z_distance
        x = \
            np.tile(np.arange(-self.grid_size[0] / 2, self.grid_size[0] / 2 + 1, self.cell_size[0]),
                    (int(self.grid_size[1] / self.cell_size[1] + 1), 1))
        y = \
            np.tile(np.arange(self.grid_size[1] / 2, -self.grid_size[1] / 2 - 1, -self.cell_size[1]),
                    (int(self.grid_size[0] / self.cell_size[0] + 1), 1)).T
        ss_position[0, :, :] = x
        ss_position[1, :, :] = y
        ss_position = np.reshape(ss_position, (3, -1))
        # print(ss_position)
        # print('#Create temporal sound source position list')
        return ss_position
    
    
if __name__ == '__main__':
    sd = SoundDirections()
    azimuth = sd(bm_azimuth=True)
    image = sd(bm_image=True)
