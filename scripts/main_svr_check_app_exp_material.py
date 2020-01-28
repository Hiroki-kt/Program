from _support_vector_regression import ExecuteSVR
from parametric_eigenspace import PrametricEigenspace
import numpy as np

# if use mic id is '9-11' make test data for torn using mic data, use test num 800 = '9', 1000 = '10', 2000 = '11'
# if use mic id is '-1' make test data for torn useing three data
# else select 0 ~ 8, you can make data set using id's mic

'''
How to use ExecuteSVR

input parametor

data_set_file                           : string (file path)
use_mic_id(default = 0)                 : int (mic id 0 ~ 3, or 0 ~  7)
use_test_num(default = 3)               : int (test data num)
use_model_file(default = None)          : string (file path) when you use svr model file
out_put_file_name(default = test.pkl)   : string (file path) when you make new svr model, the out put file name
'''

data_set_file_path = '../../_array/200123/'
config_path = '../config_'
model_file = '../../_array/191217/1025_svr_model.pkl'


data_name = '200121_PTs06_'

material_list = ['glass', 'cardboard']
size_list = ['50_100', '100_200', '200_400']

mic = 0

print()
print("*******************************")
for i in size_list:
    config_file_glass = config_path + data_name + material_list[0] + '_' + str(i) + '.ini'
    config_file_card = config_path + data_name + material_list[1] + '_' + str(i) + '.ini'
    data_set_file_glass = np.load(data_set_file_path + data_name + material_list[0] + '_' + str(i) + '.npy')
    data_set_file_card = np.load(data_set_file_path + data_name + material_list[1] + '_' + str(i) + '.npy')
    print(data_set_file_card.shape, data_set_file_glass.shape)
    data_set = np.concatenate((data_set_file_glass, data_set_file_card), axis=2)
    # data_set = np.reshape(data_set[:, mic, :, :], (-1, 1385))
    x = np.empty((0, data_set.shape[3]), dtype=np.float)
    for j in range(int(data_set.shape[2])):
        x = np.append(x, data_set[:, mic, j, :], axis=0)
    # output_file = 'svr_' + data_name + str(i) + '_' + str(i) + '.pkl'
    # es = ExecuteSVR(data_set_file,
    #                 use_mic_id=0,
    #                 use_test_num=2,
    #                 output_file_name=output_file,
    #                 # use_model_file=model_file,
    #                 config_name=config_file
    #                 )
    pe = PrametricEigenspace(config_file_glass)
    pe.pca(x)
    print("*******************************")
