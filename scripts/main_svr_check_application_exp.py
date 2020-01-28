from _support_vector_regression import ExecuteSVR

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
model_file = '../../_array/200125/svr_200121_PTs06_cardboard_200_400.pkl'
model_file_path = '../../_array/200125/'


data_name = '200121_PTs06_'

material_list = ['glass', 'cardboard']
size_list = ['50_100', '100_200', '200_400']

print()
print("*******************************")
for i in material_list:
    for j in size_list:
        config_file = config_path + data_name + str(i) + '_' + str(j) + '.ini'
        data_set_file = data_set_file_path + data_name + str(i) + '_' + str(j) + '.npy'
        model_file = model_file_path + 'svr_' + data_name + str(i) + '_' + str(j) + '.pkl'
        output_file = 'svr_' + data_name + str(i) + '_' + str(j) + '.pkl'
        es = ExecuteSVR(data_set_file,
                        use_mic_id=0,
                        use_test_num=2,
                        output_file_name=output_file,
                        # use_model_file=model_file,
                        config_name=config_file
                        )
        print("*******************************")

# config_file = config_path + data_name + 'multi_200_400' + '.ini'
# data_set_file = data_set_file_path + data_name + 'multi_200_400' + '.npy'
# output_file = 'svr_' + data_name + 'multi_200_400' + '.pkl'
# es2 = ExecuteSVR(data_set_file,
#                  use_mic_id=0,
#                  use_test_num=2,
#                  output_file_name=output_file,
#                  use_model_file=model_file,
#                  config_name=config_file
#                  )
