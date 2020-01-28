from _create_data_set import CreateDataSet

# if use mic id is '9-11' make test data for torn using mic data, use test num 800 = '9', 1000 = '10', 2000 = '11'
# if use mic id is '-1' make test data for torn useing three data
# else select 0 ~ 8, you can make data set using id's mic

'''
How to use CreateDataSet

only make config path
'''

config_path = '../config_'

data_name = '200121_PTs06_'

material_list = ['glass', 'cardboard']
size_list = ['50_100', '100_200', '200_400']

print()
print("*******************************")
for i in material_list:
    for j in size_list:
        config_file = config_path + data_name + str(i) + '_' + str(j) + '.ini'
        print(config_file)
        cd = CreateDataSet(config_file)
        cd()
        print("*******************************")
        
config_file = config_path + data_name + 'multi_200_400' + '.ini'
print(config_file)
cd = CreateDataSet(config_file)
cd()
