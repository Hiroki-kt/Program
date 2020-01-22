import numpy as np
from matplotlib import pyplot as plt

data_set_file_path = "../../_array/191219/191015_PTs01_beam.npy"
data_set = np.load(data_set_file_path)

print("Data set shape:", data_set.shape)

plt.figure()
plt.plot(data_set[50, 45, 0, :])
plt.show()