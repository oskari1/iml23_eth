import numpy as np
from matplotlib import pyplot as plt

arr = np.array((4,1,7,1,8,2,9,3,1,9))
mean = np.mean(arr)
flipped_arr = -(arr - mean) + mean
plt.plot(range(len(arr)), arr)
plt.plot(range(len(arr)), flipped_arr)
plt.show()