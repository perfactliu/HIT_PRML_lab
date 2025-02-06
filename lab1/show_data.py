import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np


data_file = os.path.join('logistic_data1.csv')
data = pd.read_csv(data_file)

data_0 = data[data['y']==0]
data_1 = data[data['y']==1]


x = np.linspace(0, 100, 100)
hx = -2.5 * x + 250
# LR desicion surface example 

plt.scatter(data_0['x0'], data_0['x1'], c='red')
plt.scatter(data_1['x0'], data_1['x1'], c='blue')
plt.plot(hx,x)
plt.legend(['class 0', 'class 1'])

plt.show()
