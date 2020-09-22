import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2:3].values

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(x,y)

regressor1 = 

x_grid = np.arange(min(x),max(x),0.1)
x_grid = x_grid.reshape(len(x_grid),1)
plt.scatter(x,y,color='red')
plt.plot(x_grid,regressor.predict(x_grid),color='blue')
plt.show()