import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2:3].values

from sklearn.preprocessing import StandardScaler
ss1 = StandardScaler()
ss2 = StandardScaler()
x = ss1.fit_transform(x)
y = ss2.fit_transform(y)

from sklearn.svm import SVR
regressor = SVR()
regressor.fit(x,y)

ss2.inverse_transform(regressor.predict(ss1.transform(np.array([[6.5]]))))

x_grid = np.arange(min(x),max(x),0.1)
x_grid = x_grid.reshape(len(x_grid),1)
plt.scatter(x,y,color='blue')
plt.plot(x_grid,regressor.predict(x_grid),color='red')
plt.title('Support Vector Regression')
plt.show()