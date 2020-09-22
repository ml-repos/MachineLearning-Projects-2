import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x,y)

from sklearn.preprocessing import PolynomialFeatures
pr = PolynomialFeatures(degree=4)
x_poly = pr.fit_transform(x)
lr1 = LinearRegression()
lr1.fit(x_poly,y)

plt.scatter(x,y,color='red')
plt.plot(x, lr.predict(x), color='blue')
plt.title('Linear Regression')
plt.show()


x_grid = np.arange(min(x),max(x),0.1)
x_grid = x_grid.reshape(len(x_grid),1)
plt.scatter(x,y,color='red')
plt.plot(x_grid, lr1.predict(pr.fit_transform(x_grid)), color='blue')
plt.title('Polynimial Regression')
plt.show()


lr.predict([[6.5]])

lr1.predict(pr.fit_transform([[6.5]]))