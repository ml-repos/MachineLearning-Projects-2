import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

# Catagorical Data
labelx = LabelEncoder()
x[:,3] = labelx.fit_transform(x[:,3])
oneencode = OneHotEncoder(categorical_features=[3])
x=oneencode.fit_transform(x).toarray()
x=x[:,1:]


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)


# BAckward Elimimnation
x = np.append(np.ones((50,1)).astype(int), x, axis=1) 
  #add All predictors
x_optimize = x[:,[0,1,2,3,4,5]]
  #than fit them to Ordinary least square mathod
regressor_OLS = sm.OLS(y,x_optimize).fit()
regressor_OLS.summary()
  # if p value is grater than significance level (like 0.5) so delete it. in loop till it p is not up to 0.5

x_optimize = x[:,[0,3,4,5]]  
regressor_OLS = sm.OLS(y,x_optimize).fit()
regressor_OLS.summary()

x_optimize = x[:,[0,3,4]]  
regressor_OLS = sm.OLS(y,x_optimize).fit()
regressor_OLS.summary()

x_optimize = x[:,[0,3]]  
regressor_OLS = sm.OLS(y,x_optimize).fit()
regressor_OLS.summary()