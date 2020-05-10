import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from numpy import argmax
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression


filename = '../Datasets/Data3.csv'
dataset = pd.read_csv(filename)

array = dataset.values
X = array[:,0:1]
y = array[:,1]

X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)
# print(X_train)
# print(X_validation)
# print(Y_train)
# print(Y_validation)



regressor = LinearRegression()
regressor.fit(X_train, Y_train)
Y_pred = regressor.predict(X_train)


plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, Y_pred, color='blue')
plt.title('Salary vs Experiece Train Set')
plt.xlabel('Years of Exp')
plt.ylabel('Salary')
plt.show()


plt.scatter(X_validation, Y_validation, color='red')
plt.plot(X_train, Y_pred, color='blue')
plt.title('Salary vs Experiece Test Set')
plt.xlabel('Years of Exp')
plt.ylabel('Salary')
plt.show()