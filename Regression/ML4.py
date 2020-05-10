import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression # inbuilt functions for logistic regression
from sklearn.metrics import r2_score

filename = '../Datasets/Data4.csv'
dataset = pd.read_csv(filename)

array = dataset.values
X = array[:,0:4]
y = array[:,4:]
print(y.shape)
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
#print(X)

X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

y_pred = regressor.predict(X_validation)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), Y_validation.reshape(len(Y_validation),1)),1))
print(Y_validation,y_pred)
print(r2_score(Y_validation, y_pred , multioutput='variance_weighted'))
