import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from numpy import argmax
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


filename = '../Datasets/Data2.csv'
dataset = pd.read_csv(filename)

array = dataset.values
X = array[:,0:3]
y = array[:,3]

#MiSSing Value
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer= imputer.fit(X[:,1:])
X[:,1:] = imputer.transform(X[:,1:])
#print(X)

df = pd.DataFrame(dataset)



#Encode Categorical Data
labelencoder_x = LabelEncoder()
labels_x = labelencoder_x.fit_transform(X[:,0])
#print(labels_x)

encode_X = OneHotEncoder(sparse=False)
labels_x = labels_x.reshape(len(labels_x), 1)
feature_X = encode_X.fit_transform(labels_x)
labels_x = labels_x.reshape(len(labels_x), 1)
X = X[:,1:3]
X = np.append(feature_X, X, axis=1)
#print(X)

labelencoder_y = LabelEncoder()
y =labelencoder_y.fit_transform(y)
#print(y)


# Encoding the Dependent Variable
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)
# print(X_train)
# print(X_validation)
# print(Y_train)
# print(Y_validation)

sc_X = StandardScaler()
X_train[:, 3:] = sc_X.fit_transform(X_train[:, 3:])
X_validation[:, 3:] = sc_X.transform(X_validation[:, 3:])
# print(X_train)
# print(X_validation)

