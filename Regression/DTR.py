import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

filename = '../Datasets/Data5.csv'
dataset = pd.read_csv(filename)
print(dataset)

array = dataset.values
X = array[:,1:2]
y = array[:,2:]
print(X.shape, y.shape)

# Training the Decision Tree Regression model on the whole dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

# Predicting a new result
P = regressor.predict([[6.5]])
print(P)

# Visualising the Decision Tree Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()