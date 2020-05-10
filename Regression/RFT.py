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

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 500, random_state = 0)
regressor.fit(X, y)

p = regressor.predict([[6.5]])
print(p)


# Visualising the Random Forest Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.001)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()