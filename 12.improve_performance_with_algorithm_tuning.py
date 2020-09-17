# Algorithm tuning or hyperparameter optimization is a final step to finalize the model.
# Optimization suggests the search-nature of the problem, the goal is to use different search strategies to find a good and robust parameter or set of parateters for an algorithm on a given problem.

import pandas as pd
import numpy as np
import os
script_dir = os.path.dirname(__file__)
raw_data = os.path.join(script_dir,"csv","pima-indians-diabetes.csv")
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv(raw_data, names=names,  comment='#')
value = data.values
X = value[:,0:-1]
Y = value[:,-1]

# Grid search parameter tuning
# This algorithm will methodically build and evaluate a model for each combination of algorithm parameters specified in a grid.
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
alphas = np.array([1,0.1,0.01,0.001,0.0001,0])
param_grid = dict(alpha=alphas)
model = Ridge()
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid.fit(X, Y)
print(grid.best_score_)
print(grid.best_estimator_.alpha)

# Random search parameter tuning
# This will sample algorithm parameters from a random distribution for a fixed number of iterations.
# A model is constructed and evaluated for each combination of parameters chosen.
from scipy.stats import uniform
from sklearn.model_selection import RandomizedSearchCV
param_grid = {'alpha': uniform()}
model = Ridge()
rsearch = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=100, random_state=7)
rsearch.fit(X, Y)
print(rsearch.best_score_)
print(rsearch.best_estimator_.alpha)