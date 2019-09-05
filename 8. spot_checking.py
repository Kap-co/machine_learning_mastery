# Classification algorithms:

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
from sklearn.linear_model import LogisticRegression

from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# Linear algorithms
# Logistic regression:
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
num_folds = 10
kfold = KFold(n_splits=10, random_state=7)
model = LogisticRegression()
results = cross_val_score(model, X, Y, cv=kfold)
print("Logistic regression: {}".format(results.mean()))

# Linear Discriminant Analysis:
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
num_folds = 10
kfold = KFold(n_splits=10, random_state=7)
model = LinearDiscriminantAnalysis()
results = cross_val_score(model, X, Y, cv=kfold)
print("Linear Discriminant Analysis: {}".format(results.mean()))

# Nonlinear algorithms
# K-nearest neighbors:
from sklearn.neighbors import KNeighborsClassifier
num_folds = 10
kfold = KFold(n_splits=10, random_state=7)
model = KNeighborsClassifier()
results = cross_val_score(model, X, Y, cv=kfold)
print("K-nearest: {}".format(results.mean()))

# Naive Bayes:
from sklearn.naive_bayes import GaussianNB
kfold = KFold(n_splits=10, random_state=7)
model = GaussianNB()
results = cross_val_score(model, X, Y, cv=kfold)
print("Naive Bayes: {}".format(results.mean()))

# Classification and regression trees:
from sklearn.tree import DecisionTreeClassifier
kfold = KFold(n_splits=10, random_state=7)
model = DecisionTreeClassifier()
results = cross_val_score(model, X, Y, cv=kfold)
print("Classification and regression trees: {}".format(results.mean()))

# Support vector machines:
from sklearn.svm import SVC
kfold = KFold(n_splits=10, random_state=7)
model = SVC()
results = cross_val_score(model, X, Y, cv=kfold)
print("SVM: {}".format(results.mean()))


# Regression algorithm:
raw_data = os.path.join(script_dir,"csv","housing.csv")
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data = pd.read_csv(raw_data, names=names,  comment='#')
value = data.values
X = value[:,0:-1]
Y = value[:,-1]

# Linear algorithms
# Linear Regression:
from sklearn.linear_model import LinearRegression
kfold = KFold(n_splits=10, random_state=7)
model = LinearRegression()
scoring = 'neg_mean_squared_error'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("Linear regression: {}".format(results.mean()))

# Ridge Regression - The modificaiton of linear regression where the loss function is modeled as sum squared value of the coefficient values (aka L2-norm)
from sklearn.linear_model import Ridge
num_folds = 10
kfold = KFold(n_splits=10, random_state=7)
model = Ridge()
scoring = 'neg_mean_squared_error'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("Ridge regression: {}".format(results.mean()))

# LASSO Regression - Linear regression that uses L1-norm
from sklearn.linear_model import Lasso
kfold = KFold(n_splits=10, random_state=7)
model = Lasso()
scoring = 'neg_mean_squared_error'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("LASSO regression: {}".format(results.mean()))

# ElasticNet Regression - The combination of Ridge and LASSO regression
from sklearn.linear_model import ElasticNet
kfold = KFold(n_splits=10, random_state=7)
model = ElasticNet()
scoring = 'neg_mean_squared_error'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("ElasticNet regression: {}".format(results.mean()))

# Nonlinear algorithms
# K-nearest neighbors:
from sklearn.neighbors import KNeighborsRegressor
kfold = KFold(n_splits=10, random_state=7)
model = KNeighborsRegressor()
scoring = 'neg_mean_squared_error'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("K-nearest neighbours: {}".format(results.mean()))

# Classification and regression trees:
from sklearn.tree import DecisionTreeRegressor
kfold = KFold(n_splits=10, random_state=7)
model = DecisionTreeRegressor()
scoring = 'neg_mean_squared_error'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("Classificaiton and regression trees: {}".format(results.mean()))

# Support vector machines:
from sklearn.svm import SVR
kfold = KFold(n_splits=10, random_state=7)
model = SVR()
scoring = "neg_mean_squared_error"
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("SVR: {}".format(results.mean()))

