
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

# Bagging (Bootstrap Aggregation): takes multiple subsamples from the training dataset and training a model for each sample.
# The final output prediction is averaged acorss the predictions of all of the sub-models. It usually works well when model has high variance

# Bagged Decision Trees
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
seed = 7
kfold = KFold(n_splits=10, random_state=seed)
cart = DecisionTreeClassifier()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
results = cross_val_score(model,X,Y, cv=kfold)
print(results.mean())

# Random Forest
# An extended version of bagged dicision trees, samples of the training dataset are taken with repalcement, and rather than greedily choosing the best split point, only a random subset of features are considered for each split.
from sklearn.ensemble import RandomForestClassifier
num_trees = 100
max_features = 3
kfold = KFold(n_splits=10, random_state=7)
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

# Extra Trees
# Random trees are constructed from samples of the training dataset.
from sklearn.ensemble import ExtraTreesClassifier
num_trees = 100
max_features = 7
kfold = KFold(n_splits=10, random_state=7)
model = ExtraTreesClassifier(n_estimators=num_trees, max_features=max_features)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

# Boosting: creates a sequance of models that attempt to correct the mistakes of the model before them in the sequence
# The final result is compared by the results created in terms of accuracy or other weighting methods

# AdaBoost
# It works by weighting instances in the datset by how easy or difficult they are to classify, allowing the algorithm to pay or less attention to them in the constuction of subsequent models.
# The result is created by their demonstrated accuracy and the results are combined to created a final output prediction.
from sklearn.ensemble import AdaBoostClassifier
num_trees = 30
seed=7
kfold = KFold(n_splits=10, random_state=seed)
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

# Stochastic Gradient Boosting
# A technique that is proving to be perhaps one of the best techniques available for improving performance via ensembles.
from sklearn.ensemble import GradientBoostingClassifier
seed = 7
num_trees = 100
kfold = KFold(n_splits=10, random_state=seed)
model = GradientBoostingClassifier(n_estimators=num_trees, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

# Voting: Its simple and works by first creating two or more standalone models from your traning dataset, and select the best. The selection of sub-model and the scope are supported but difficult.
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
estimators = []
model1 = LogisticRegression()
estimators.append(('logistic', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 = SVC()
estimators.append(('svm', model3))
# create the ensemble model
ensemble = VotingClassifier(estimators)
results = cross_val_score(ensemble, X, Y, cv=kfold)
print(results.mean())
