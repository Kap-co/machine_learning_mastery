# The pipleline help to clearly define and automate stardard workflows.
# Pipelines work by allowing for a linear sequence of data transforms to be chained together culminating in a modeling process that can be evaluated.
# The pipeline is defined with two steps: 1. Stardardize the data 2. Learn a Linear Discriminant Analysis model 

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

# create pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# create pipeline
estimators = []
estimators.append(("standardize", StandardScaler()))
estimators.append(("lda", LinearDiscriminantAnalysis()))
model = Pipeline(estimators)
# evaluate pipeline
kfold = KFold(n_splits=10, random_state=7)
results = cross_val_score(model,X,Y,cv=kfold)
print(results.mean())


# Feature Extraction, an improved version of pipleline, it provides a handy tool called the FeatureUnion which allows the reuslts of multiple fuature selection and extraction procedures to be combined into a larger dataset on which model can be trained.
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import FeatureUnion
#create feature union
features = []
features.append(("pca",PCA(n_components=3)))
features.append(("select_best", SelectKBest(k=6)))
feature_union = FeatureUnion(features)
# create pipeline
estimators = []
estimators.append(("feature_union", feature_union))
estimators.append(("logistic", LogisticRegression()))
model = Pipeline(estimators)
# evaluate pipeline
kfold = KFold(n_splits=10, random_state=7)
results = cross_val_score(model,X,Y,cv=kfold)
print(results.mean())