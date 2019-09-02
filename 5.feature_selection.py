# By having irrelevant features in the data, it can decrease the accuracy of many models, especially the linear algorithms:
"""
The three benefit of feature selection are:
    1. Reduces overfitting.
    2. Improves accuracy.
    3. Reduces training time.
"""
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

from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# Univariate selection - frature extraction with univariate statistical tests (chi-squared for classification)
print("Univariate selection:")
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# feature extraction
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X,Y)
# summarize scores: The higher the result, the more significant relationship with the ourput variable.
np.set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)
# summerize selected features
print(features[0:5,:])


# Recursive feature elimination - RFE with the logistic regression
# it workds by recursively removing attributes and building a model on those attributes that remain.
print("Recursive feature elimination:")
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
rfe = RFE(model, 3) # Choose three feature from dataset
fit = rfe.fit(X, Y)
print("Num features: {}".format(fit.n_features_))
print("Selected features: {}".format(fit.support_))
print("Feature ranking: {}".format(fit.ranking_))

# Principla compunenet analysis
# PAC uses linear algebra to transform the dataset into compressed form, aka a data reduction technique. A property of PCA is that you can choose the number of dimensions or principal components in the transformed result.
print("PCA:")
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
fit = pca.fit(X)
# summarize components
print("Explained variance: {}".format(fit.explained_variance_ratio_))
print(fit.components_)

# Feature importance with extra trees classifier
from sklearn.ensemble import ExtraTreesClassifier
print("feature importance")
model = ExtraTreesClassifier()
model.fit(X,Y)
print(model.feature_importances_)


