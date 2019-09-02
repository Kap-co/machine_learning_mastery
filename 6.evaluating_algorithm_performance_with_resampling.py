# With consideration of over and underfitting, the evaluation is a nestimate that we can use to talk about how well we think that algorihm may actually do in practice.
"""
The four major techniques to split trainnig dataset, and create useful estimates of performace:
    1. Train and Test Sets.
    2. k-fold Cross Validation.
    3. Leave One Out Cross validation.
    4. Repeated Random Test-Train Splits
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

# Split into Train and Test Sets:
# use of different training and testing datasets, by spliting one datset to several, some for training some for predict and testing.
print("split into train and test sets:")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

test_size = 0.33
seed = 5 # By specifying the random seed we ensure that we get the same random numbers each time we run the code and in turn the same split of data.
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=test_size, random_state=7)
model = LogisticRegression()
model.fit(X_train,Y_train)
result = model.score(X_test, Y_test)
print("Accuracy: {:.3f}".format(result*100))


# K-fold cross validation
# The data set is equally divided into k groups, for each group takes one for output for test data others as training data, fit the model individually to retian the evaluation score.
# When all folds are evaluated, the mean value is calculated as the final result.
print("k-fold validation:")
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
num_folds = 10
seed = 7
kfold = KFold(n_splits=num_folds, random_state=seed)
model = LogisticRegression()
results = cross_val_score(model, X, Y, cv=kfold)
print("Accuracy: {:.3f} ({:.3f})".format(results.mean()*100, results.std()*100))

# Leave one out cross validation
# K-fold where k == 1, this will relativily have lower baised and siutable for small dataset.
print("leave one out validation:")
from sklearn.model_selection import LeaveOneOut
#from  sklearn.model_selection import cross_val_score
loocv = LeaveOneOut()
model = LogisticRegression()
results = cross_val_score(model, X, Y, cv=loocv)
print("Accuracy: {:.3f} ({:.3f})".format(results.mean()*100, results.std()*100))


# Repeated random test-train splits
# The repeated randomized version of test-train splits, imporves accuracy but may increase bias.
print("repeated random test-train:")
from sklearn.model_selection import ShuffleSplit
#from sklearn.model_selection import cross_val_score
n_splits = 10
test_size = 0.33
seed = 7
kfold = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
model = LogisticRegression()
result = cross_val_score(model,X,Y,cv=kfold)
print("Accuracy: {:.3f} ({:.3f})".format(results.mean()*100, results.std()*100))
