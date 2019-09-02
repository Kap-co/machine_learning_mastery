# The choice of metrics(scoring method) to the data changes the algorithms mesure and how it compared, further influences the choice of final algorithm.
# There are different metrhic evaluation method according to its nature of the problem
"""
For classificaiton metric:
    1. Classification Accuracy
    2. Logarithmic Locss
    3. Area Under ROC Curve
    4. Confusion Matrix
    5. Classification Report
"""

# In this example will use logistic regression for classification problem of indian diabetes, the kfold (k=10) will be used as evaluation method
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

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
kfold = KFold(n_splits=10, random_state=7)
model = LogisticRegression()

# Classification accuracy - the number of correct predictions made as a ratio of all predictions made.
# The samll mean result the better prediction, 0 representing the perfect logloss
scoring = 'accuracy'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("Accuracy: {:.3f} ({:.3f})".format(results.mean(), results.std()))

# Logarithmic loss - a performnce metric for evaluating the predictions of probabilities of membership to a given class, measures the confidence of given prediction.
scoring = 'neg_log_loss'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("Logloss: {:.3f} ({:.3f})".format(results.mean(), results.std()))

# ROC curve (aka. AUC) - a performance metric for binary classification problems, this represents a model's abbility to discriminate between positive and negative classes.
# An area of 1.0 represents perfect prediction, 0.5 represents a model that is as good as random.
scoring = 'roc_auc'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("AUC: {:.3f} ({:.3f})".format(results.mean(), results.std()))

# Confusion matrix - a matrix representing the count of every prediction, the x-axis represents prediction outcome, while y-axis represents the actual value of Y, therefore, the higher diagonal matrix value, the better the prediction.
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=test_size, random_state=seed)
model = LogisticRegression()
model.fit(X_train, Y_train)
predicted = model.predict(X_test)
matrix = confusion_matrix(Y_test, predicted)
print(matrix)

# Classification report - provides a quick idea of the accuracy of model by displaying the precision, recall, F1-score, and support for each class.
from sklearn.metrics import classification_report
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
model = LogisticRegression()
model.fit(X_train,Y_train)
predicted = model.predict(X_test)
report = classification_report(Y_test,predicted)
print(report)

"""
For regression metric:
    1. Mean Absolute Error
    2. Mean Squared Error
    3. R^2
"""

raw_data = os.path.join(script_dir,"csv","housing.csv")
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data = pd.read_csv(raw_data, names=names,  comment='#')
value = data.values
X = value[:,0:-1]
Y = value[:,-1]
from sklearn.linear_model import LinearRegression

# Mean absolute error - some of absolute difference between prediction and the actual, it only gives the magniture but not the direction of the difference, the 0 represents no error.
kfold = KFold(n_splits=10, random_state=7)
model = LinearRegression()
scoring = "neg_mean_absolute_error"
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("MAE: {:.3f} {:.3f}").format(results.mean(), results.std())

# Mean squared error - 
