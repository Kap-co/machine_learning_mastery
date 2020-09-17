import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
script_dir = os.path.dirname(__file__)
raw_data = os.path.join(script_dir,"csv","pima-indians-diabetes.csv")
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv(raw_data, names=names,  comment='#')
value = data.values
X = value[:,0:-1]
Y = value[:,-1]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.33, random_state=7)
model = LogisticRegression()
model.fit(X_train, Y_train)

# Pickle
# make the machine learning algorithms and models to serialized format to a file or deserialize to make new predictions.

# save
from pickle import dump
filename = 'finalized_model.sav'
dump(model, open(filename, 'wb'))

# load
from pickle import load
loaded_model = load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result)


# Joblib, a slightly better save-load method for algorithms that require a lot of parameters or store the entire dataset (k_nearest Neighbors)
from sklearn.externals.joblib import dump
# save
filename = 'finalized_model.sav'
dump(model,filename)

# load
from sklearn.externals.joblib import load
loaded_model = load(filename)
result = loaded_model.score(X_test,Y_test)
print(result)