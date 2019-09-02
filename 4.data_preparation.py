"""
Three general steps to preapare data:
    1. Split the dataset into the input and output variables for machine learning
    2. Apply a pre-processing transform to the input variable, ex normalizing
    3. Summarize the data to show the change.
"""
import os
import pandas as pd
import numpy as np
script_dir = os.path.dirname(__file__)
raw_data = os.path.join(script_dir,"csv","pima-indians-diabetes.csv")
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv(raw_data, names=names,  comment='#')
pd.set_option('display.expand_frame_repr', False) # This line aviods shortening the print


# Organizing data, seperating X and Y, because when we prepare data, it often does not include the output.
value = data.values # This will exclude all the attribute and instance heading
# Seperate value into input and output components
X = value[:,0:-1]
Y = value[:,-1]
np.set_printoptions(precision=3) # Three number after decimal points when printed.


# 1. Data rescaling: Change the scaling of data by attribute unit from a minimum to maximum, 0 to 1 value is used here aka normalization, this is crucial to some algorithms like k-nearest, all regression-related, neural network, gradient decent.
from sklearn.preprocessing import MinMaxScaler

# Rescale the data into 0,1 aka normalize
scaler = MinMaxScaler(feature_range=(0,1))
rescaledX = scaler.fit_transform(X)

# summarize transformed data
print("Sample rescaled data:")
print(rescaledX[0:5,:]) # First five


# 2. Standardize data: transform attributes to normal distrubution, setting the mean as 0 and sd to 1.
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X)
standarizedX = scaler.transform(X)

print("Sample standarized data:")
print(standarizedX[0:5,:])

# 3. Normailize data: A bit different concept from MinMax, the normalizer rescales each observation(row) to have a length of 1, this pre-processing method can be useful for sparse datasets.
from sklearn.preprocessing import Normalizer

# Rescale the data into 0,1 aka normalize
scaler = Normalizer().fit(X)
normalizedX = scaler.transform(X)

print("Sample normalized data:")
print(normalizedX[0:5,:])

# 4. Binarized Data: Make all values above the threshold are makred 1 and all equal to or  bolow are marked as 0. Useful when you have probabilities that you want to make crisp values.
from sklearn.preprocessing import Binarizer
binarizer = Binarizer(threshold=0.0).fit(X)
binaryX = binarizer.transform(X)

print("Sample binarized data:")
print(binaryX[0:5,:])
