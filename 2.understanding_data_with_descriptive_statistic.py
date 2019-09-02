"""
There are 7 recipes needed to better understand the machine learning data, they are:
1. Take a peek at your raw data.
2. Review the dimensions of your dataset.
3. Review the data types of attributes in your data.
4. Summarize the distribution of instances across classes in your dataset.
5. Summarize your data using descriptive statistics.
6. Understand the relationships in your data using correlations.
7. Review the skew of the distributions of each attribute.
"""

# Import csv file as pandas
import os
import pandas as pd
script_dir = os.path.dirname(__file__)
raw_data = os.path.join(script_dir,"csv","pima-indians-diabetes.csv")
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv(raw_data, names=names,  comment='#')
pd.set_option('display.expand_frame_repr', False) # This line aviods shortening the print

# 1. Peek at your data: Top n lines of the data.
print("1. The peek of data:")
peek = data.head(5)
print (peek)

# 2. Dimensions of your data: Review what is the dimemsionality of the data.
print("2. The dimensionality of data:")
print(data.shape)

# 3. Data type for each attribute.
print("3. The data type for each attribute:")
print(data.dtypes)

# 4. Descriptive statistics.
print("4. Deiscriptive statistics:")
print(data.describe())

# 5. Class distribution.
print("5. Class distribution:")
class_counts = data.groupby('class').size()
print(class_counts)

# 6. Correlation between attributes.
print("6. Correlation between attributes:")
correlations = data.corr(method='pearson')
print(correlations)

# 7. Skewness of univariate distributions.
print("7. Skewness:")
print(data.skew())