# Python Project Template

# 1. Prepare Problem
# a) Load libraries (the modules, classes and funcitons used)
# Load libraries
import numpy
from numpy import arange
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
import os
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# b) Load dataset
script_dir = os.path.dirname(__file__)
raw_data = os.path.join(script_dir,"csv","housing.csv")
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO',
'B', 'LSTAT', 'MEDV']
dataset = read_csv(raw_data, names=names, comment='#')


# 2. Summarize Data
# a) Descriptive statistics
def summarize(arg):
    # shape
    if(arg == "shape"):
        print("There are {} lines of tuples, {} parameters each.".format(dataset.shape[0], dataset.shape[1]))
    # Peek at the data
    elif (arg == "head"):
        print("the first 20 lines are:")
        print(dataset.head(20))
    # descriptions
    elif (arg == "describe"):
        print("Viewing statistic description:") 
        print(dataset.describe())
    else:
        print("{} is not a valid argument.".format(arg))

        
# b) Data visualizations
def visualize(arg):
    # Univariate plots
    if (arg == "univariate"):
        dataset.plot(kind="box", subplots=True, layout=(14,14), sharex=False, sharey=False)
        pyplot.show()
    # Histograms
    elif (arg == "histograms"):
        dataset.hist()
        pyplot.show()
    # Multivariate plots
    elif (arg == "multivariate"):
        scatter_matrix(dataset)
        pyplot.show()
    # Correlation
    elif (arg == "correlation"):
        correlations = dataset.corr()
        # plot correlation matrix
        fig = pyplot.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(correlations, vmin=-1, vmax=1)
        fig.colorbar(cax)
        pyplot.show()
    else:
        print("{} is not a valid argument.".format(arg))

visualize("correlation")

# 3. Prepare Data
# a) Data Cleaning (remove duplicates, mark or input the missing values)
# b) Feature Selection (remove redundant features)
# c) Data Transforms (rescale and redistribute)

# 4. Evaluate Algorithms
# a) Split-out validation dataset
# b) Test options and evaluation metric
# c) Spot Check Algorithms
# d) Compare Algorithms
# this step takes time, need at least 3-5 well performing machine learning algorithms

# 5. Improve Accuracy (choose one between a or b)
# a) Algorithm Tuning (search for a combination of parameters for each algortihm that yields the best results)
# b) Ensembles (combine the prediction of multiple models)

# 6. Finalize Model
# a) Predictions on validation dataset
# b) Create standalone model on entire training dataset
# c) Save model for later use