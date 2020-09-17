# Python Project Template

# 1. Prepare Problem
# a) Load libraries (the modules, classes and funcitons used)
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import os
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# b) Load dataset
script_dir = os.path.dirname(__file__)
raw_data = os.path.join(script_dir,"csv","iris.csv")
names = ["sepal.length","sepal.width","petal.length","petal.width","variety"]
dataset = read_csv(raw_data, names=names,  comment='#')

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
    # class distribution
    elif (arg == "distribution"):
        print(dataset.groupby('variety').size())
    else:
        print("{} is not a valid argument.".format(arg))

# b) Data visualizations
def visualize(arg):
    # Univariate plots
    if (arg == "univariate"):
        dataset.plot(kind="box", subplots=True, layout=(2,2), sharex=False, sharey=False)
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

# 3. Prepare Data
# Skipped

# 4. Evaluate Algorithms
# a) Split-out validation dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Spot-check and evaluate

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=10, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "{0}: {1:.6f} ({2:.6f})".format(name, cv_results.mean(), cv_results.std())
    print(msg)
    
# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))