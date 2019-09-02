import os
import pandas as pd
script_dir = os.path.dirname(__file__)
raw_data = os.path.join(script_dir,"csv","pima-indians-diabetes.csv")
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv(raw_data, names=names,  comment='#')
pd.set_option('display.expand_frame_repr', False) # This line aviods shortening the print


from matplotlib import pyplot
# Histograms:
data.hist()
#pyplot.show()

# Density Plots:
data.plot(kind='density', subplots=True, layout=(3,3), sharex=False)
#pyplot.show()

# Box and whisker plots
data.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
pyplot.show()

# Correlation matrix plot
import numpy as np
correlations = data.corr()
# plot correlation matrix
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
pyplot.show()

# Scatterplot Matrix
from pandas.plotting import scatter_matrix
scatter_matrix(data)
pyplot.show()