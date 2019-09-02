# See curent working directory
import os
cwd = os.getcwd()

# Run from current scripts's directory or beyond
script_dir = os.path.dirname(__file__)
dataset = os.path.join(script_dir,"ADDITIONAL SUBDIRECTORY","YOUR FILE NAME")

# Three ways to import csv file
# 1. standard import
import csv
import numpy as np
script_dir = os.path.dirname(__file__)
dataset = os.path.join(script_dir,"csv","hello_world.csv")
with open(dataset,'r') as f: # File open
    reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONE)
    fList = list(reader) 
    data = np.array(fList).astype('float')

# 2. import with numpy
#import numpy as np
script_dir = os.path.dirname(__file__)
raw_data = os.path.join(script_dir,"csv","hello_world.csv")
dataset = open(raw_data, 'rb')
data = np.loadtxt(dataset, delimiter=",")

# Load CSV from URL using numpy
#import numpy as np
#from urllib.request import urlopen
#url = "https://goo.gl/vhm1eU"
#raw_data = urlopen(url)
#data = np.loadtxt(raw_data, delimiter=",")

# 3. import with pandas
import pandas as pd
script_dir = os.path.dirname(__file__)
raw_data = os.path.join(script_dir,"csv","hello_world.csv")
names = ["X1","X2","Y"]
data = pd.read_csv(raw_data, names=names,  comment='#') # comment='#' indicate commnets are denoted in # in csv file
print(data)