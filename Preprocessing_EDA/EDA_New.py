
###################  LOAD THE LIBRARIES ####################### 

# for data mining
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
import warnings
from scipy import stats
from scipy.stats import norm, skew #for some statistics
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
py.init_notebook_mode(connected=True)
import matplotlib.ticker as mticker
import random



###################  ROUTINE ####################### 

# Variables 
wd = "D:\RITESH\Data Science\GIT WD\KAGGLE - Home-Credit-Default-Risk"
train_dataset = "Input/application_train.csv"
# Set working directory
os.chdir(wd)

# Import Libraries
import Model.FunctionLib as f

# Import working dataset
df = pd.read_csv(train_dataset)


### TARGET ###
df.TARGET.value_counts(normalize=True).plot.bar()
df.TARGET.value_counts()/df.TARGET.size*100

one = df[df['SK_ID_CURR']==100002]
one_T = df[df['SK_ID_CURR']==100002].T

stats = df[['SK_ID_CURR','TARGET']]
stats['CREDIT_INCOME_ratio'] = df.AMT_CREDIT/df.AMT_INCOME_TOTAL
stats['ANNUITY_INCOME_ratio'] = df.AMT_ANNUITY/(df.AMT_INCOME_TOTAL)


num,cat = f.distinct_feats(df)