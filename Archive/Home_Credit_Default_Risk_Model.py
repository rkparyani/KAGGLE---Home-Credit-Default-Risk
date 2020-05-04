####################### LIBRARIES ################################
# Data and visualization
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew #for some statistics
# from autoimpute.imputations import SingleImputer, MultipleImputer

pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points
get_ipython().run_line_magic('matplotlib', 'inline')


# for ML Modelling
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC, LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error,confusion_matrix, accuracy_score, roc_curve, roc_auc_score
from sklearn import tree
import statsmodels as sm

###################  ROUTINE ####################### 

# Variables 
wd = "D:\RITESH\Data Science\GIT WD\KAGGLE - Home-Credit-Default-Risk"
train_dataset = "application_train.csv"
# Set working directory
os.chdir(wd)

# Import Libraries
import FunctionLib as f

# Import working dataset
train_df = pd.read_csv(train_dataset)

# Summarize data infor from dataset    
#f.dataframeinfo(train_df)

# Seperate the target from working dataset
#target = train_df.TARGET
#x_df = train_df.drop(train_df[['TARGET']],axis=1)

# Create a new dataset same as train data
x_df = train_df.sample(frac=0.1, random_state=1).reset_index(drop=True)

# Delete the original dataset and work with Sample to free some space for processing.
del train_df

# Observe the features with missing values
f.get_missing_value_feats(x_df)

# Seperate the categorical and numerical features
num_feats,cat_feats = f.distinct_feats(x_df)
print(len(num_feats),len(cat_feats))
num_feats.remove('TARGET')

# Change the datatype of categorical and numerical values
f.change_type(x_df,num_feats,10)


# Seperate the categorical and numerical features
num_feats,cat_feats = f.distinct_feats(x_df)
print(len(num_feats),len(cat_feats))

# Identify na values exist and add them to a list
missing_value_feats = f.get_missing_value_feats(x_df)
missing_value_feats

# Calculate Missing Value percentage and Visualize
missing_values_perc_df = f.missing_val_perc(missing_value_feats,x_df)
val = missing_values_perc_df[0].sort_values(ascending=False)
f.plot_bar(val.index,(50,10),val)


# Check direct imputations such as remove the records for attributes which contain less than 5% of null values or remove
# attributes which contain more than 65% of null values.
imp_df = f.impute_values(x_df,missing_value_feats,65,action=True)
imp_df.reset_index(drop=True)


# How row in dataframe having more than x% NaN values
na_row_cnt = f.get_rowcnt_most_missing_val(imp_df,30)


# Identify na values exist and add them to a list
missing_value_feats = f.get_missing_value_feats(imp_df)
missing_value_feats

# if want to exclude any outliers add attribute name them to the list 
ignore_outliers = []

# Identify Outliers for the missing values
for i in imp_df.columns:
    out_df = imp_df[[i]].dropna()
    if i in num_feats:
        print(i)
        if i in ignore_outliers:
            print("For Outlier removal ignored")
        else:
            out_df = f.TurkyOutliers(out_df,i,outlier_step=1.5, drop=False)
            
# Outlier Treatment - There are lot of outliers 
# Outlier with high Skew and Kurt            
val = ((abs(imp_df[[x for x in imp_df.columns if x in num_feats]].skew()) > 1) 
       & (abs(imp_df[[x for x in imp_df.columns if x in num_feats]].kurt()) > 1))
outlier_feats = val[val==True].index

param_df = pd.DataFrame()
param_df['SKEW'] = imp_df[[x for x in imp_df.columns if x in num_feats]].skew()
param_df['KURT'] = imp_df[[x for x in imp_df.columns if x in num_feats]].kurt()
param_df['IS_NA'] = imp_df[[x for x in imp_df.columns if x in num_feats]].isna().sum()

# Observation - Numerical Features with higher Skew means that the outliers exist and kurt means that the number of outliers is very high,
# In such case need to do log transformation

# AMT INCOME TOTAL
feature = 'AMT_INCOME_TOTAL'

# High Skew and Kurt hence take log transformation
imp_df = f.log_transform(imp_df,feature)
imp_df.drop(imp_df[[feature]],axis=1,inplace=True)

# AMT_CREDIT
feature = 'AMT_CREDIT'

# High Skew and Kurt hence take log transformation
imp_df = f.log_transform(imp_df,feature)
imp_df.drop(imp_df[[feature]],axis=1,inplace=True)

# AMT_ANNUITY
feature = 'AMT_ANNUITY'

# High Skew and Kurt hence take log transformation
imp_df = f.log_transform(imp_df,feature)
imp_df.drop(imp_df[[feature]],axis=1,inplace=True)

# AMT_GOODS_PRICE
feature = 'AMT_GOODS_PRICE'

# High Skew and Kurt hence take log transformation
imp_df = f.log_transform(imp_df,feature)
imp_df.drop(imp_df[[feature]],axis=1,inplace=True)

# REGION_POPULATION_RELATIVE
feature = 'REGION_POPULATION_RELATIVE'
abcd,min,max = f.TurkyOutliers(imp_df,feature, drop=False)

# Only one value "0.073" is an outlier , making it an inlier
imp_df.loc[abcd] = round(max,3)


# APARTMENTS_AVG - No Change as the column is normalized and does not have any outliers.

# BASEMENTAREA_AVG
feature = 'BASEMENTAREA_AVG'
out_ind,min,max = f.TurkyOutliers(imp_df,feature, drop=False)
imp_df.loc[out_ind] = round(max,3)


# YEARS_BEGINEXPLUATATION_AVG
feature = 'YEARS_BEGINEXPLUATATION_AVG'
out_ind,min,max = f.TurkyOutliers(imp_df,feature, drop=False)
imp_df.loc[out_ind] = round(0.199,3)









############################ ROUGH WORK ###############################################

abcd,min,max = f.TurkyOutliers(imp_df,feature, drop=False)
imp_df[feature].value_counts()
imp_df[feature].describe()
imp_df[feature].skew()
f.hist_perc(imp_df.loc[out_ind],feature,20,0,1)

max
outlier_feats


abcd = f.TurkyOutliers(imp_df,'DAYS_EMPLOYED', drop=False)
temp = imp_df.loc[abcd]
imp_df.DAYS_EMPLOYED.value_counts()


nval = imp_df['DAYS_EMPLOYED']
nval[(nval<=-6506.0) | (nval>=3438.0)].count()











f.hist_perc(imp_df,'AMT_INCOME_TOTAL',20,0,1000000)

_ = f.TurkyOutliers(imp_df,'REGION_POPULATION_RELATIVE', drop=False)
imp_df.REGION_POPULATION_RELATIVE.skew()

df = imp_df[imp_df['REGION_POPULATION_RELATIVE']> 0.056648500000000004]
df.REGION_POPULATION_RELATIVE.unique()


