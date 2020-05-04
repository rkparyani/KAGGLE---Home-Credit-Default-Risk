####################### LIBRARIES ################################
# Data and visualization
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew #for some statistics
from sklearn.neighbors import KNeighborsClassifier
#rom sklearn.impute import MissingIndicator
# from autoimpute.imputations import SingleImputer, MultipleImputer

pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points
get_ipython().run_line_magic('matplotlib', 'inline')

#
## for ML Modelling
#from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC, LogisticRegression, LogisticRegressionCV
#from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
#from sklearn.kernel_ridge import KernelRidge
#from sklearn.pipeline import make_pipeline
#from sklearn.preprocessing import RobustScaler
#from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
#from sklearn.model_selection import KFold, cross_val_score, train_test_split
#from sklearn.metrics import mean_squared_error,confusion_matrix, accuracy_score, roc_curve, roc_auc_score
#from sklearn import tree
#import statsmodels as sm

###################  ROUTINE ####################### 
# Variables 
wd = "F:\Work Folder\Ritesh\ML\KAGGLE-CASESTUDY\KAGGLE-RISK-CS"
pos_cash_dataset = "Input\\POS_CASH_balance.csv"

train_clean_bureau = "Output\\application_train_bureau_clean.csv"
# Set working directory
os.chdir(wd)

# Import Libraries
import Model.FunctionLib as f

# Import working dataset
train_clean_bureau_df = pd.read_csv(train_clean_bureau)
pos_cash_df = pd.read_csv(pos_cash_dataset)

p_df = pos_cash_df.loc[pos_cash_df[pos_cash_df['SK_ID_CURR'].isin(train_clean_bureau_df.SK_ID_CURR)].index]
p_df.reset_index(drop=True)


del pos_cash_df

################################ CHANGING THE DATA TYPES ################################

# Seperate the categorical and numerical features
num_feats_p,cat_feats_p = f.distinct_feats(p_df)
print(len(num_feats_p),len(cat_feats_p))

# Change the datatype of categorical and numerical values
f.change_type(p_df,num_feats_p,count_threshold=5)

# Seperate the categorical and numerical features
# Create dataframe with Skew kurt, Missing val and Outliers for num_feats_imp_df
num_feats_p,cat_feats_p = f.distinct_feats(p_df)
for i in ['SK_ID_PREV','SK_ID_CURR']:
    num_feats_p.remove(i)
print(len(num_feats_p),len(cat_feats_p))

par_num_df_start, par_cat_df_start = f.get_params(p_df, num_feats_p, cat_feats_p)

############################# FEATURE TREATMENT AND EXTRACTION #########################
# As the features are expected to be extracted and grouped at SK_ID_CURR level 
# to synchronise at the Loan Application Client level.Hence we need to extract 
# aggregated information at Client level out of the dataset
# This means that treatment to individual columns would not be generallised as 
# we might loose information. Hence we will extract aggregated features first and then
# apply data correction (MISSING VALUES and OUTLIERS etc) at aggregate level based on the
# features qualitatively

p_agg_df = pd.DataFrame() 
# Create a object for aggregation at SK_ID_CURR level
#b_agg = b_df.groupby('SK_ID_CURR')

#Aggregating bureau data at Customer Id level

for feature in num_feats_p:  
    p_agg_df = f.get_aggregate_features_num(p_df,p_agg_df, feature,'SK_ID_CURR')
    p_agg_df[feature+'_std'] = np.where((p_agg_df[feature+'_std'].isna()==True) & 
            ((p_agg_df[feature+'_mean'])==(p_agg_df[feature+'_median'])), 
         0, 
         p_agg_df[feature+'_std'])
p_agg_df.insert(0,'SK_ID_CURR',p_agg_df.index)
p_agg_df.reset_index(drop=True)


for feature in cat_feats_p:    
    p_agg_cat = p_df.groupby('SK_ID_CURR')[feature].value_counts()
    for i in p_df[feature].unique():
        p_agg_df[feature+'_'+i+'_count'] = p_agg_cat.xs(key=i,level=1)
        p_agg_df[feature+'_'+i+'_count'].fillna(value=0,inplace=True)
        


# Assuming the NA values where Bureau does not have the data which mean that 
# in such scenarios the client does not have that entry which mean Zero 
# Imputing all the NA values with 0
p_agg_df.fillna(value=0,inplace=True)

p_agg_df.columns = p_agg_df.columns.map(f.remove_space)
p_agg_df = p_agg_df.add_prefix('POS_CASH_')   
    
    
num_feats_p_agg,cat_feats_p_agg = f.distinct_feats(p_agg_df)
for i in ['POS_CASH_SK_ID_CURR']:
    num_feats_p_agg.remove(i)
print(len(num_feats_p_agg),len(cat_feats_p_agg))
par_num_df_end, par_cat_df_end = f.get_params(p_agg_df, num_feats_p_agg, cat_feats_p_agg)

train_df = pd.merge(train_clean_bureau_df,p_agg_df, left_on='SK_ID_CURR',right_on = 'POS_CASH_SK_ID_CURR',how='left')
train_df.fillna(value=0,inplace=True)
train_df.drop(['POS_CASH_SK_ID_CURR'],axis=1,inplace=True)  


num_feats,cat_feats= f.distinct_feats(train_df)
for i in ['SK_ID_CURR']:
    num_feats.remove(i)
print(len(num_feats),len(num_feats))
par_num_df_end, par_cat_df_end = f.get_params(train_df, num_feats, cat_feats)


# Write the file to the Output directory for future reference
train_df.drop(train_df.filter(like='Unnamed').columns,axis=1,inplace=True)
train_df.to_csv(wd+"\\Output\\application_train_bureau_poscash_clean.csv")


