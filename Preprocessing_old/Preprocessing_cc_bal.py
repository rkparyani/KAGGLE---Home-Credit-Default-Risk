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
cc_bal_dataset = "Input\\credit_card_balance.csv"
train_bureau_poscash_instpmt_clean = "Output\\application_train_bureau_poscash_instpmt_clean.csv"
# Set working directory
os.chdir(wd)

# Import Libraries
import Model.FunctionLib as f

# Import working dataset
train_bureau_poscash_instpmt_clean_df = pd.read_csv(train_bureau_poscash_instpmt_clean)
cc_bal_df = pd.read_csv(cc_bal_dataset)

c_df = cc_bal_df.loc[cc_bal_df[cc_bal_df['SK_ID_CURR'].isin(train_bureau_poscash_instpmt_clean_df.SK_ID_CURR)].index]
c_df.reset_index(drop=True)

del cc_bal_df

################################ CHANGING THE DATA TYPES ################################

# Seperate the categorical and numerical features
num_feats_c,cat_feats_c = f.distinct_feats(c_df)
print(len(num_feats_c),len(cat_feats_c))

# Change the datatype of categorical and numerical values
f.change_type(c_df,num_feats_c,count_threshold=5)

# Seperate the categorical and numerical features
# Create dataframe with Skew kurt, Missing val and Outliers for num_feats_imp_df
num_feats_c,cat_feats_c = f.distinct_feats(c_df)
for i in ['SK_ID_CURR','SK_ID_PREV']:
    num_feats_c.remove(i)
print(len(num_feats_c),len(cat_feats_c))

par_num_df_start, par_cat_df_start = f.get_params(c_df, num_feats_c, cat_feats_c)

############################# FEATURE TREATMENT AND EXTRACTION #########################
# As the features are expected to be extracted and grouped at SK_ID_CURR level 
# to synchronise at the Loan Application Client level.Hence we need to extract 
# aggregated information at Client level out of the dataset
# This means that treatment to individual columns would not be generallised as 
# we might loose information. Hence we will extract aggregated features first and then
# apply data correction (MISSING VALUES and OUTLIERS etc) at aggregate level based on the
# features qualitatively

c_agg_df = pd.DataFrame() 
# Create a object for aggregation at SK_ID_CURR level
#b_agg = b_df.groupby('SK_ID_CURR')

#Aggregating bureau data at Customer Id level

for feature in num_feats_c:  
    c_agg_df = f.get_aggregate_features_num(c_df,c_agg_df, feature,'SK_ID_CURR')
    c_agg_df[feature+'_std'] = np.where((c_agg_df[feature+'_std'].isna()==True) & 
            ((c_agg_df[feature+'_mean'])==(c_agg_df[feature+'_median'])), 
         0, 
         c_agg_df[feature+'_std'])
c_agg_df.insert(0,'SK_ID_CURR',c_agg_df.index)
c_agg_df.reset_index(drop=True)

for feature in cat_feats_c:    
    c_agg_cat = c_df.groupby('SK_ID_CURR')[feature].value_counts()
    for i in c_df[feature].unique():
        c_agg_df[feature+'_'+i+'_count'] = c_agg_cat.xs(key=i,level=1)
        c_agg_df[feature+'_'+i+'_count'].fillna(value=0,inplace=True)
        

# Assuming the NA values where Bureau does not have the data which mean that 
# in such scenarios the client does not have that entry which mean Zero 
# Imputing all the NA values with 0
c_agg_df.fillna(value=0,inplace=True)

c_agg_df.columns = c_agg_df.columns.map(f.remove_space)
c_agg_df = c_agg_df.add_prefix('CREDIT_CARD_BAL_')   
    
    
num_feats_c_agg,cat_feats_c_agg = f.distinct_feats(c_agg_df)
for i in ['CREDIT_CARD_BAL_SK_ID_CURR']:
    num_feats_c_agg.remove(i)
print(len(num_feats_c_agg),len(cat_feats_c_agg))
par_num_df_end, par_cat_df_end = f.get_params(c_agg_df, num_feats_c_agg, cat_feats_c_agg)

train_df = pd.merge(train_bureau_poscash_instpmt_clean_df,c_agg_df, left_on='SK_ID_CURR',right_on = 'CREDIT_CARD_BAL_SK_ID_CURR',how='left')
train_df.fillna(value=0,inplace=True)
train_df.drop(['CREDIT_CARD_BAL_SK_ID_CURR'],axis=1,inplace=True)  


num_feats,cat_feats= f.distinct_feats(train_df)
for i in ['SK_ID_CURR']:
    num_feats.remove(i)
print(len(num_feats),len(num_feats))
par_num_df_end, par_cat_df_end = f.get_params(train_df, num_feats, cat_feats)

# Write the file to the Output directory for future reference
train_df.to_csv(wd+"\\Output\\application_train_bureau_poscash_instpmt_ccbal_clean.csv")