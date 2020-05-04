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
prev_application_dataset = "Input\\previous_application.csv"
pos_cash_dataset = "Input\\POS_CASH_balance.csv"
inst_pmt_dataset = "Input\\installments_payments.csv"
credit_card_balance_dataset = "Input\\credit_card_balance.csv"
train_bureau_poscash_instpmt_ccbal_clean = "Output\\application_train_bureau_poscash_instpmt_ccbal_clean.csv"

# Set working directory
os.chdir(wd)

# Import Libraries
import Model.FunctionLib as f

# Import working dataset
train_bureau_poscash_instpmt_ccbal_clean_df = pd.read_csv(train_bureau_poscash_instpmt_ccbal_clean)
prev_app_df = pd.read_csv(prev_application_dataset)
pos_cash_df = pd.read_csv(pos_cash_dataset)
inst_pmt_df = pd.read_csv(inst_pmt_dataset)
cc_bal_df = pd.read_csv(credit_card_balance_dataset)

pa_df = prev_app_df.loc[prev_app_df[prev_app_df['SK_ID_CURR'].isin(train_bureau_poscash_instpmt_ccbal_clean_df.SK_ID_CURR)].index]
pa_df.reset_index(drop=True)

pc_df = pos_cash_df.loc[pos_cash_df[pos_cash_df['SK_ID_PREV'].isin(pa_df.SK_ID_PREV)].index]
pc_df.reset_index(drop=True)

del prev_app_df, pos_cash_df

################################ CHANGING THE DATA TYPES ################################

# Seperate the categorical and numerical features
num_feats_pa,cat_feats_pa = f.distinct_feats(pa_df)
print(len(num_feats_pa),len(cat_feats_pa))

#num_feats_bal,cat_feats_bal = f.distinct_feats(bal_df)
#print(len(num_feats_bal),len(cat_feats_bal))

# Change the datatype of categorical and numerical values
f.change_type(pa_df,num_feats_pa,count_threshold=5)

# Seperate the categorical and numerical features
# Create dataframe with Skew kurt, Missing val and Outliers for num_feats_imp_df
num_feats_pa,cat_feats_pa = f.distinct_feats(pa_df)
for i in ['SK_ID_PREV','SK_ID_CURR']:
    num_feats_pa.remove(i)
print(len(num_feats_pa),len(cat_feats_pa))

par_num_df_start, par_cat_df_start = f.get_params(pa_df, num_feats_pa, cat_feats_pa)

############################# FEATURE TREATMENT AND EXTRACTION #########################
# As the features are expected to be extracted and grouped at SK_ID_CURR level 
# to synchronise at the Loan Application Client level.Hence we need to extract 
# aggregated information at Client level out of the dataset
# This means that treatment to individual columns would not be generallised as 
# we might loose information. Hence we will extract aggregated features first and then
# apply data correction (MISSING VALUES and OUTLIERS etc) at aggregate level based on the
# features qualitatively

pa_agg_df = pd.DataFrame() 
# Create a object for aggregation at SK_ID_CURR level
#b_agg = b_df.groupby('SK_ID_CURR')

#Aggregating bureau data at Customer Id level

for feature in num_feats_pa:  
    pa_agg_df = f.get_aggregate_features_num(pa_df,pa_agg_df, feature,'SK_ID_CURR')
#    na_ind = b_agg_df[(b_agg_df[feature + '_std'].isna()==True) & 
#                  ((b_agg_df[feature+'_mean'])==(b_agg_df[feature+'_median']))].index
#
#    b_agg_df.loc[na_ind][feature+'_std'].fillna(0)
#    b_agg_df.loc[na_ind][feature'_std'].isna().sum()
    pa_agg_df[feature+'_std'] = np.where((pa_agg_df[feature+'_std'].isna()==True) & 
            ((pa_agg_df[feature+'_mean'])==(pa_agg_df[feature+'_median'])), 
         0, 
         pa_agg_df[feature+'_std'])
pa_agg_df.insert(0,'SK_ID_CURR',pa_agg_df.index)
pa_agg_df.reset_index(drop=True)



for feature in cat_feats_pa:    
    pa_agg_cat = pa_df.groupby('SK_ID_CURR')[feature].value_counts()
    print(pa_df[feature].unique())
    for i in [x for x in pa_df[feature].unique() if str(x) !='nan']:
        pa_agg_df[feature+'_'+str(i)+'_count'] = pa_agg_cat.xs(key=i,level=1)
        pa_agg_df[feature+'_'+str(i)+'_count'].fillna(value=0,inplace=True)


### MOVING WITH pos_cash.csv and aggregating at Bureau_id level to add to bureau data 
bal_agg_df = pd.DataFrame() 
num_feats_bal,cat_feats_bal = f.distinct_feats(bal_df)
for i in ['SK_ID_BUREAU']:
    num_feats_bal.remove(i)
print(len(num_feats_bal),len(cat_feats_bal))

for feature in num_feats_bal:  
    bal_agg_df = f.get_aggregate_features_num(bal_df,bal_agg_df, feature, 'SK_ID_BUREAU')
    bal_agg_df[feature+'_std'] = np.where((bal_agg_df[feature+'_std'].isna()==True) & 
            ((bal_agg_df[feature+'_mean'])==(bal_agg_df[feature+'_median'])), 
            0, 
            bal_agg_df[feature+'_std'])
bal_agg_df.insert(0,'SK_ID_BUREAU',bal_agg_df.index)
bal_agg_df.reset_index(drop=True)

### Creating Categorical Feats
for feature in cat_feats_bal:    
    bal_agg_cat = bal_df.groupby('SK_ID_BUREAU')[feature].value_counts()
    for i in bal_df[feature].unique():
        bal_agg_df[feature+'_'+i+'_count'] = bal_agg_cat.xs(key=i,level=1)
        bal_agg_df[feature+'_'+i+'_count'].fillna(value=0,inplace=True)
        
# Merging with b_agg_df after aggregating at SK_ID_CURR level
bal_agg_df.rename(columns = {"SK_ID_BUREAU": "SK_ID_BUREAU_ID"}, inplace = True)
b_df = pd.merge(b_df,bal_agg_df, left_on='SK_ID_BUREAU',right_on = 'SK_ID_BUREAU_ID',how='left')



# Assuming the NA values where Bureau does not have the data which mean that 
# in such scenarios the client does not have that entry which mean Zero 
# Imputing all the NA values with 0
b_agg_df.fillna(value=0,inplace=True)

grp_b = b_df.groupby('SK_ID_CURR')
b_df.drop(b_df[['SK_ID_BUREAU_ID','MONTHS_BALANCE_mean','MONTHS_BALANCE_median','MONTHS_BALANCE_std']],axis=1,inplace=True)

for i in (b_df.filter(like='STATUS').columns):
    b_agg_df[i] = grp_b[i].sum()

b_agg_df.columns = b_agg_df.columns.map(f.remove_space)
b_agg_df = b_agg_df.add_prefix('BUREAU_')   
    
    
num_feats_b_agg,cat_feats_b_agg = f.distinct_feats(b_agg_df)
for i in ['BUREAU_SK_ID_CURR']:
    num_feats_b_agg.remove(i)
print(len(num_feats_b_agg),len(cat_feats_b_agg))
par_num_df_end, par_cat_df_end = f.get_params(b_agg_df, num_feats_b_agg, cat_feats_b_agg)

train_df = pd.merge(train_clean_df,b_agg_df, left_on='SK_ID_CURR',right_on = 'BUREAU_SK_ID_CURR',how='left')
train_df.fillna(value=0,inplace=True)
train_df.drop(['BUREAU_SK_ID_CURR'],axis=1,inplace=True)  


num_feats,cat_feats= f.distinct_feats(train_df)
for i in ['SK_ID_CURR']:
    num_feats.remove(i)
print(len(num_feats),len(num_feats))
par_num_df_end, par_cat_df_end = f.get_params(train_df, num_feats, cat_feats)

# Write the file to the Output directory for future reference
train_df.to_csv(wd+"\\Output\\application_train_bureau_clean.csv")



pa_df.NAME_YIELD_GROUP.value_counts()

