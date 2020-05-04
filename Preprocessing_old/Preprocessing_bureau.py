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

###################  ROUTINE ####################### 
# Variables 
wd = "F:\Work Folder\Ritesh\ML\KAGGLE-CASESTUDY\KAGGLE-RISK-CS"
bureau_dataset = "Input\\bureau.csv"
bureau_balance_dataset = "Input\\bureau_balance.csv"
train_clean = "Output\\application_train_clean.csv"
# Set working directory
os.chdir(wd)

# Import Libraries
import Model.FunctionLib as f

# Import working dataset
train_clean_df = pd.read_csv(train_clean)
bureau_df = pd.read_csv(bureau_dataset)
bureau_balance_df = pd.read_csv(bureau_balance_dataset)

b_df = bureau_df.loc[bureau_df[bureau_df['SK_ID_CURR'].isin(train_clean_df.SK_ID_CURR)].index]
b_df.reset_index(drop=True)

bal_df = bureau_balance_df.loc[bureau_balance_df[bureau_balance_df['SK_ID_BUREAU'].isin(b_df.SK_ID_BUREAU)].index]
bal_df.reset_index(drop=True)

del bureau_df, bureau_balance_df

################################ CHANGING THE DATA TYPES ################################

# Seperate the categorical and numerical features
num_feats_b,cat_feats_b = f.distinct_feats(b_df)
print(len(num_feats_b),len(cat_feats_b))

# Change the datatype of categorical and numerical values
f.change_type(b_df,num_feats_b,count_threshold=5)

# Seperate the categorical and numerical features
# Create dataframe with Skew kurt, Missing val and Outliers for num_feats_imp_df
num_feats_b,cat_feats_b = f.distinct_feats(b_df)
for i in ['SK_ID_BUREAU','SK_ID_CURR']:
    num_feats_b.remove(i)
print(len(num_feats_b),len(cat_feats_b))

par_num_df_start, par_cat_df_start = f.get_params(b_df, num_feats_b, cat_feats_b)

############################# FEATURE TREATMENT AND EXTRACTION #########################
# As the features are expected to be extracted and grouped at SK_ID_CURR level 
# to synchronise at the Loan Application Client level.Hence we need to extract 
# aggregated information at Client level out of the dataset
# This means that treatment to individual columns would not be generallised as 
# we might loose information. Hence we will extract aggregated features first and then
# apply data correction (MISSING VALUES and OUTLIERS etc) at aggregate level based on the
# features qualitatively

b_agg_df = pd.DataFrame() 
# Create a object for aggregation at SK_ID_CURR level
#b_agg = b_df.groupby('SK_ID_CURR')

#Aggregating bureau data at Customer Id level

for feature in num_feats_b:  
    b_agg_df = f.get_aggregate_features_num(b_df,b_agg_df, feature,'SK_ID_CURR')
#    na_ind = b_agg_df[(b_agg_df[feature + '_std'].isna()==True) & 
#                  ((b_agg_df[feature+'_mean'])==(b_agg_df[feature+'_median']))].index
#
#    b_agg_df.loc[na_ind][feature+'_std'].fillna(0)
#    b_agg_df.loc[na_ind][feature'_std'].isna().sum()
    b_agg_df[feature+'_std'] = np.where((b_agg_df[feature+'_std'].isna()==True) & 
            ((b_agg_df[feature+'_mean'])==(b_agg_df[feature+'_median'])), 
         0, 
         b_agg_df[feature+'_std'])
b_agg_df.insert(0,'SK_ID_CURR',b_agg_df.index)
b_agg_df.reset_index(drop=True)

for feature in cat_feats_b:    
    b_agg_cat = b_df.groupby('SK_ID_CURR')[feature].value_counts()
    for i in b_df[feature].unique():
        b_agg_df[feature+'_'+i+'_count'] = b_agg_cat.xs(key=i,level=1)
        b_agg_df[feature+'_'+i+'_count'].fillna(value=0,inplace=True)
        
### MOVING WITH Bureau_Balance.csv and aggregating at Bureau_id level to add to bureau data 
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