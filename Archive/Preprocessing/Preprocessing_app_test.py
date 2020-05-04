####################### LIBRARIES ################################
# Data and visualization
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.impute import MissingIndicator
# from autoimpute.imputations import SingleImputer, MultipleImputer
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points

###################  ROUTINE ####################### 
# Variables 
wd = "F:\Work Folder\Ritesh\ML\KAGGLE-CASESTUDY\KAGGLE-RISK-CS"
test_dataset = "Input\\application_test.csv"

# Set working directory
os.chdir(wd)

# Import Libraries
import Model.FunctionLib as f

# Import working dataset
test_df = pd.read_csv(test_dataset)

# Summarize data infor from dataset    
#f.dataframeinfo(train_df)

# Seperate the target from working dataset
#target = train_df.TARGET
#x_df = train_df.drop(train_df[['TARGET']],axis=1)

# Create a new dataset same as train data
x_df = test_df.sample(frac=0.1, random_state=1).reset_index(drop=True)

# Delete the original dataset and work with Sample to free some space for processing.
del test_df

################################ CHANGING THE DATA TYPES ################################

# Observe the features with missing values
f.get_missing_value_feats(x_df)

# Seperate the categorical and numerical features
num_feats,cat_feats = f.distinct_feats(x_df)
print(len(num_feats),len(cat_feats))
#num_feats.remove('TARGET')
num_feats.remove('SK_ID_CURR')

# Change the datatype of categorical and numerical values
f.change_type(x_df,num_feats,count_threshold=5)

# Seperate the categorical and numerical features
num_feats,cat_feats = f.distinct_feats(x_df)
print(len(num_feats),len(cat_feats))
num_feats.remove('SK_ID_CURR')

# Create dataframe with Skew kurt, Missing val and Outliers for num_feats_imp_df
par_num_df_start, par_cat_df_start = f.get_params(x_df, num_feats, 
                                      cat_feats)


############################# IDENTIFYING MISSING FEATS #########################

# Identify na values exist and add them to a list
missing_value_feats = f.get_missing_value_feats(x_df)
missing_value_feats

# Calculate Missing Value percentage and Visualize
missing_values_perc_df = f.missing_val_perc(missing_value_feats,x_df)
val = missing_values_perc_df[0].sort_values(ascending=False)
f.plot_bar(val.index,(50,10),val)

#################### REMOVING THE VALUES DIRECTLY ##########################
# Check direct imputations such as remove the records for attributes which contain less than 5% of null values or remove
# attributes which contain more than 65% of null values.
imp_df = f.impute_values(x_df,missing_value_feats,65,action=True)
imp_df.reset_index(drop=True)


# How row in dataframe having more than x% NaN values
na_row_cnt = f.get_rowcnt_most_missing_val(imp_df,30)

# Identify na values exist and add them to a list
missing_value_feats = f.get_missing_value_feats(imp_df)
missing_value_feats
            
##################### OUTLIERS TREATMENT ###################################
# Outlier with high Skew and Kurt         
            
# Find the num and cat feats for imp_df
num_feats_imp_df,cat_feats_imp_df = f.distinct_feats(imp_df)
#num_feats_imp_df.remove('TARGET')
num_feats_imp_df.remove('SK_ID_CURR')
print(len(num_feats_imp_df),len(cat_feats_imp_df))

# All Normalized treatment is same : Irrespective of Skew and Kurt Identify outliers and fill with its boundary value
normalized_feats = ['REGION_POPULATION_RELATIVE','EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','APARTMENTS_AVG','BASEMENTAREA_AVG','YEARS_BEGINEXPLUATATION_AVG','YEARS_BUILD_AVG','COMMONAREA_AVG',
'ELEVATORS_AVG','ENTRANCES_AVG','FLOORSMAX_AVG','FLOORSMIN_AVG','LANDAREA_AVG','LIVINGAPARTMENTS_AVG','LIVINGAREA_AVG','NONLIVINGAPARTMENTS_AVG','NONLIVINGAREA_AVG',
'APARTMENTS_MODE','BASEMENTAREA_MODE','YEARS_BEGINEXPLUATATION_MODE','YEARS_BUILD_MODE','COMMONAREA_MODE','ELEVATORS_MODE','ENTRANCES_MODE','FLOORSMAX_MODE','FLOORSMIN_MODE',
'LANDAREA_MODE','LIVINGAPARTMENTS_MODE','LIVINGAREA_MODE','NONLIVINGAPARTMENTS_MODE','NONLIVINGAREA_MODE','APARTMENTS_MEDI','BASEMENTAREA_MEDI','YEARS_BEGINEXPLUATATION_MEDI',
'YEARS_BUILD_MEDI','COMMONAREA_MEDI','ELEVATORS_MEDI','ENTRANCES_MEDI','FLOORSMAX_MEDI','FLOORSMIN_MEDI','LANDAREA_MEDI','LIVINGAPARTMENTS_MEDI','LIVINGAREA_MEDI','NONLIVINGAPARTMENTS_MEDI',
'NONLIVINGAREA_MEDI','FONDKAPREMONT_MODE','HOUSETYPE_MODE','TOTALAREA_MODE','WALLSMATERIAL_MODE','EMERGENCYSTATE_MODE']
    
other_feats = [x for x in num_feats_imp_df if x not in normalized_feats]

# Anamolies and data correction.
# DAYS_EMPLOYED has abnormal value '365243' which would be changed to nan for imputation at a later stage
feature = 'DAYS_EMPLOYED'
imp_df[feature].loc[imp_df[imp_df[feature]==365243].index] = np.nan

# Log transformation of all numerical non normalized highly skewed values to remove outliers
for feature in other_feats:
    print(feature)
    imp_df = f.log_transform(imp_df,feature)
    imp_df.drop(imp_df[[feature]],axis=1,inplace=True)


#normalized_num_feats_imp_df = [x for x in normalized_feats if x in num_feats_imp_df]
num_feats_imp_df,cat_feats_imp_df = f.distinct_feats(imp_df)
num_feats_imp_df.remove('SK_ID_CURR')
print(len(num_feats_imp_df),len(cat_feats_imp_df))

for i in num_feats_imp_df:
    print(i)
    out_l,out_r,min,max = f.TurkyOutliers(imp_df,i,drop=False)
    if (len(out_l)|len(out_r)) > 0:
        imp_df[i].loc[out_l] = round(min,3)
        imp_df[i].loc[out_r] = round(max,3)

#################################### MISSING VALUES #############################
# Since the numerical univariate distribution are symmetrical now with no difference 
# between median and mean. Lets impute all the numerical missing values with mean

# Record missing values for further validations:
indicator = MissingIndicator(missing_values=np.nan)
mask_missing_values_only = indicator.fit_transform(imp_df)
mask_missing_values_only.shape

# Num missing values imputations
imp_df[num_feats_imp_df] = imp_df[num_feats_imp_df].fillna(value = imp_df[num_feats_imp_df].mean())

# Left missing values are categorical.
missing_feats_cat = f.get_missing_value_feats(imp_df)

par_num_df, par_cat_df = f.get_params(imp_df, num_feats_imp_df, 
                                      cat_feats_imp_df)
# Categorical values where mode frequency is more than 80% - Impute na with Mode
# If not then use the KNN model to impute the values
mode_threshold = 80

for feature in missing_feats_cat:
    if par_cat_df.loc[feature]['MODE_PERCENTAGE'] > mode_threshold:
        imp_df[feature].fillna(value= par_cat_df.loc[feature]['MODE'],inplace=True)
        print("Method : MODE , Feature : {} , Mode_Percentage : {}".format(feature, par_cat_df.loc[feature]['MODE_PERCENTAGE']))
    else: 
        imp_list, score = f.impute_knn_classifier(imp_df,feature, 49)
        imp_df[feature].fillna(value = imp_list,inplace=True)
        print("Method : KNN , Feature : {} , Imputation Accuracy Score : {}".format(feature,score))

# Create Numerical features information dataframe
# CHECK  DIFF from param_df and check the mean median diff
num_feats_imp_df,cat_feats_imp_df = f.distinct_feats(imp_df)
num_feats_imp_df.remove('SK_ID_CURR')
print(len(num_feats_imp_df),len(cat_feats_imp_df))

par_num_df_end, par_cat_df_end = f.get_params(imp_df, num_feats_imp_df, 
                                      cat_feats_imp_df)

# Delete the data objects created for processing
del par_num_df, par_cat_df, cat_feats, feature, i, mask_missing_values_only, max, min, missing_values_perc_df, na_row_cnt, normalized_feats,num_feats, other_feats, test_dataset, val, x_df, mode_threshold, cat_feats_imp_df, num_feats_imp_df, imp_list, score

# =============================================================================
############## MISSING VALUES IMPUTATIONS USING FEATURE CORRELATIONS ########## 
# # Code Block to find the correlated features for various features including featues including each category correlations
# # This can be used to derive/impute na values when the correlations are strong with other features using sklearn.Impute Iterativeimputer        
# # Not using this approach for now as there are no strong correlations with missing value columns
#
# 
#x_df_dum = pd.get_dummies(imp_df)
#x_df_Default_dum = x_df_dum[x_df_dum['TARGET']==1]
#
#x_df_dum.columns = x_df_dum.columns.map(f.remove_space)
#x_df_Default_dum.columns = x_df_Default_dum.columns.map(f.remove_space)
#
# # General correlations wrt Correlations in case of default.
#x_corr_default = x_df_Default_dum.corr()
#x_corr = x_df_dum.corr()
#
#corr_threshold = 0.2
#get_highly_corr_feats = f.corr_feats (x_df_dum,x_df_dum.columns,corr_threshold)
#get_highly_corr_feats = pd.DataFrame(get_highly_corr_feats)
#print('Highly correlated features description more than pearsonsr',corr_threshold)
# 
# =============================================================================

    
# Export the file to output dir for the next module.
imp_df.to_csv(wd+'\\Output\\application_test_clean.csv')
imp_df.columns

######################################TO DO ##########################################
# DAYS_EMPLOYED is highly correlated with DAYS_BIRTH hence the nan could have been derived from it. Use Autoimpute


