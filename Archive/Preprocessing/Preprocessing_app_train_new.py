# Import user defined Libraries
import Model.FunctionLib as f
import Preprocessing.Preprocessing as prep


##################  ROUTINE ####################### 
# Set working directory
wd = "D:\RITESH\Data Science\GIT WD\KAGGLE - Home-Credit-Default-Risk"
prep.os.chdir(wd)

################ PREPROCESSING - APPLICATION_TRAIN.CSV ################################

# DataSet Path 
train_dataset = wd+"\\Input\\application_train.csv"

# Create object for Preprocessing class
p = prep.preprocessing(train_dataset, 0.1, 1)
p.set_dataset()

feats_ignore = ['TARGET', 'SK_ID_CURR']
p.set_feats_ignore(feats_ignore)
p.lst

num_ds_start, cat_ds_start = p.define_dataset()

# Identifying missing value feats
min_value = 65
p.missing_value_treatment(min_value)   

# Outlier Treatment     
normalized_feats = ['REGION_POPULATION_RELATIVE','EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','APARTMENTS_AVG','BASEMENTAREA_AVG','YEARS_BEGINEXPLUATATION_AVG','YEARS_BUILD_AVG','COMMONAREA_AVG',
'ELEVATORS_AVG','ENTRANCES_AVG','FLOORSMAX_AVG','FLOORSMIN_AVG','LANDAREA_AVG','LIVINGAPARTMENTS_AVG','LIVINGAREA_AVG','NONLIVINGAPARTMENTS_AVG','NONLIVINGAREA_AVG',
'APARTMENTS_MODE','BASEMENTAREA_MODE','YEARS_BEGINEXPLUATATION_MODE','YEARS_BUILD_MODE','COMMONAREA_MODE','ELEVATORS_MODE','ENTRANCES_MODE','FLOORSMAX_MODE','FLOORSMIN_MODE',
'LANDAREA_MODE','LIVINGAPARTMENTS_MODE','LIVINGAREA_MODE','NONLIVINGAPARTMENTS_MODE','NONLIVINGAREA_MODE','APARTMENTS_MEDI','BASEMENTAREA_MEDI','YEARS_BEGINEXPLUATATION_MEDI',
'YEARS_BUILD_MEDI','COMMONAREA_MEDI','ELEVATORS_MEDI','ENTRANCES_MEDI','FLOORSMAX_MEDI','FLOORSMIN_MEDI','LANDAREA_MEDI','LIVINGAPARTMENTS_MEDI','LIVINGAREA_MEDI','NONLIVINGAPARTMENTS_MEDI',
'NONLIVINGAREA_MEDI','FONDKAPREMONT_MODE','HOUSETYPE_MODE','TOTALAREA_MODE','WALLSMATERIAL_MODE','EMERGENCYSTATE_MODE']
    
p.outlier_treatment(normalized_feats)

# Missing value Imputations 
p.missing_value_imputations()

# Create Numerical features information dataframe
# CHECK  DIFF from param_df and check the mean median diff
num_ds_end, cat_ds_end = p.define_dataset()
train_clean_df = p.imp_df
# Export the file to output dir for the next module.
p.imp_df.to_csv(wd+'\\Output\\application_train_clean_new.csv')

################ PREPROCESSING - BUREAU AND BUREAU BALANCE ################################

bureau_dataset = "F:\Work Folder\Ritesh\ML\KAGGLE-CASESTUDY\KAGGLE-RISK-CS\Input\bureau.csv"
bureau_balance_dataset = "F:\Work Folder\Ritesh\ML\KAGGLE-CASESTUDY\KAGGLE-RISK-CS\Input\bureau_balance.csv"
train_clean = "Output\\application_train_clean.csv"

# Import working dataset and isolate for the working data as per sample
p.set_dataset(bureau_dataset, 0.1, 1, bureau_balance_dataset)
p.isolate_dataset('SK_ID_CURR', 'SK_ID_BUREAU', train_clean_df)












######################################TO DO ##########################################
# DAYS_EMPLOYED is highly correlated with DAYS_BIRTH hence the nan could have been derived from it. Use Autoimpute
