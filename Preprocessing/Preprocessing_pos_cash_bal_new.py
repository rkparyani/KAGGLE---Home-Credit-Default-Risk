# Import user defined Libraries
import Preprocessing.Preprocessing as prep


##################  ROUTINE ####################### 
# Set working directory
wd = "D:\RITESH\Data Science\GIT WD\KAGGLE - Home-Credit-Default-Risk"
prep.os.chdir(wd)

pos_cash_dataset = wd+"\\Input\\POS_CASH_balance.csv"
# bureau_balance_dataset = wd+"\\Input\\bureau_balance.csv"

poscash = prep.preprocessing(pos_cash_dataset)

train_bureau_clean_df = poscash.imp_dataset(wd + '\\Output\\application_train_bureau_clean_new.csv')
# train_clean_df.drop(train_clean_df[['Unnamed: 0']],axis=1,inplace=True)

# Import working dataset and isolate for the working data as per sample
poscash.set_dataset1()
poscash.isolate_dataset('SK_ID_CURR', None, train_bureau_clean_df)


################################ CHANGING THE DATA TYPES ################################
feats_ignore = ['SK_ID_PREV','SK_ID_CURR','POS_CASH_SK_ID_CURR']
poscash.set_feats_ignore(feats_ignore)
num_ds_start, cat_ds_start = poscash.define_dataset()

pos_agg_df = poscash.basic_feature_extraction(poscash.ds1_df, 'SK_ID_CURR')
pos_agg_df.reset_index(drop=True,inplace=True) 

pos_agg_df = poscash.aggregate_datasets_2 (pos_agg_df,
                                                    'SK_ID_CURR',
                                                    ['_mean','_median','_min','_max','_std'],
                                                    ['_count'],
                                                    'mean',
                                                    'count',
                                                    {},
                                                    'POS_CASH_',
                                                    [])


app_train_bureau_poscash_clean_df = poscash.merge_datasets(train_bureau_clean_df, 
                                                           pos_agg_df, 
                                                           'SK_ID_CURR',
                                                           'POS_CASH_SK_ID_CURR','left')


app_train_bureau_poscash_clean_df.drop(['POS_CASH_SK_ID_CURR'],axis=1,inplace=True)  
app_train_bureau_poscash_clean_df.fillna(value=0,inplace=True)
num_ds_end, cat_ds_end = poscash.define_params(app_train_bureau_poscash_clean_df)

app_train_bureau_poscash_clean_df.to_csv(wd + "\\Output\\application_train_bureau_poscash_clean_new.csv", index = False)
