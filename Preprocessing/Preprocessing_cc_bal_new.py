# Import user defined Libraries
import Preprocessing.Preprocessing as prep


##################  ROUTINE ####################### 
# Set working directory
wd = "D:\RITESH\Data Science\GIT WD\KAGGLE - Home-Credit-Default-Risk"
prep.os.chdir(wd)

ccbal_dataset = wd+"\\Input\\credit_card_balance.csv"
# bureau_balance_dataset = wd+"\\Input\\bureau_balance.csv"

ccbal = prep.preprocessing(ccbal_dataset)

train_bureau_poscash_instpmt_clean_df = ccbal.imp_dataset(wd + '\\Output\\application_train_bureau_poscash_clean_new.csv')


# Import working dataset and isolate for the working data as per sample
ccbal.set_dataset1()
ccbal.isolate_dataset('SK_ID_CURR', None, train_bureau_poscash_instpmt_clean_df)

################################ CHANGING THE DATA TYPES ################################
feats_ignore = ['SK_ID_PREV','SK_ID_CURR','CCBAL_SK_ID_CURR']
ccbal.set_feats_ignore(feats_ignore)
num_ds_start, cat_ds_start = ccbal.define_dataset()

ccbal_agg_df = ccbal.basic_feature_extraction(ccbal.ds1_df, 'SK_ID_CURR')
ccbal_agg_df = ccbal.aggregate_datasets_2 (ccbal_agg_df,
                                                    'SK_ID_CURR',
                                                    ['_mean','_median','_min','_max','_std'],
                                                    ['_count'],
                                                    'mean',
                                                    'count',
                                                    {},
                                                    'CCBAL_',
                                                    [])


app_train_bureau_poscash_instpmt_ccbal_clean_df = ccbal.merge_datasets(train_bureau_poscash_instpmt_clean_df, 
                                                           ccbal_agg_df, 
                                                           'SK_ID_CURR',
                                                           'CCBAL_SK_ID_CURR','left')


app_train_bureau_poscash_instpmt_ccbal_clean_df.drop(['CCBAL_SK_ID_CURR'],axis=1,inplace=True)  
app_train_bureau_poscash_instpmt_ccbal_clean_df.fillna(value=0,inplace=True)
num_ds_end, cat_ds_end = ccbal.define_params(app_train_bureau_poscash_instpmt_ccbal_clean_df)

app_train_bureau_poscash_instpmt_ccbal_clean_df.to_csv(wd + "\\Output\\application_train_bureau_poscash_instpmt_ccbal_clean_new.csv",index=False)

