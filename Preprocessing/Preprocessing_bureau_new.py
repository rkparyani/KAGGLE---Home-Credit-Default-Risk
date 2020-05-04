# Import user defined Libraries
import Preprocessing.Preprocessing as prep


##################  ROUTINE ####################### 
# Set working directory
wd = "D:\RITESH\Data Science\GIT WD\KAGGLE - Home-Credit-Default-Risk"
prep.os.chdir(wd)

bureau_dataset = wd+"\\Input\\bureau.csv"
bureau_balance_dataset = wd+"\\Input\\bureau_balance.csv"

bureau = prep.preprocessing(bureau_dataset, bureau_balance_dataset)

train_clean_df = bureau.imp_dataset(wd + '\\Output\\application_train_clean_new.csv')
#train_clean_df.drop(train_clean_df[['Unnamed: 0.1']],axis=1,inplace=True)

# Import working dataset and isolate for the working data as per sample
bureau.set_dataset1()
bureau.set_dataset2()

bureau.isolate_dataset('SK_ID_CURR', 'SK_ID_BUREAU', train_clean_df)

################################ CHANGING THE DATA TYPES ################################
feats_ignore = ['SK_ID_BUREAU','SK_ID_CURR','BUREAU_SK_ID_CURR']
bureau.set_feats_ignore(feats_ignore)
num_ds_start, cat_ds_start = bureau.define_dataset()

b_agg_df = bureau.basic_feature_extraction(bureau.ds1_df, 'SK_ID_CURR')
b_agg_df = bureau._column_name_streamline(b_agg_df, 'BUREAU_')

bal_agg_df = bureau.basic_feature_extraction(bureau.ds2_df, 'SK_ID_BUREAU')
bal_agg_df.rename(columns = {"SK_ID_BUREAU": "SK_BUREAU_ID"}, inplace = True)
bal_agg_df = bureau.merge_datasets(bureau.ds1_df[['SK_ID_CURR', 'SK_ID_BUREAU']], bal_agg_df, 'SK_ID_BUREAU','SK_BUREAU_ID','right')

#b_agg_df = b_agg_df_bkp.copy()
#bureau.ds1_df = bureau.ds1_df_bkp.copy()
#b_agg_df = bureau.aggregate_datasets (b_agg_df, 'SK_ID_CURR', 'STATUS', 'BUREAU_', 
#                    False
#                    )

bal_agg_df = bureau.aggregate_datasets_2 (bal_agg_df,
                                                    'SK_ID_CURR',
                                                    ['_mean','_median','_min','_max','_std'],
                                                    ['_count'],
                                                    'mean',
                                                    'count',
                                                    {},
                                                    'BUREAU_',
                                                    ['MONTHS_BALANCE_mean','MONTHS_BALANCE_median','MONTHS_BALANCE_std'])

b_agg_df = bureau.merge_datasets(b_agg_df, bal_agg_df, 'BUREAU_SK_ID_CURR','BUREAU_SK_ID_CURR', 'left')
app_train_bureau_clean_df = bureau.merge_datasets(train_clean_df, b_agg_df, 'SK_ID_CURR','BUREAU_SK_ID_CURR','left')
app_train_bureau_clean_df.fillna(value=0,inplace=True)
app_train_bureau_clean_df.drop(['BUREAU_SK_ID_CURR'],axis=1,inplace=True)  
num_ds_end, cat_ds_end = bureau.define_params(app_train_bureau_clean_df)
app_train_bureau_clean_df.to_csv(wd+"\\Output\\application_train_bureau_clean_new.csv", index = False)


######################################TO DO ##########################################
# DAYS_EMPLOYED is highly correlated with DAYS_BIRTH hence the nan could have been derived from it. Use Autoimpute
