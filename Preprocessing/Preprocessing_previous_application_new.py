# Import user defined Libraries
import Preprocessing.Preprocessing as prep


##################  ROUTINE ####################### 
# Set working directory
wd = "D:\RITESH\Data Science\GIT WD\KAGGLE - Home-Credit-Default-Risk"
prep.os.chdir(wd)

prev_app_dataset = wd+"\\Input\\previous_application.csv"
poscash_dataset = wd+"\\Input\\POS_CASH_balance.csv"

prevapp = prep.preprocessing(prev_app_dataset, poscash_dataset)

train_clean_df = prevapp.imp_dataset(wd + '\\Output\\application_train_bureau_poscash_instpmt_ccbal_clean_new.csv')
#train_clean_df.drop(train_clean_df[['Unnamed: 0.1']],axis=1,inplace=True)

# Import working dataset and isolate for the working data as per sample
prevapp.set_dataset1()
prevapp.set_dataset2()

prevapp.isolate_dataset('SK_ID_CURR', 'SK_ID_PREV', train_clean_df)

############ POS_CASH ############
feats_ignore = ['SK_ID_PREV','SK_ID_CURR','PREV_POS_CASH_SK_ID_CURR']
prevapp.set_feats_ignore(feats_ignore)
num_ds_start, cat_ds_start = prevapp.define_dataset()


prev_agg_df = prevapp.basic_feature_extraction(prevapp.ds1_df, 'SK_ID_CURR')
prev_agg_df = prevapp._column_name_streamline(prev_agg_df, 'PREV_')

pos_cash_agg_df = prevapp.basic_feature_extraction(prevapp.ds2_df, 'SK_ID_PREV')
pos_cash_agg_df.rename(columns = {"SK_ID_PREV": "SK_PREV_ID"}, inplace = True)
pos_cash_agg_df = prevapp.merge_datasets(prevapp.ds1_df[['SK_ID_CURR','SK_ID_PREV']], pos_cash_agg_df, 'SK_ID_PREV','SK_PREV_ID', 'right')



#pos_cash_agg_df_bkp = prevapp.aggregate_datasets (pos_cash_agg_df,
#                                          'SK_ID_CURR', 
#                                          '_count', 
#                                          'PREV_POS_CASH_', 
#                                          ['PREV_POS_CASH_SK_ID_PREV', 'PREV_POS_CASH_SK_PREV_ID' ],
#                                          ['_mean','_median','_min','_max','_std']
#                                          )


pos_cash_agg_df = prevapp.aggregate_datasets_2 (pos_cash_agg_df,
                                                    'SK_ID_CURR',
                                                    ['_mean','_median','_min','_max','_std'],
                                                    ['_count'],
                                                    'mean',
                                                    'count',
                                                    {},
                                                    'PREV_POS_CASH_',
                                                    [])

############ INST_PMT ############
instpmt_dataset = wd+"\\Input\\installments_payments.csv"
prevapp.path2 = instpmt_dataset
prevapp.set_dataset2()
prevapp.isolate_dataset('SK_ID_CURR', 'SK_ID_PREV', train_clean_df)

feats_ignore = ['SK_ID_PREV','SK_ID_CURR','PREV_INST_PMT_SK_ID_CURR']
prevapp.set_feats_ignore(feats_ignore)

inst_pmt_agg_df = prevapp.basic_feature_extraction(prevapp.ds2_df, 'SK_ID_PREV')
inst_pmt_agg_df.rename(columns = {"SK_ID_PREV": "SK_PREV_ID"}, inplace = True)
inst_pmt_agg_df = prevapp.merge_datasets(prevapp.ds1_df[['SK_ID_CURR','SK_ID_PREV']], inst_pmt_agg_df, 'SK_ID_PREV','SK_PREV_ID', 'right')

#inst_pmt_agg_df = prevapp.aggregate_datasets (inst_pmt_agg_df, 
#                                          'SK_ID_CURR', 
#                                          '_count', 
#                                          'PREV_INST_PMT_', 
#                                          ['PREV_INST_PMT_SK_ID_PREV', 'PREV_INST_PMT_SK_PREV_ID' ],
#                                          ['_mean','_median','_min','_max','_std']
#                                          )


inst_pmt_agg_df = prevapp.aggregate_datasets_2 (inst_pmt_agg_df,
                                                    'SK_ID_CURR',
                                                    ['_mean','_median','_min','_max','_std'],
                                                    ['_count'],
                                                    'mean',
                                                    'count',
                                                    {},
                                                    'PREV_INST_PMT_',
                                                    [])

inst_pmt_agg_df.columns
## CCBAL ##

ccbal_dataset = wd+"\\Input\\credit_card_balance.csv"
prevapp.path2 = ccbal_dataset
prevapp.set_dataset2()
prevapp.isolate_dataset('SK_ID_CURR', 'SK_ID_PREV', train_clean_df)

feats_ignore = ['SK_ID_PREV','SK_ID_CURR','PREV_CCBAL_SK_ID_CURR']
prevapp.set_feats_ignore(feats_ignore)

ccbal_agg_df = prevapp.basic_feature_extraction(prevapp.ds2_df, 'SK_ID_PREV')
ccbal_agg_df.rename(columns = {"SK_ID_PREV": "SK_PREV_ID"}, inplace = True)
ccbal_agg_df = prevapp.merge_datasets(prevapp.ds1_df[['SK_ID_CURR','SK_ID_PREV']], ccbal_agg_df, 'SK_ID_PREV','SK_PREV_ID', 'right')

#ccbal_agg_df = prevapp.aggregate_datasets (ccbal_agg_df, 
#                                          'SK_ID_CURR', 
#                                          '_count', 
#                                          'PREV_CCBAL_', 
#                                          ['PREV_CCBAL_SK_ID_PREV', 'PREV_CCBAL_SK_PREV_ID'],
#                                          ['_mean','_median','_min','_max','_std']
#                                          )

ccbal_agg_df = prevapp.aggregate_datasets_2 (ccbal_agg_df,
                                                    'SK_ID_CURR',
                                                    ['_mean','_median','_min','_max','_std'],
                                                    ['_count'],
                                                    'mean',
                                                    'count',
                                                    {},
                                                    'PREV_CCBAL_',
                                                    [])
# Merging all datasets
prev_agg_temp_df = prevapp.merge_datasets(prev_agg_df, pos_cash_agg_df, 'PREV_SK_ID_CURR','PREV_POS_CASH_SK_ID_CURR', 'left')
prev_agg_temp_df = prevapp.merge_datasets(prev_agg_temp_df, inst_pmt_agg_df, 'PREV_SK_ID_CURR','PREV_INST_PMT_SK_ID_CURR', 'left')
prev_agg_temp_df = prevapp.merge_datasets(prev_agg_temp_df, ccbal_agg_df, 'PREV_SK_ID_CURR','PREV_CCBAL_SK_ID_CURR', 'left')
train_clean_df = prevapp.merge_datasets(train_clean_df, prev_agg_temp_df , 'SK_ID_CURR','PREV_SK_ID_CURR', 'left')


train_clean_df.drop(train_clean_df[['PREV_SK_ID_CURR','PREV_CCBAL_SK_ID_CURR','PREV_INST_PMT_SK_ID_CURR','PREV_POS_CASH_SK_ID_CURR']], axis=1, inplace=True)
train_clean_df.fillna(value=0,inplace=True)
num_ds_end, cat_ds_end = prevapp.define_params(train_clean_df)

train_clean_df.to_csv(wd+"\\Output\\application_train_clean_final.csv", index = False)


######################################TO DO ##########################################
# DAYS_EMPLOYED is highly correlated with DAYS_BIRTH hence the nan could have been derived from it. Use Autoimpute
