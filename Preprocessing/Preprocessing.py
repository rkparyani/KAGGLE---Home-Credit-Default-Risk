####################### LIBRARIES ################################
# Data and visualization
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#from sklearn.impute import MissingIndicator
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew #for some statistics
from sklearn.neighbors import KNeighborsClassifier

import Model.FunctionLib as f
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points



class preprocessing():
# Class Variables
    ds1_df = None
    ds2_df = None
    x_df = None
    imp_df = None
    lst = None
    
# Init Variables        
    def __init__(self, path1, path2=None, frac = None,rs = None):
        self.path1 = path1
        self.path2 = path2
        self.frac = frac
        self.rs = rs
        
    def set_feats_ignore(self, x):
        self.lst = x    

    def imp_dataset(self,path):
        train_df = pd.read_csv(path)
        train_df.drop(train_df.columns[train_df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
        if self.frac != None:
            train_df = train_df.sample(frac=self.frac, random_state=self.rs).reset_index(drop=True)
        return train_df

    def set_dataset1(self):
        _s_df = self.imp_dataset(self.path1)
        self.ds1_df = _s_df
    
    def set_dataset2(self):
        _s_df = self.imp_dataset(self.path2)
        self.ds2_df = _s_df    
    
    def isolate_dataset(self, ds1_df_id, ds2_df_id, train_clean_df):
        
        self.ds1_df = self.ds1_df.loc[self.ds1_df[self.ds1_df[ds1_df_id].isin(train_clean_df[ds1_df_id])].index]
        self.ds1_df.reset_index(drop=True)
        
        if ds2_df_id != None:
            self.ds2_df= self.ds2_df.loc[self.ds2_df[self.ds2_df[ds2_df_id].isin(self.ds1_df[ds2_df_id])].index]
            self.ds2_df.reset_index(drop=True)
            
    def seperate_cat_num_var(self, df):
        # Seperate the categorical and numerical features
        num_feats,cat_feats = f.distinct_feats(df)
        print(len(num_feats),len(cat_feats))
        for x in range(len(self.lst)):
            if self.lst[x] in num_feats:
                num_feats.remove(self.lst[x])    
        return num_feats,cat_feats
    
    def define_params(self,df):
        num_feats, cat_feats = self.seperate_cat_num_var(df)
        par_num_df_start, par_cat_df_start = f.get_params(df, num_feats, 
                                                  cat_feats)
        return par_num_df_start, par_cat_df_start

    def define_dataset(self, df=None, ch_type=False, cnt_threshold=2):                
        # Observe the features with missing values
        if df == None:            
            df = self.ds1_df
        f.get_missing_value_feats(df)
        
        # Seperate the categorical and numerical features
        num_feats,cat_feats = self.seperate_cat_num_var(df)
            
        # Change the datatype of categorical and numerical values
        if ch_type==True:
            f.change_type(df, num_feats, count_threshold=cnt_threshold)
        
        # Seperate the categorical and numerical features
        par_num_df_start, par_cat_df_start = self.define_params(df)
        stats_df = f.feature_stats(df)

        par_num_df_start = par_num_df_start.join(stats_df,how='left')
        par_cat_df_start = par_cat_df_start.join(stats_df,how='left')

        return par_num_df_start, par_cat_df_start

    
    def missing_value_treatment(self,min_threshold):
        # Identify na values exist and add them to a list
        
        missing_value_feats = f.get_missing_value_feats(self.ds1_df)
        print(missing_value_feats)
        # Calculate Missing Value percentage and Visualize
        missing_values_perc_df = f.missing_val_perc(missing_value_feats,self.ds1_df)
        val = missing_values_perc_df[0].sort_values(ascending=False)
        f.plot_bar(val.index,(50,10),val)
        
        # Check direct imputations such as remove the records for attributes which contain less than 5% of null values or remove
        # attributes which contain more than 65% of null values.
        self.ds1_df = f.impute_values(self.ds1_df,missing_value_feats,min_threshold,action=True)
        self.ds1_df.reset_index(drop=True)
        
        # How row in dataframe having more than x% NaN values
        na_row_cnt = f.get_rowcnt_most_missing_val(self.ds1_df,30)
        print('No of rows having more than 30% NA Values', na_row_cnt)
        
        # Identify na values exist and add them to a list
        missing_value_feats = f.get_missing_value_feats(self.ds1_df)
        print(missing_value_feats)


    def outlier_treatment(self, normalized_feats):
        # Find the num and cat feats for imp_df   

        num_feats_imp_df, cat_feats_imp_df = self.seperate_cat_num_var(self.ds1_df)
        other_feats = [x for x in num_feats_imp_df if x not in normalized_feats]
        
        # Anamolies and data correction.
        # DAYS_EMPLOYED has abnormal value '365243' which would be changed to nan for imputation at a later stage
        feature = 'DAYS_EMPLOYED'
        self.ds1_df[feature].loc[self.ds1_df[self.ds1_df[feature]==365243].index] = np.nan
        
        # XNA values exist in ORGANIZATION_TYPE feature, replacing it by np.NaN to be imputed. 
        self.ds1_df['ORGANIZATION_TYPE'].replace("XNA",np.nan,inplace=True)
        
        # Log transformation of all numerical non normalized highly skewed values to remove outliers

        
        for feature in other_feats:
            print('log_transform',feature)
            self.ds1_df = f.log_transform(self.ds1_df,feature)
            self.ds1_df.drop(self.ds1_df[[feature]],axis=1,inplace=True)
        
        #normalized_num_feats_imp_df = [x for x in normalized_feats if x in num_feats_imp_df]
        num_feats_imp_df,cat_feats_imp_df = self.seperate_cat_num_var(self.ds1_df)
        
        for i in num_feats_imp_df:
            print(i)
            out_l,out_r,min,max = f.TurkyOutliers(self.ds1_df,i,drop=False)
            if (len(out_l)|len(out_r)) > 0:
                self.ds1_df[i].loc[out_l] = round(min,3)
                self.ds1_df[i].loc[out_r] = round(max,3)
        
                

    def missing_value_imputations(self):    
        #################################### MISSING VALUES #############################
        # Since the numerical univariate distribution are symmetrical now with no difference 
        # between median and mean. Lets impute all the numerical missing values with mean
        # Record missing values for further validations:
        #indicator = MissingIndicator(missing_values=np.nan)
        #mask_missing_values_only = indicator.fit_transform(self.ds1_df)
        #mask_missing_values_only.shape
        
        num_feats_imp_df, cat_feats_imp_df = self.seperate_cat_num_var(self.ds1_df)
        # Num missing values imputations
        self.ds1_df[num_feats_imp_df] = self.ds1_df[num_feats_imp_df].fillna(value = self.ds1_df[num_feats_imp_df].mean())
        
        # Left missing values are categorical.
        missing_feats_cat = f.get_missing_value_feats(self.ds1_df)
        
        par_num_df, par_cat_df = f.get_params(self.ds1_df, num_feats_imp_df, 
                                              cat_feats_imp_df)
        # Categorical values where mode frequency is more than 80% - Impute na with Mode
        # If not then use the KNN model to impute the values
        
        mode_threshold = 80
        for feature in missing_feats_cat:
            if par_cat_df.loc[feature]['MODE_PERCENTAGE'] > mode_threshold:
                self.ds1_df[feature].fillna(value= par_cat_df.loc[feature]['MODE'],inplace=True)
                print("Method : MODE , Feature : {} , Mode_Percentage : {}".format(feature, par_cat_df.loc[feature]['MODE_PERCENTAGE']))
                
            else: 
                imp_list, score = f.impute_knn_classifier(self.ds1_df, feature, 5)
                self.ds1_df[feature].fillna(value = imp_list,inplace=True)
                print("Method : KNN , Feature : {} , Imputation Accuracy Score : {}".format(feature,score))
        return par_num_df, par_cat_df                
    
    def _num_feature_extraction(self, num_feats_b, b_df, b_agg_df, p_id):        
        
        for feature in num_feats_b:  
            print(feature)
            b_agg_df = f.get_aggregate_features_num(b_df,b_agg_df, feature, p_id)
        #    na_ind = b_agg_df[(b_agg_df[feature + '_std'].isna()==True) & 
        #                  ((b_agg_df[feature+'_mean'])==(b_agg_df[feature+'_median']))].index
        #
        #    b_agg_df.loc[na_ind][feature+'_std'].fillna(0)
        #    b_agg_df.loc[na_ind][feature'_std'].isna().sum()
            b_agg_df[feature+'_std'] = np.where((b_agg_df[feature+'_std'].isna()==True) & 
                    ((b_agg_df[feature+'_mean'])==(b_agg_df[feature+'_median'])), 
                 0, 
                 b_agg_df[feature+'_std'])
        b_agg_df.insert(0, p_id, b_agg_df.index)
        b_agg_df.reset_index(drop=True,inplace=True)
        return b_agg_df
    

    def _cat_feature_extraction(self, cat_feats_b, b_df, b_agg_df, p_id):
        
        for feature in cat_feats_b:   
            b_agg_cat = b_df.groupby(p_id)[feature].value_counts()
            _unique_items_list = f.get_unique_val_list(b_df, feature)
            
            for i in _unique_items_list:
                b_agg_df[feature+'_'+f'{i}'+'_count'] = b_agg_cat.xs(key=i,level=1)
                b_agg_df[feature+'_'+f'{i}'+'_count'].fillna(value=0,inplace=True)


    def basic_feature_extraction(self, df, p_id):
        b_agg_df = pd.DataFrame()
        num_feats_b, cat_feats_b = self.seperate_cat_num_var(df)

        self._num_feature_extraction(num_feats_b, df, b_agg_df, p_id)
        self._cat_feature_extraction(cat_feats_b, df, b_agg_df, p_id)
        b_agg_df.fillna(value=0,inplace=True)
        return b_agg_df
    
    @staticmethod
    def merge_datasets(ds1, ds2, left_on, right_on,join_type):
        # Merging with b_agg_df after aggregating at SK_ID_CURR level
        m_df = pd.merge(ds1, ds2, left_on=left_on, right_on = right_on, how = join_type)
        return m_df
    
    def _column_name_streamline(self, b_agg_df, prefix):
        b_agg_df.columns = b_agg_df.columns.map(f.remove_space)
        b_agg_df = b_agg_df.add_prefix(prefix)
        return b_agg_df
        
    def _aggregate_count(self, b_agg_df, group_on, filter_on):
        grp_b = self.ds1_df.groupby(group_on)
        for i in (self.ds1_df.filter(like=filter_on).columns):
            b_agg_df[i] = grp_b[i].sum()
        return b_agg_df
    
    def _aggregate_num(self,b_agg_df, colnm_lst, group_on):
        grp_b = self.ds1_df.groupby(group_on)
        for val in colnm_lst:
            for i in (self.ds1_df.filter(like=val).columns):
                if i not in self.lst:
                    b_agg_df[i] = grp_b[i].mean()
        return b_agg_df
    
    
    def _create_file_for_agg(self, df, colnm_lst, treatment, n_dic):
        for val in colnm_lst:
            for i in (df.filter(like=val).columns):
                if i not in self.lst:
                    n_dic[i] = treatment
        return n_dic

            
    def aggregate_datasets(self, b_agg_df, group_on, filter_on, prefix, drop_cols, colnm_lst):
        #self.ds1_df = self._merge_datasets(self.ds1_df, b_agg_df, left_on, right_on)
 
        if filter_on !=False:
            b_agg_df = self._aggregate_count(b_agg_df, group_on, filter_on)
            
        if colnm_lst != False:
            b_agg_df = self._aggregate_num(b_agg_df, colnm_lst, group_on)
            
        b_agg_df = self._column_name_streamline(b_agg_df, prefix)
        
        if drop_cols != False:
            b_agg_df.drop(b_agg_df[drop_cols],axis=1,inplace=True) 
            
        b_agg_df.fillna(value=0,inplace=True)
        
        return b_agg_df
    
    def aggregate_datasets_2(self, df, group_on, colnm_lst1, colnm_lst2, treatment1, treatment2, n_dic, prefix, drop_cols):
        
        n_dic = self._create_file_for_agg(df,colnm_lst1, treatment1, {})
        n_dic = self._create_file_for_agg(df, colnm_lst2, treatment2, n_dic)
        
        n_df = df.groupby(group_on).agg(n_dic)
        n_df.insert(0, group_on, n_df.index)
        n_df.reset_index(drop=True,inplace=True)
        
        if drop_cols != False:
            n_df.drop(n_df[drop_cols],axis=1,inplace=True) 
            
        n_df = self._column_name_streamline(n_df, prefix)
        
        n_df.fillna(value=0,inplace=True)
        return n_df
        

    
          