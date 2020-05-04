# =============================================================================
# Steps
# DS-1
# 1. Bring the dataset into final form (One hot encoding)
# 2. Remove all but one of very highly correlated feats which originated from same column
# 
# DS-2
# 1. Prepare dataset for Parametric models complying to the model principle. No two columns should be highly correlated with each other > .7
# 
# 
# DS-3-6
# 1. Perform dimensionality reduction to DS-1 after step 2
# 2. Create different datasets for different dim reduction techniques
# DS-3. PCA
# DS-4. LDA
# DS-5. Regularization
# DS-6. Feature importance using Random forest
# 
# =============================================================================
from Preprocessing.Preprocessing import preprocessing as prep
import Model.FunctionLib as f
import pandas as pd
        
class data_def(prep):
    
    # Class Variables
    dim_red_by_corr_df = None
    
    
    def __init__(self, path):
        prep.__init__(self,path)
        self.df = super().imp_dataset(path)
    
    def create_dataset_remove_corr_feats(self, target_var, filter_val,corr_threshold, feats_ignore):
        df = self.df.copy()
        x_df_dum = pd.get_dummies(df)
        x_df_Default_dum = x_df_dum[x_df_dum[target_var]==filter_val]
        
        x_df_dum.columns = x_df_dum.columns.map(f.remove_space)
        x_df_Default_dum.columns = x_df_Default_dum.columns.map(f.remove_space)
                
        _corr_threshold = corr_threshold
        get_highly_corr_feats = f.corr_feats (x_df_dum,
                                              x_df_dum.columns,
                                              _corr_threshold)
        
        get_highly_corr_feats = pd.DataFrame(get_highly_corr_feats)
        print('Highly correlated features description more than pearsonsr',
              _corr_threshold)
        
        corr_lst = []
        for i in range(len(get_highly_corr_feats.index)-1):
            
            lst_feat = get_highly_corr_feats.iloc[i,0]
            lst_corr_feat = get_highly_corr_feats.iloc[i,1]
        
            for j in range(len(lst_corr_feat)):
                _str = f.match_strings(lst_feat, lst_corr_feat[j])
                if len(_str) > f.min_len_col(df.drop(df[feats_ignore],axis=1)):
                    corr_lst.append(lst_corr_feat[j])
                    
        corr_lst = pd.DataFrame(corr_lst)[0].unique().tolist()
        print(corr_lst)
        _train_drop_cols_df = x_df_dum.copy()
        _train_drop_cols_df.drop(_train_drop_cols_df[corr_lst],axis=1,inplace=True)     
        self.dim_red_by_corr_df = _train_drop_cols_df.copy()

