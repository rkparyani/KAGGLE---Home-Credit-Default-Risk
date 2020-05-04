############################# IMPORT REQUIRED LIBRARIES ###########################
import os
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
#from sklearn.impute import MissingIndicator

pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points

py.init_notebook_mode(connected=True)

import warnings
warnings.filterwarnings('ignore')

## STATISTICS

from statsmodels.stats import weightstats as stests
from scipy.stats import ttest_ind, chi2, chi2_contingency


## for ML Modelling
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC, LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict, StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error,confusion_matrix, accuracy_score, roc_curve, roc_auc_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from scipy.stats import uniform



############################# MISSING VALUES ############################
# =============================================================================
# Imputing multivariate feature na values using KNN classifier
# =============================================================================

def impute_knn_classifier (input_df,feature, n_neighbour):
    to_impute = input_df[feature].copy()
    
    if 'TARGET' in input_df:
        input_var = ['TARGET','SK_ID_CURR',feature]
    else:
        input_var = ['SK_ID_CURR',feature]
        
    df = input_df.drop(input_df[input_var],axis=1)
    num_feats_imp_df , _ = distinct_feats(df)
    df = apply_standardscalar(df, num_feats_imp_df)
    
    df = pd.get_dummies(df)
    df.columns = df.columns.map(remove_space)
    df[feature] = to_impute
    
    train_df = df[df[feature].isna()==False]
    test_df = df[df[feature].isna()==True]
    
    y_train = to_impute.dropna()
    x_train = train_df.drop(train_df[[feature]],axis=1)
    x_test = test_df.drop(test_df[[feature]],axis=1)
    
    x_train = apply_PCA(x_train,10)
    x_test = apply_PCA(x_test,10)

    neigh = KNeighborsClassifier(n_neighbors=n_neighbour)
    neigh.fit(x_train, y_train) 
    y_pred = neigh.predict(x_test)
    y_pred_1 = pd.Series(y_pred)
    y_pred_1.index = input_df[input_df[feature].isna()==True].index
    score = neigh.score(x_train,y_train)
    return y_pred_1,score



def distinct_feats(df):
    num_feats = [x for x in df.columns if df[x].dtypes!='object']
    cat_feats = [x for x in df.columns if df[x].dtypes=='object']
    return (num_feats,cat_feats)



def change_type (df,num_feats, count_threshold =5,change_type=True):
    for name in num_feats:
        _x_size = df[name].value_counts().count()
        if _x_size <= count_threshold:
            print('Seems categorical column:',name,_x_size)
            if change_type ==True:
                print(name,'changing type to Object')
                df[name] = df[name].astype(object)
        else:
            print('Seems Numerical:',name,_x_size)



def is_mean_imputable(df, feature,skew_threshold,kurt_threshold):
    if df[feature].dtypes!='object':
        if abs(df[feature].skew())<=skew_threshold and abs(df[feature].kurt()) <=kurt_threshold:
            print('For Feature',feature,'Skew:',float(df[feature].skew()),',Kurt:',float(df[feature].kurt()))
            return 1
        else: 
            return 0



# function to drop na values and impute the values with mean and median whereever required
def impute_values (df, misval_list,max_na_threshold,skew_threshold=0.1,kurt_threshold=1, imp_type=True, action=False):
    for feat in misval_list:
        if imp_type == True:
            misval_per = (df[feat].isna().sum()/df[feat].size)*100
            #print(feat,":",misval_per)
            if misval_per <= 5 and misval_per != 0.0:
                print('Delete for feature {} with {} na rows with {} missing percentage values'.format(feat,df[feat].isna().sum(),misval_per))
                if action == True:
                    df = df.drop(df[df[feat].isna()].index)
            elif misval_per >= max_na_threshold:
                print('Dropping feature {} with {} missing percentage values'.format(feat,misval_per))
                if action == True:
                    df = df.drop(df[[feat]],axis=1) 
            #df = impute_univariate(df, mean_list, median_list, mode_list, feat)
            elif is_mean_imputable(df,feat,skew_threshold,kurt_threshold) == 1:
                print('Imputing',feat,'with mean value as',df[feat].mean())    
                if action == True:
                    df[feat] = df[feat].fillna(df[feat].mean())
    return (df)


def get_missing_value_feats(df):
    _x = df.isna().sum()
    _m_val = _x[_x>0].index
    return _m_val


def missing_val_perc(missing_val_list,x_df):
    missing_values_percentage = []
    for x in x_df[missing_val_list].columns:    
        val = (x_df[x].isna().sum()/x_df[x].size)*100
        missing_values_percentage.append(val)
    df = pd.DataFrame(missing_values_percentage,index = missing_val_list)
    return df

def plot_bar(x_items, size, x_item_val,size_font=20):
    import numpy as np
    import matplotlib.pyplot as plt
    y_pos = np.arange(len(x_items))
    plt.figure(figsize=size)
    plt.bar(y_pos, x_item_val, align='center', alpha=0.5)
    plt.xticks(y_pos, x_items,rotation='vertical',fontsize=size_font)
    plt.yticks(fontsize=size_font)
    plt.ylabel('PERCENTAGE',fontsize=size_font)
    plt.title('MISSING VALUES',fontsize=size_font)
    plt.show()

def get_rowcnt_most_missing_val(df, per):
    _mis_val_cnt = round((per*len(df.columns))/100)
    _cnt = df.isna().sum(axis=1).where(lambda _x : _x>_mis_val_cnt).count()
    print(_cnt , 'values in the dataset have more than ',_mis_val_cnt, 'features as NA')
    return (_cnt)

def positive_na_cases (df, feature, target_feat_nm,replace=False):
    _ind_t = df[(df[feature].isna()==True) & (df[target_feat_nm]==1)].index
    _ind_all = df[df[feature].isna()==True].index
    _l_all = len(_ind_all)
    _l_t = len(_ind_t)
    _perc = _l_t/_l_all
    return _l_t,_l_all,_perc, _ind_t,_ind_all

def get_corr(df, feature):
    print(feature)
    _val = df[feature].sort_values(ascending=False)
    return (_val)

# compare the strings left to right and returns the number of matched characters
def match_strings(l1, l2):  
   
    str1 = max([l1,l2], key=len)
    str2 = min([l1,l2], key=len)
    
    str2 = str2.ljust(len(str1))
    
    str1 = list(str1)
    str2 = list(str2)
    
    n_str = ''
    for i in range(len(str1)-1):
        if str1[i] == str2[i]:
            n_str = ''.join([n_str,str1[i]])
    return n_str

def min_len_col(df):
    str1 = min(df.columns.tolist(),key=len)
    return len(str1)
    
    
    


#def get_all_corr(df, num_missing_values,correlation_threhold):
#    for feat in num_missing_values:
#        get_corr(df, feat) 
#        _val = corr_matrix[feat].where(lambda x:x > correlation_threhold).sort_values(ascending=False)
        
    
def corr_feats (df,missing_val_list,correlation_threhold):
    _corr_matrix = df.corr()
    _str_corr = []
    _feat_val_corr = {}
    for feature in missing_val_list:
            val = _corr_matrix[feature][_corr_matrix[feature].where(lambda x: x > correlation_threhold).where(lambda x: x < 1).notna()].index.tolist()
            if val != []:
                _feat_val_corr['feature'] = feature
                _feat_val_corr['corr_feats'] = val
                _str_corr.append(_feat_val_corr.copy())
    return _str_corr

############################# Exploratory data analysis functions ################################
# Descriptive 
def view_na_values(df, feature):
    _arc_df = df[df[feature].isna() == True].head(5)
    _arc_df = _arc_df.append(df[df[feature].isna() == False].head(5))
    return _arc_df

def get_params(imp_df, num, cat):
    param_df = pd.DataFrame()
    if len(num)>0:
        param_df['SKEW'] = imp_df[[x for x in imp_df.columns if x in num]].skew()
        param_df['KURT'] = imp_df[[x for x in imp_df.columns if x in num]].kurt()
        param_df['IS_NA'] = imp_df[[x for x in imp_df.columns if x in num]].isna().sum()
        #param_df['IS_XNA'] = [(imp_df[imp_df[x]=='XNA'][x].count()) for x in imp_df.columns if x in num]
        param_df['IS_NA_PERCENTAGE'] = (param_df['IS_NA']/imp_df.shape[0])*100
        param_df['OUTLIERS'] = np.nan
        
        for i in param_df.index.tolist():
            print(i)
            _out_l,_out_r,min,max = TurkyOutliers(imp_df,i,drop=False)
            if (len(_out_l)|len(_out_r)) > 0:
                param_df['OUTLIERS'].loc[i] = (len(_out_l) + len(_out_r))
            else:
                param_df['OUTLIERS'].loc[i] = 0
        param_df['OUTLIER_PERCENTAGE'] = (param_df['OUTLIERS']/imp_df.shape[0])*100
        
        par_cat_df = pd.DataFrame()
    if len(cat)>0:        
        # Create Categorical features information dataframe
        cat_f = [x for x in imp_df.columns if x in cat]
        par_cat_df = imp_df[cat_f].describe().T
        par_cat_df.columns = ['NOT_NA','NUMBER_CATEGORIES','MODE','MODE_COUNT']
        par_cat_df['MODE_PERCENTAGE'] = (par_cat_df['MODE_COUNT']/imp_df.shape[0])*100
        par_cat_df ['IS_NA'] = imp_df[cat_f].isna().sum()
        par_cat_df['IS_XNA'] = [(imp_df[imp_df[x]=='XNA'][x].count()) for x in imp_df.columns if x in cat]
        par_cat_df['IS_NA_PERCENTAGE'] = (par_cat_df['IS_NA']/imp_df.shape[0])*100
    return param_df , par_cat_df

def dataframeinfo(df):
    print("the shape of the dataset is:", df.shape)
    print("********************************************************************************")
    print("the datatypes are as \n",df.dtypes)
    print("********************************************************************************")
    print(df.head())
    print("********************************************************************************")
    print(df.info())
    print("********************************************************************************")
    print(df.describe())
    print("********************************************************************************")
    print("count of missing values by attribute:\n",df.isna().sum())

    
def feat_desc(df,feats,corr_df):
    feat = df[feats]
    nfeat = df[df['TARGET']==1][feats]
    if len(corr_df[corr_df['feature']==feats].corr_feats.tolist())>0:
        _val = corr_df[corr_df['feature']==feats].corr_feats.tolist()[0]
    else:
        _val='Nothing Exists'
    
    if feat.dtypes!='object':
        print('Type: NUMERIC')
        print("***** ALL ******")
        print('count:',feat.count())
        print('ISNA:',feat.isna().sum())        
        print('Skew:',feat.skew())
        print('Kurt:',feat.kurt())
        print('Mean:',feat.mean())
        print('Median:',feat.median())
        print('Mode:',feat.mode()[0])
        print('Strong_Corr:',_val)
        print('Max:',max(feat))
        print('Min:',min(feat))

        print("*****  TARGET=1 ******")
        print('count:',nfeat.count())
        print('ISNA:',nfeat.isna().sum())        
        print('Skew:',nfeat.skew())
        print('Kurt:',nfeat.kurt())
        print('Mean:',nfeat.mean())
        print('Median:',nfeat.median())
        print('Mode:',nfeat.mode()[0])
        print('Strong_Corr:',_val)
        print('Max:',max(nfeat))
        print('Min:',min(nfeat))
        
    else:
        print("***** ALL ******")
        print('Type: CATEGORICAL')
        print('Categories count:',feat.value_counts().count())
        print('Each Category count\n:',feat.value_counts())
        print('count:',feat.count())
        print('ISNA:',feat.isna().sum())
        print('Mode:',feat.mode()[0])
        print('Strong_Corr:',_val)

        print("***** TARGET=1 ******")
        print('Categories count:',nfeat.value_counts().count())
        print('Each Category count:',nfeat.value_counts())
        print('count:',nfeat.count())
        print('ISNA:',nfeat.isna().sum())
        print('Mode:',nfeat.mode()[0])
        print('Strong_Corr:',_val)
    
# Data visualizations
    
def plot_jointplot (x_var, y_var, df):
    import seaborn as sns
    sns.jointplot(x=x_var, y=y_var, data=df, kind="reg")    
        
def plot_lmplot (x_var,y_var,df,hue,col,row):
    import seaborn as sns
    sns.lmplot(x=x_var, y=y_var, hue=hue,col=col, row=row, data=df)
    
    
def hist_perc(df, df_col,bin_size,rng_st,rng_end):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    #plt.hist(x = train_df.RevolvingUtilizationOfUnsecuredLines,bins=10,range=(,1))
    plt.hist(x = df[df_col],bins=bin_size,range=(rng_st,rng_end),alpha=0.7)
    formatter = mticker.FuncFormatter(lambda v, pos: str(round((v*100/df.shape[0]),2)))
    plt.xticks(rotation=90)
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.show()    
    
def hist_compare(df, df_col,bin_size,rng_st,rng_end,y_percentage=True):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    plt.figure(figsize=(10,10))
    for i in range(len(df_col)):
        plt.hist(x = df[df_col[i]],bins=bin_size,range=(rng_st,rng_end),alpha=0.5,label=df_col[i])    
    plt.xticks(rotation=90)
    plt.legend(loc='upper right')
    if y_percentage==True:
        formatter = mticker.FuncFormatter(lambda v, pos: str(round((v*100/df.shape[0]),2)))
        plt.gca().yaxis.set_major_formatter(formatter)
    plt.show()  

def plot_bar_bins(df,feat,rng_start,range_end,interval):
    _bins = np.arange(rng_start,range_end,interval).tolist()
    _x = pd.cut(df[feat],_bins)
    _data_df = df.copy()
    _data_df[feat]=_x
    _data_df[feat].value_counts().plot.bar()
    
def default_ratio (df,target_var,minor_var,major_var,feature_list,text_offset):  
    feat_list = target_var + feature_list
    h = df.groupby(by = feat_list).count().SK_ID_CURR
    default_ratio = h.loc[minor_var]/(h.loc[major_var]+h.loc[minor_var])
    default_ratio = default_ratio.fillna(value=0)
    
    g = df.groupby(by=feature_list)
    ax = 1 - (g.count().SK_ID_CURR/df.shape[0])
    ax = ax.fillna(value=0)
    
    result = default_ratio*ax
    ax1 = result.fillna(value=0)
    
    width = .35
    axs = ax1.plot.bar()
    
    for v,c in enumerate(result):
        axs.text(v, 
                c + text_offset, 
                round(c,3), 
                color='red', 
                fontweight='bold',
                horizontalalignment='center')

    plt.title('Probability of default for categories: {}'.format(feature_list))
    plt.show()

def plot_stats(df, feature,label_rotation=False,horizontal_layout=True):
    temp = df[feature].value_counts()
    df1 = pd.DataFrame({feature: temp.index,'Number of contracts': temp.values})

    # Calculate the percentage of target=1 per category value
    cat_perc = df[[feature, 'TARGET']].groupby([feature],as_index=False).mean()
    cat_perc.sort_values(by='TARGET', ascending=False, inplace=True)
    
    if(horizontal_layout):
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))
    else:
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12,14))
    sns.set_color_codes("pastel")
    s = sns.barplot(ax=ax1, x = feature, y="Number of contracts",data=df1)
    if(label_rotation):
        s.set_xticklabels(s.get_xticklabels(),rotation=90)
    
    s = sns.barplot(ax=ax2, x = feature, y='TARGET', order=cat_perc[feature], data=cat_perc)
    if(label_rotation):
        s.set_xticklabels(s.get_xticklabels(),rotation=90)
    plt.ylabel('Percent of target with value 1 [%]', fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=10)

    plt.show();
    
def plot_distribution(df, var):
    
    i = 0
    t1 = df.loc[df['TARGET'] == 1]
    t0 = df.loc[df['TARGET'] == 0]

    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(2,2,figsize=(12,12))

    for feature in var:
        i += 1
        plt.subplot(2,2,i)
        sns.kdeplot(t1[feature], bw=0.5,label="TARGET = 1")
        sns.kdeplot(t0[feature], bw=0.5,label="TARGET = 0")
        plt.ylabel('Density plot', fontsize=12)
        plt.xlabel(feature, fontsize=12)
        locs, labels = plt.xticks()
        plt.tick_params(axis='both', which='major', labelsize=12)
    plt.show();


def OutLiersBox(df,nameOfFeature):
    
    trace0 = go.Box(
        y = df[nameOfFeature],
        name = "All Points",
        jitter = 0.3,
        pointpos = -1.8,
        boxpoints = 'all',
        marker = dict(
            color = 'rgb(7,40,89)'),
        line = dict(
            color = 'rgb(7,40,89)')
    )

    trace1 = go.Box(
        y = df[nameOfFeature],
        name = "Only Whiskers",
        boxpoints = False,
        marker = dict(
            color = 'rgb(9,56,125)'),
        line = dict(
            color = 'rgb(9,56,125)')
    )

    trace2 = go.Box(
        y = df[nameOfFeature],
        name = "Suspected Outliers",
        boxpoints = 'suspectedoutliers',
        marker = dict(
            color = 'rgb(8,81,156)',
            outliercolor = 'rgba(219, 64, 82, 0.6)',
            line = dict(
                outliercolor = 'rgba(219, 64, 82, 0.6)',
                outlierwidth = 2)),
        line = dict(
            color = 'rgb(8,81,156)')
    )

    trace3 = go.Box(
        y = df[nameOfFeature],
        name = "Whiskers and Outliers",
        boxpoints = 'outliers',
        marker = dict(
            color = 'rgb(107,174,214)'),
        line = dict(
            color = 'rgb(107,174,214)')
    )

    data = [trace0,trace1,trace2,trace3]

    layout = go.Layout(
        title = "{} Outliers".format(nameOfFeature)
    )

    fig = go.Figure(data=data,layout=layout)
    py.iplot(fig, filename = "Outliers")

# #################################### OUTLIERS #####################################
    
def OutLierDetection(df,feature1,feature2,outliers_fraction=.1):
    
    new_df = df.copy()
    rng = np.random.RandomState(42)

    # Example settings
    n_samples = new_df.shape[0]
#     outliers_fraction = 0.2 # ************************************** imp
    clusters_separation = [0]#, 1, 2]

    # define two outlier detection tools to be compared
    classifiers = {
        "One-Class SVM": svm.OneClassSVM(nu=0.95 * outliers_fraction + 0.05,
                                         kernel="rbf", gamma=0.1),
        "Robust covariance": EllipticEnvelope(contamination=outliers_fraction),
        "Isolation Forest": IsolationForest(max_samples=n_samples,
                                            contamination=outliers_fraction,
                                            random_state=rng),
        "Local Outlier Factor": LocalOutlierFactor(
            n_neighbors=35,
            contamination=outliers_fraction)}

    
    xx, yy = np.meshgrid(np.linspace(new_df[feature1].min()-new_df[feature1].min()*10/100, 
                                     new_df[feature1].max()+new_df[feature1].max()*10/100, 50),
                         np.linspace(new_df[feature2].min()-new_df[feature2].min()*10/100,
                                     new_df[feature2].max()+new_df[feature2].max()*10/100, 50))


    #n_inliers = int((1. - outliers_fraction) * n_samples)
    n_outliers = int(outliers_fraction * n_samples)
    ground_truth = np.ones(n_samples, dtype=int)
    ground_truth[-n_outliers:] = -1

    # Fit the problem with varying cluster separation
    for i, offset in enumerate(clusters_separation):
        np.random.seed(42)
        # Data generation

        X = new_df[[feature1,feature2]].values.tolist()

        # Fit the model
        plt.figure(figsize=(9, 7))
        for i, (clf_name, clf) in enumerate(classifiers.items()):
            # fit the data and tag outliers
            if clf_name == "Local Outlier Factor":
                y_pred = clf.fit_predict(X)
                scores_pred = clf.negative_outlier_factor_
            else:
                clf.fit(X)
                scores_pred = clf.decision_function(X)
                y_pred = clf.predict(X)
            threshold = stats.scoreatpercentile(scores_pred,
                                                100 * outliers_fraction)
            n_errors = (y_pred != ground_truth).sum()
            
            unique, counts = np.unique(y_pred,return_counts=True)
            print(clf_name,dict(zip(unique, counts)))
            
            new_df[feature1+'_'+feature2+clf_name] = y_pred
#             print(clf_name,y_pred) 
            # plot the levels lines and the points
            if clf_name == "Local Outlier Factor":
                # decision_function is private for LOF
                Z = clf._decision_function(np.c_[xx.ravel(), yy.ravel()])
            else:
                Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            subplot = plt.subplot(2, 2, i + 1)
            subplot.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7),
                             cmap=plt.cm.Blues_r)
            #a = subplot.contour(xx, yy, Z, levels=[threshold], linewidths=2, colors='red')
            subplot.contourf(xx, yy, Z, levels=[threshold, Z.max()],
                             colors='orange')
            #b = plt.scatter(new_df[feature1], new_df[feature2], c='white',  s=20, edgecolor='k')

            subplot.axis('tight')

            subplot.set_xlabel("%s" % (feature1))
 
            plt.ylabel(feature2)#, fontsize=18)
            plt.title("%d. %s (errors: %d)" % (i + 1, clf_name, n_errors))

        plt.subplots_adjust(0.04, 0.1, 0.96, 0.94, 0.1, 0.26)
#         plt.suptitle("Outlier detection")

    plt.show()
    return new_df
    

def TurkyOutliers(df_out,nameOfFeature,outlier_step=1.5,drop=False):

    #feature_number = 1
    #df_out = df_t
    #nameOfFeature = df_name[feature_number]
    #drop = True
    
    valueOfFeature = df_out[nameOfFeature].dropna()
    # Calculate Q1 (25th percentile of the data) for the given feature
    Q1 = np.percentile(valueOfFeature, 25.)

    # Calculate Q3 (75th percentile of the data) for the given feature
    Q3 = np.percentile(valueOfFeature, 75.)

    # Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
    step = (Q3-Q1)*outlier_step
    
    min = Q1-step
    max = Q3+step

    while ((step==0) & (len(valueOfFeature.unique()) > 1)):
        exclude_val = valueOfFeature.value_counts().index[0]
        valueOfFeature = valueOfFeature[(valueOfFeature !=exclude_val)]
        valueOfFeature = valueOfFeature.dropna()
        # Calculate Q1 (25th percentile of the data) for the given feature
        Q1 = np.percentile(valueOfFeature, 25.)
    
        # Calculate Q3 (75th percentile of the data) for the given feature
        Q3 = np.percentile(valueOfFeature, 75.)
    
        # Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
        step = (Q3-Q1)*outlier_step
        print("The loop is executing.....................",len(valueOfFeature),step)
        min = Q1-step
        max = Q3+step
        print([valueOfFeature])
        print("Printing values.......",min,max,Q1,Q3)
        
    if len(valueOfFeature)>0:         
        # print "Outlier step:", step
        print("........ Finding index")
        _outliers_left = valueOfFeature[((valueOfFeature <= min) & (valueOfFeature != 0)) ].index
        _outliers_right = valueOfFeature[((valueOfFeature >= max) & (valueOfFeature != 0)) ].index
        feature_outliers = valueOfFeature[(((valueOfFeature <= min) | (valueOfFeature >= max)) & (valueOfFeature != 0))].values
        
        print ("Number of outliers (inc duplicates): {} and outliers like: {}".format((len(_outliers_left)+len(_outliers_right)), feature_outliers[0:10]))
        # Remove the outliers, if any were specified
        if drop:
            good_data = df_out.drop(df_out.index[outliers]).reset_index(drop = True)
            print ("New dataset with removed outliers has {} samples with {} features each.".format(*good_data.shape))
            return good_data
        else: 
            print ("Nothing happens, df.shape = ",df_out.shape)
            return _outliers_left,_outliers_right,min,max
    else:
        print("No Outliers")


####################################### STRING FUNCTIONS ####################################################
def concatenate_list_data(list):
    result= ''
    for element in list:
        result += str(element)+ "_"
    return result


def get_unique_val_list(df, feature):
    _unique_items_list = df[feature].unique().tolist()
    _unique_items_list = [x for x in _unique_items_list if str(x) != 'nan']
    return _unique_items_list

def remove_space(txt):
    text = txt.split(" ")
    if "" in text:
        text.remove("")
    nt = concatenate_list_data(text)
    return nt[0:-1]

def log_transform(df,feature):
    feat_val = df[feature]
    feat_val = feat_val[(feat_val != 0)]
    feat_val = feat_val.dropna()
# High Skew and Kurt hence take log transformation
    n_feat_val = pd.Series((-np.log(abs(x))) if x < 0 else np.log(x) for x in feat_val)
    n_feat_val.index = feat_val.index
    df[feature + '_log'] =np.nan
    df[feature + '_log'].loc[n_feat_val.index]  = n_feat_val
    return df


###################################### FEATURE ENGINEERING #####################
# =============================================================================
# Function to get the aggregate features from a feature    
# =============================================================================

def get_aggregate_features_num(df,b_agg_df,feature,uid):
    if df[feature].isna().sum() > 0:
        df[[uid,feature]].dropna()
    b_agg = df.groupby(uid)
    #b_agg_df = pd.DataFrame()
    b_agg_df[feature + '_mean'] = b_agg[feature].mean()
    b_agg_df[feature + '_median'] = b_agg[feature].median()
    #b_agg_df[feature + '_mode'] = b_agg[feature].mode()
    b_agg_df[feature + '_max'] = b_agg[feature].max()
    b_agg_df[feature + '_min'] = b_agg[feature].min()
    b_agg_df[feature + '_std'] = b_agg[feature].std()
    b_agg_df[feature + '_count'] = b_agg[feature].count()
    #b_agg_df['SK_ID_CURR'] = b_agg_df.index
    return b_agg_df

###################### MODELLING ###################################################    
def get_model_performance(X_train, Y_train,models,SEED, scoring_type):
    # Test options and evaluation metric
    num_folds = 10
    scoring = scoring_type

    results = []
    names = []
    for name, model in models:
        kfold = StratifiedKFold(n_splits=num_folds, random_state=SEED)
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
        
    return names, results

class PlotBoxR(object):    
    
    def __Trace(self,nameOfFeature,value): 
    
        trace = go.Box(
            y=value,
            name = nameOfFeature,
            marker = dict(
                color = 'rgb(0, 128, 128)',
            )
        )
        return trace

    def PlotResult(self,names,results):
        
        data = []

        for i in range(len(names)):
            data.append(self.__Trace(names[i],results[i]))

        py.iplot(data)

def ScoreDataFrame(names,results,score_name):
    def floatingDecimals(f_val, dec=3):
        prc = "{:."+str(dec)+"f}" 
    
        return float(prc.format(f_val))

    scores = []
    for r in results:
        scores.append(floatingDecimals(r.mean(),4))

    scoreDataFrame = pd.DataFrame({'Model':names, score_name: scores})
    return scoreDataFrame        


# Spot-Check Algorithms
def GetBasedModel():
    basedModels = []
    basedModels.append(('LR'   , LogisticRegression()))
    basedModels.append(('LDA'  , LinearDiscriminantAnalysis()))
    basedModels.append(('KNN'  , KNeighborsClassifier()))
    basedModels.append(('CART' , DecisionTreeClassifier()))
    basedModels.append(('NB'   , GaussianNB()))
    #basedModels.append(('SVM'  , SVC(probability=True)))
    basedModels.append(('AB'   , AdaBoostClassifier()))
    basedModels.append(('GBM'  , GradientBoostingClassifier()))
    basedModels.append(('RF'   , RandomForestClassifier()))
    basedModels.append(('ET'   , ExtraTreesClassifier()))    
    return basedModels

def GetScaledModel(nameOfScaler):
    
    if nameOfScaler == 'standard':
        scaler = StandardScaler()
    elif nameOfScaler =='minmax':
        scaler = MinMaxScaler()

    pipelines = []
    pipelines.append((nameOfScaler+'LR'  , Pipeline([('Scaler', scaler),('LR'  , LogisticRegression())])))
    pipelines.append((nameOfScaler+'LDA' , Pipeline([('Scaler', scaler),('LDA' , LinearDiscriminantAnalysis())])))
    pipelines.append((nameOfScaler+'KNN' , Pipeline([('Scaler', scaler),('KNN' , KNeighborsClassifier())])))
    pipelines.append((nameOfScaler+'CART', Pipeline([('Scaler', scaler),('CART', DecisionTreeClassifier())])))
    pipelines.append((nameOfScaler+'NB'  , Pipeline([('Scaler', scaler),('NB'  , GaussianNB())])))
    #pipelines.append((nameOfScaler+'SVM' , Pipeline([('Scaler', scaler),('SVM' , SVC())])))
    pipelines.append((nameOfScaler+'AB'  , Pipeline([('Scaler', scaler),('AB'  , AdaBoostClassifier())])  ))
    pipelines.append((nameOfScaler+'GBM' , Pipeline([('Scaler', scaler),('GMB' , GradientBoostingClassifier())])  ))
    pipelines.append((nameOfScaler+'RF'  , Pipeline([('Scaler', scaler),('RF'  , RandomForestClassifier())])  ))
    pipelines.append((nameOfScaler+'ET'  , Pipeline([('Scaler', scaler),('ET'  , ExtraTreesClassifier())])  ))
    return pipelines 

def GetScaledModelwithfactorizedCW(nameOfScaler):
    if nameOfScaler == 'standard':
        scaler = StandardScaler()
    elif nameOfScaler =='minmax':
        scaler = MinMaxScaler()

    pipelines = []
    pipelines.append((nameOfScaler+'LR'+'CW'  , Pipeline([('Scaler', scaler),('LR'  , LogisticRegression(class_weight='balanced'))])))
    pipelines.append((nameOfScaler+'LDA' , Pipeline([('Scaler', scaler),('LDA' , LinearDiscriminantAnalysis())])))
    pipelines.append((nameOfScaler+'KNN' , Pipeline([('Scaler', scaler),('KNN' , KNeighborsClassifier())])))
    pipelines.append((nameOfScaler+'CART'+'CW', Pipeline([('Scaler', scaler),('CART', DecisionTreeClassifier(class_weight='balanced'))])))
    pipelines.append((nameOfScaler+'NB'  , Pipeline([('Scaler', scaler),('NB'  , GaussianNB())])))
    #pipelines.append((nameOfScaler+'SVM'+'CW' , Pipeline([('Scaler', scaler),('SVM' , SVC(class_weight='balanced'))])))
    pipelines.append((nameOfScaler+'AB'  , Pipeline([('Scaler', scaler),('AB'  , AdaBoostClassifier())])  ))
    pipelines.append((nameOfScaler+'GBM' , Pipeline([('Scaler', scaler),('GMB' , GradientBoostingClassifier())])  ))
    pipelines.append((nameOfScaler+'RF'+'CW'  , Pipeline([('Scaler', scaler),('RF'  , RandomForestClassifier(class_weight='balanced'))])  ))
    pipelines.append((nameOfScaler+'ET'+'CW'  , Pipeline([('Scaler', scaler),('ET'  , ExtraTreesClassifier(class_weight='balanced'))])  ))
    return pipelines 


def floatingDecimals(f_val, dec=3):
        prc = "{:."+str(dec)+"f}" #first cast decimal as str
    #     print(prc) #str format output is {:.3f}
        return float(prc.format(f_val))

   
def confusion_matrix_elements(act_response, predicted_response):
    cm = confusion_matrix(act_response,predicted_response)
    TN = cm[0][0]
    TP = cm[1][1]
    FP = cm[0][1]
    FN = cm[1][0]
    dic = {}
    dic['accuracy'] =  floatingDecimals((TP+TN)/(TP+FN+FP+TN),5)
    dic['precision_cl_1'] = floatingDecimals(TP/(TP+FP),5)
    dic['precision_cl_2'] = floatingDecimals(TN/(FN+TN),5)
    dic['recall_cl_1'] = floatingDecimals(TP/(TP+FN),5)
    dic['recall_cl_2'] = floatingDecimals(TN/(TN+FP),5)
    dic['sensitivity_cl_1'] = floatingDecimals(TP/(TP+FN),5)
    dic['sensitivity_cl_2'] = floatingDecimals(TN/(TN+FP),5)
    dic['specificity_cl_1'] = floatingDecimals(TN/(TN+FP),5)
    dic['specificity_cl_2'] = floatingDecimals(TP/(TP+FN),5)
    dic['MCC'] = floatingDecimals((TP*TN + FP*FN)/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**0.5),5)
    dic['AUC_cl_1'] = floatingDecimals(((TP/(TP+FN))+(1-(FP/(FP+TN))))/2,5)
    dic['AUC_cl_2'] = floatingDecimals(((TN/(TN+FP))+(1-(FN/(FN+TP))))/2,5)
    return(dic,cm)
    
def cv_score(X_train, Y_train, models, scoring_type, SEED):
    # Test options and evaluation metric
    num_folds = 10
    scoring = scoring_type
    results = []
    names = []
    cv_results = {}
    for name, model in models:
        print(name, model)
        if model != 'skip':
            
            kfold = StratifiedKFold(n_splits=num_folds, random_state=SEED)
            pred_value = cross_val_predict(model, X_train, Y_train, cv=kfold)
            cf_elements, cm = confusion_matrix_elements(Y_train,pred_value)
            print(name,':\n Confusion Matrix: \n',cm)
            for score in scoring:
                cv_results[score] = cf_elements[score]
            results.append(cv_results.copy())
            names.append(name)
        else:
            
            for score in scoring:
                cv_results[score] = np.nan
            results.append(cv_results)
            names.append(name)
    return names, results

def concat_model_score(names, results, curr_scorecard=False):
    Scrd = pd.DataFrame(results)
    Scrd.insert(0, 'model_name', names)
    if curr_scorecard is False:
        ScoreCard = Scrd
    else:
        ScoreCard = pd.concat([curr_scorecard, Scrd],axis=1)
    return ScoreCard

def cv_metrics(names, results):
    Scrd = pd.DataFrame(results)
    Scrd.insert(0, 'model_name', names)
    return Scrd


class RandomSearch(object):
    
    def __init__(self,X_train,y_train,model,hyperparameters):
        
        self.X_train = X_train
        self.y_train = y_train
        self.model = model
        self.hyperparameters = hyperparameters
        
    def RandomSearch(self):
        # Create randomized search 10-fold cross validation and 100 iterations
        cv = 10
        clf = RandomizedSearchCV(self.model,
                                 self.hyperparameters,
                                 random_state=1,
                                 n_iter=100,
                                 cv=cv,
                                 verbose=0,
                                 n_jobs=-1,
                                 )
        # Fit randomized search
        best_model = clf.fit(self.X_train, self.y_train)
        message = (best_model.best_score_, best_model.best_params_)
        print("Best: %f using %s" % (message))

        return best_model, best_model.best_params_
    
    def BestModelPredict(self,X_test):
        
        best_model,_ = self.RandomSearch()
        pred = best_model.predict(X_test)
        return pred
    

class GridSearch(object):
    
    def __init__(self,X_train,y_train,model,hyperparameters):
        
        self.X_train = X_train
        self.y_train = y_train
        self.model = model
        self.hyperparameters = hyperparameters
        
    def GridSearch(self):
        # Create randomized search 10-fold cross validation and 100 iterations
        cv = 10
        clf = GridSearchCV(self.model,
                                 self.hyperparameters,
                                 cv=cv,
                                 verbose=0,
                                 n_jobs=-1,
                                 )
        # Fit randomized search
        best_model = clf.fit(self.X_train, self.y_train)
        message = (best_model.best_score_, best_model.best_params_)
        print("Best: %f using %s" % (message))

        return best_model,best_model.best_params_
    
    def BestModelPredict(self,X_test):
        
        best_model,_ = self.GridSearch()
        pred = best_model.predict(X_test)
        return pred


def GetScaledModelwithbestparams(nameOfScaler,LR_best_params,
                                 AB_best_params,GB_best_params,RF_best_params,
                                 CART_best_params,KNN_best_params):
    if nameOfScaler == 'standard':
        scaler = StandardScaler()
    elif nameOfScaler =='minmax':
        scaler = MinMaxScaler()
    pipelines = []
    pipelines.append((nameOfScaler+'LR'+'BP'  , Pipeline([('Scaler', scaler),('LR'  , LogisticRegression(**LR_best_params))])))
    pipelines.append((nameOfScaler+'LDA'+'BP' , 'skip'))
    pipelines.append((nameOfScaler+'KNN'+'BP' , 'skip'))
    pipelines.append((nameOfScaler+'CART'+'BP', 'skip'))
    pipelines.append((nameOfScaler+'NB'+'BP'  ,'skip'))
    #pipelines.append((nameOfScaler+'SVM'+'BP' , Pipeline([('Scaler', scaler),('SVM' , SVC(class_weight='balanced'))])))
    pipelines.append((nameOfScaler+'AB'+'BP'  , Pipeline([('Scaler', scaler),('AB'  , AdaBoostClassifier(**AB_best_params))])  ))
    pipelines.append((nameOfScaler+'GBM'+'BP' , Pipeline([('Scaler', scaler),('GMB' , GradientBoostingClassifier(**GB_best_params))])  ))
    pipelines.append((nameOfScaler+'RF'+'BP'  , 'skip'))
    pipelines.append((nameOfScaler+'ET'+'BP'  , 'skip'))    
    return pipelines 


def apply_standardscalar(df, num_feats_imp_df):
    scaler = StandardScaler()
    df[num_feats_imp_df] = scaler.fit_transform(df[num_feats_imp_df].values)    
    return df

def apply_PCA(df, n_comp):
    pca = PCA(n_components=n_comp)
    principalComponents = pca.fit_transform(df.values)
    x = pd.DataFrame(data = principalComponents)    
    return x



####### STATISTICS #########

def ttest_cat_cont_var(df, feature1, feature2):
    sample_0 = np.array(df.loc[df[feature1]==0, feature2].dropna())
    sample_1 = np.array(df.loc[df[feature1]==1, feature2].dropna())

    test_stat, pval, _ = stests.ttest_ind(sample_0, sample_1,usevar='unequal')
    test_stat_eq, pval_eq, _ = stests.ttest_ind(sample_0, sample_1,usevar='pooled')

    print("Two independent samples with equal sample size and unequal variances")
    print("test_statistic :{:.3f}".format(test_stat_eq))
    print("p-value :{:.3f}".format(pval_eq))

    print("\nTwo independent samples with unequal sample size and unequal variances")
    print("test_statistic :{:.3f}".format(test_stat))
    print("p-value :{:.3f}".format(pval))

    # Decision Making
    print('-------------------------------------')
    alpha = 0.05
    if pval_eq<=alpha:
        print("Reject H0,There is a relationship between {} and {} variables".format(feature1,feature2))
    else:
        print("Retain H0,The two population  variables are not related to each other as the difference between means is not significant")
        
    return test_stat,pval_eq,test_stat, pval



def chi_square_test(df,feature1,feature2):
    contingency_table=pd.crosstab(df[feature1],df[feature2])
    print('contingency_table :-\n',contingency_table)

    #Observed Values
    Observed_Values = contingency_table.values 
    print("Observed Values :-\n",Observed_Values)
    b=chi2_contingency(contingency_table)
    print('Chi2_contingency:\n',b)

    # Expected Values
    Expected_Values = b[3]
    print("Expected Values :-\n",Expected_Values)


    no_of_rows=len(contingency_table.iloc[0:2,0])
    no_of_columns=len(contingency_table.iloc[0,0:2])
    ddof=(no_of_rows-1)*(no_of_columns-1)
    alpha = 0.05

    chi_square=sum([(o-e)**2./e for o,e in zip(Observed_Values,Expected_Values)])
    chi_square_statistic=chi_square[0]+chi_square[1]
    critical_value=chi2.ppf(q=1-alpha,df=ddof)

    #p-value
    p_value=1-chi2.cdf(x=chi_square_statistic,df=ddof)

    # Print all Parameters
    print('-------------------------------------')
    print('p-value:',p_value)
    print('Significance level: ',alpha)
    print('Degree of Freedom: ',ddof)
    print('chi-square statistic:',chi_square_statistic)
    print('critical_value:',critical_value)
    print('p-value:',p_value)


    # Decision Making
    print('-------------------------------------')
    if chi_square_statistic>=critical_value:
        print("Reject H0,There is a relationship between 2 categorical variables")
    else:
        print("Retain H0,There is no relationship between 2 categorical variables")

    if p_value<=alpha:
        print("Reject H0,There is a relationship between 2 categorical variables")
    else:
        print("Retain H0,There is no relationship between 2 categorical variables")
    return chi_square_statistic, p_value




def feature_stats(df):    
    num, cat = distinct_feats(df)
    num.remove('SK_ID_CURR')
    num.remove('TARGET')
    
    tstat = {}
    tdic = {}
    pdic = {}
    
    for feat in df.columns:
        if feat in num:
            t, p, _,_ = ttest_cat_cont_var(df,'TARGET',feat)
        else:
            t,p = chi_square_test(df,'TARGET',feat)
            
        tdic[feat]=t
        pdic[feat]=p
    
    tstat['T-statistic'] = tdic
    tstat['P-Value'] = pdic
    _xdf = pd.DataFrame(tstat)
    return _xdf
