from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

def distinct_feats(df):
    num_feats = [x for x in df.columns if df[x].dtypes!='object']
    cat_feats = [x for x in df.columns if df[x].dtypes=='object']
    return (num_feats,cat_feats)

def change_type (df,num_feats, count_threshold =10,change_type=True):
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

def is_median_imputable(df, feature,skew_threshold,kurt_threshold):
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


def get_corr(df, feature):
    print(feature)
    _val = df[feature].sort_values(ascending=False)
    return (_val)


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
    _arc_df = df[df[feature].isna() == True]
    _arc_df = _arc_df.append(df[df[feature].isna() == False].head(5))
    return _arc_df


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
    

