#!/usr/bin/env python
# coding: utf-8

# Basic programming and data visualization libraries
import sys
import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
get_ipython().run_line_magic('matplotlib', 'inline')

import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
py.init_notebook_mode(connected=True)

import warnings
warnings.filterwarnings('ignore')


# Declaration of Varibles
SEED = 7

# Importing the dataset into memory

np.random.seed(SEED)
train_df = pd.read_csv("Alert_Training.csv")
#test_df = pd.read_csv("Alert_Testing.csv")

# ## REMOVE THE INSIGNIFICANT COLUMNS

train_df.drop(['situation_id', 'issuer_id', 'account_id', 'business_date','region','Key','Days','Alert Status','Alert Status.1','Alerted',],axis=1, inplace=True)
train_df.drop(train_df[train_df.alerted.isna()].index,inplace=True)

x = train_df.isna().sum()

x[x>=train_df.index.size*.8].index

train_df.drop(x[x>=train_df.index.size*.8].index,axis=1,inplace=True)

train_df.isna().sum()


# ## COLUMN WISE IMPUTATIONS FOR NA VALUES

feat_to_impute = ['current_qty','prior_day_qty','avg_qty','price_avail_ind','asset_class','eaddate',
'daysawayfromead','lscp','last_alert_status','de_min_calc1','de_min_calc2','flow1_constant',
'flow2_constant','de_min_breach','flow2_avg_chk','flow2_posxdays_chk','flow1_breach','flow2_breach']

# Examine current_qty and chose the imputations
ilist = train_df[train_df[feat_to_impute[0]].isna()].index

train_df.dtypes

# drop restriction_code as most of the values are null and does not add much value to dataset
train_df.iloc[ilist,0] = 0.0

# Examine prior_day_qty and chose the imputations
ilist1 = train_df[train_df[feat_to_impute[1]].isna()].index
train_df.drop(index=ilist1,inplace=True)

# Examine price_avail_ind and chose the imputations
train_df.drop(['price_avail_ind'],axis=1, inplace=True)


# Examine avg_qty and chose the imputations
ilist2 = train_df[train_df[feat_to_impute[2]].isna()].index
train_df.loc[ilist2,'avg_qty'] = 0

# Examine asset_class and chose the imputations
ilist4 = train_df[train_df[feat_to_impute[4]].isna()].index
train_df.drop(index = ilist4,inplace=True)

# Examine eaddate and chose the imputations
ilist5 = train_df[train_df[feat_to_impute[5]].isna()].index
train_df.drop(index=ilist5,inplace=True)

# Examine lscp and chose the imputations
ilist7 = train_df[train_df[feat_to_impute[7]].isna()].index
train_df.loc[ilist7,'lscp'] = 0

# Examine last_alert_status and chose the imputations
ilist8 = train_df[train_df[feat_to_impute[8]].isna()].index
train_df.drop(['last_alert_status'],axis=1, inplace=True)

# Examine de_min_calc1 and chose the imputations
ilist9 = train_df[train_df[feat_to_impute[9]].isna()].index
train_df.drop(['de_min_calc1','de_min_calc2'],axis=1, inplace=True)

# Examine flow1_constant and chose the imputations
ilist11 = train_df[train_df[feat_to_impute[11]].isna()].index
train_df.loc[ilist11,'flow1_constant'] = 0

# Examine flow1_constant and chose the imputations
ilist12 = train_df[train_df[feat_to_impute[12]].isna()].index
train_df.loc[ilist12,'flow2_constant'] = 0

# Examine de_min_breach and chose the imputations
ilist13 = train_df[train_df[feat_to_impute[13]].isna()].index
train_df.loc[ilist13,'de_min_breach'] =-1

train_df.de_min_breach.isna().sum()

# Examine flow2_avg_chk and chose the imputations
ilist14 = train_df[train_df[feat_to_impute[14]].isna()].index

train_df[(train_df.flow2_avg_chk.notna()) & (train_df.flow2_posxdays_chk.isna()) ]
train_df[(train_df.flow2_avg_chk.isna()) & (train_df.flow2_posxdays_chk.notna()) ]
train_df.loc[ilist14,'flow2_avg_chk'] = 0
train_df.loc[ilist14,'flow2_posxdays_chk'] = 0

train_df.flow2_posxdays_chk.isna().sum()
train_df[train_df.flow1_breach.isna() & train_df.flow2_breach.isna()]

train_df.flow1_breach.value_counts()
train_df.flow2_breach.value_counts()

# Examine flow1_breach and chose the imputations
ilist16 = train_df[train_df[feat_to_impute[16]].isna()].index
train_df.loc[ilist16,'flow1_breach'] = 0

# Examine flow1_breach and chose the imputations
ilist17 = train_df[train_df[feat_to_impute[17]].isna()].index
train_df.loc[ilist17,'flow2_breach'] = 0

# ## CHANGE DATA TYPES
train_df.reindex()
train_df.iloc[0:5,0:10]
train_df.iloc[0:5,10:20]
train_df.iloc[0:5,20:35]
train_df.describe()
train_df.dtypes

# Handling the date columns
train_df['eaddate']=pd.to_datetime(train_df['eaddate'])
train_df['dt_on_list']=pd.to_datetime(train_df['dt_on_list'])
train_df['dt_off_list'] = train_df.dt_off_list.replace("31/12/9999","11/04/2050")
train_df['dt_off_list']=pd.to_datetime(train_df['dt_off_list'])

# Correcting the data types 
train_df.daysawayfromead = train_df.daysawayfromead.astype("int")
train_df.de_min_threshold = train_df.de_min_threshold.astype("int")
train_df.ppmt = train_df.ppmt.astype("int")
train_df.flow1_constant = train_df.flow1_constant.astype("int")
train_df.flow2_constant = train_df.flow2_constant.astype("int")
train_df.de_min_breach = train_df.de_min_breach.astype("int")
train_df.flow2_posxdays_chk = train_df.flow2_posxdays_chk.astype("int")
train_df.flow1_breach = train_df.flow1_breach.astype("int")
train_df.flow2_breach = train_df.flow2_breach.astype("int")
train_df.alerted = train_df.alerted.astype("int")

# #### EXPLORATORY DATA ANALYSIS ####

def hist_perc(df_col,bin_size,rng_st,rng_end):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    #plt.hist(x = train_df.RevolvingUtilizationOfUnsecuredLines,bins=10,range=(,1))
    plt.hist(x = df_col,bins=bin_size,range=(rng_st,rng_end))
    formatter = mticker.FuncFormatter(lambda v, pos: str(round((v*100/train_df.shape[0]),2)))
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.show()

#sns.pairplot(train_df)
#plt.figure(figsize=(20,20))
#sns.heatmap(train_df.corr(),annot=True,cmap="YlGnBu")

# #### TRAIN AND VAL DATA 
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.font_manager

from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

## HiCS: High Contrast Subspaces for Density-Based Outlier Ranking.
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
    n_inliers = int((1. - outliers_fraction) * n_samples)
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
            a = subplot.contour(xx, yy, Z, levels=[threshold],
                                linewidths=2, colors='red')
            subplot.contourf(xx, yy, Z, levels=[threshold, Z.max()],
                             colors='orange')
            b = plt.scatter(new_df[feature1], new_df[feature2], c='white',
                     s=20, edgecolor='k')
            subplot.axis('tight')
            subplot.set_xlabel("%s" % (feature1)) 
            plt.ylabel(feature2)#, fontsize=18)
            plt.title("%d. %s (errors: %d)" % (i + 1, clf_name, n_errors))
        plt.subplots_adjust(0.04, 0.1, 0.96, 0.94, 0.1, 0.26)
#         plt.suptitle("Outlier detection")
    plt.show()
    return new_df

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


# Remove outliers
def TurkyOutliers(df_out,nameOfFeature, multi_factor, drop=False):
    #feature_number = 1
    #df_out = df_t
    #nameOfFeature = df_name[feature_number]
    #drop = True    
    valueOfFeature = df_out[nameOfFeature]
    # Calculate Q1 (25th percentile of the data) for the given feature
    Q1 = np.percentile(valueOfFeature, 25.)
    # Calculate Q3 (75th percentile of the data) for the given feature
    Q3 = np.percentile(valueOfFeature, 75.)
    # Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
    step = (Q3-Q1)*multi_factor
    # print "Outlier step:", step
    outliers = valueOfFeature[~((valueOfFeature >= Q1 - step) & (valueOfFeature <= Q3 + step))].index.tolist()
    feature_outliers = valueOfFeature[~((valueOfFeature >= Q1 - step) & (valueOfFeature <= Q3 + step))].values
    # df[~((df[nameOfFeature] >= Q1 - step) & (df[nameOfFeature] <= Q3 + step))]
    # Remove the outliers, if any were specified
    print ("Number of outliers (inc duplicates): {} and outliers: {}".format(len(outliers), feature_outliers))
    if drop:
        good_data = df_out.drop(df_out.index[outliers]).reset_index(drop = True)
        print ("New dataset with removed outliers has {} samples with {} features each.".format(*good_data.shape))
        return good_data
    else: 
        print ("Nothing happens, df.shape = ",df_out.shape)
        return df_out

def get_numerical_feats(df):
    _num_feats = []
    for i in df.columns:
        if df[i].dtypes!='object':
            _num_feats.append(i)
    return _num_feats    

num_feats = get_numerical_feats(train_df)
feature_number  = 0
print(num_feats[feature_number])
#OutLiersBox(train_df,num_feats[feature_number])

df_clean = TurkyOutliers(train_df,num_feats[feature_number], 1.5, False)
#OutLiersBox(train_df,num_feats[feature_number])

feature_number  = 1
print(num_feats[feature_number])
#OutLiersBox(train_df,num_feats[feature_number])

df_clean = TurkyOutliers(train_df,num_feats[feature_number], 2.5, False)
#OutLiersBox(train_df,num_feats[feature_number])

plt.figure(figsize=(5,5))
train_df[num_feats[feature_number]].plot.kde()

hist_perc(train_df[num_feats[feature_number]],5,-3,3)

train_df[train_df['min_qty']==0].alerted.value_counts()

# ## ENCODING
a = [x for x in train_df.columns if train_df[x].dtypes=='object']
train_df = pd.get_dummies(train_df,columns = a )

a = [x for x in train_df.columns if train_df[x].dtypes=='datetime64[ns]']
train_df.drop(a,axis=1,inplace=True)

from sklearn.model_selection import train_test_split

Y = train_df.alerted
X = train_df.drop(['alerted'],axis=1)
X_train,X_val, Y_train, Y_val = train_test_split(X,Y,test_size = 0.2,random_state = SEED)
print(X_train.shape,X_val.shape, Y_train.shape, Y_val.shape)

# Check Class imbalance
Y_train.value_counts(normalize=True).plot.bar()

Y_val.value_counts(normalize=True).plot.bar()

Y_val.value_counts()

# ##### BASE MODEL PREDICTIONS

from pandas import set_option
from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report,precision_score,recall_score,precision_recall_curve,make_scorer,f1_score,confusion_matrix


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

def floatingDecimals(f_val, dec=3):
        prc = "{:."+str(dec)+"f}" #first cast decimal as str
    #     print(prc) #str format output is {:.3f}
        return float(prc.format(f_val))

   
def confusion_matrix_elements(act_response, predicted_response):
    cm = confusion_matrix(act_response,predicted_response)
    TN = cm[0][0]
    TP = cm[1][1]
    FP = cm[1][0]
    FN = cm[0][1]
    dic = {}
    dic['accuracy_cl_1'] =  floatingDecimals((TP+FN)/(TP+FN+FP+TN),5)
    dic['accuracy_cl_2'] =  floatingDecimals((FP+TN)/(TP+FN+FP+TN),5)
    dic['precision_cl_1'] = floatingDecimals(TP/(TP+FP),5)
    dic['precision_cl_2'] = floatingDecimals(TN/(FN+TN),5)
    dic['recall_cl_1'] = floatingDecimals(TP/(TP+FN),5)
    dic['recall_cl_2'] = floatingDecimals(TN/(TN+FP),5)
    dic['sensitivity_cl_1'] = floatingDecimals(TP/(TP+FN),5)
    dic['sensitivity_cl_2'] = floatingDecimals(TN/(TN+FP),5)
    dic['specificity_cl_1'] = floatingDecimals(TN/(TN+FP),5)
    dic['specificity_cl_2'] = floatingDecimals(TP/(TP+FN),5)
    return(dic)

def BasedLine2(X_train, Y_train,models,scoring_type):
    # Test options and evaluation metric
    num_folds = 10
    scoring = scoring_type
    results = []
    names = []
    cv_results = {}
    #std = {}
    for name, model in models:
        kfold = StratifiedKFold(n_splits=num_folds, random_state=SEED)
        for score in scoring:
            print(score)
            x = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=score)
            cv_results[score] = x.mean() 
            #std[score] = x.std()
        results.append(cv_results.copy())
        names.append(name)
        #msg = "%s: %f (%f)" % (name, cv_results, std)
        #msg = "{} has mean values of {} and std dev of {}".format(name, results, std)
        #print(msg)
    return names, results


def cv_score(X_train, Y_train,models,scoring_type):
    # Test options and evaluation metric
    num_folds = 10
    scoring = scoring_type
    results = []
    names = []
    cv_results = {}
    #std = {}
    for name, model in models:
        kfold = StratifiedKFold(n_splits=num_folds, random_state=SEED)
        pred_value = cross_val_predict(model, X_train, Y_train, cv=kfold)
        for score in scoring:
            print(score)
            cf_elements = confusion_matrix_elements(Y_train,pred_value)
            cv_results[score] = cf_elements[score]
            # cv_results[score] = x.mean() 
            # std[score] = x.std()
        results.append(cv_results.copy())
        names.append(name)
        #msg = "%s: %f (%f)" % (name, cv_results, std)
        #msg = "{} has mean values of {} and std dev of {}".format(name, results, std)
        #print(msg)
    return names, results

models = GetBasedModel()
names, results = cv_score(X_train,Y_train,models,['precision_cl_1',
                                                  'precision_cl_2',
                                                  'recall_cl_1',
                                                  'recall_cl_2',
                                                  'specificity_cl_1',
                                                  'specificity_cl_2'])

def concat_model_score(names, results, curr_scorecard=False):
    Scrd = pd.DataFrame(results)
    Scrd.insert(0, 'model_name', names)
    if curr_scorecard is False:
        ScoreCard = Scrd
    else:
        ScoreCard = pd.concat([curr_scorecard, Scrd],axis=1)
    return ScoreCard

ScoreCard = concat_model_score(names, results)

#PlotBoxR().PlotResult(names,results)

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

def GetScaledModel(nameOfScaler):
    if nameOfScaler == 'standard':
        scaler = StandardScaler()
    elif nameOfScaler =='minmax':
        scaler = MinMaxScaler()
    pipelines = []
    pipelines.append((nameOfScaler+'LR'  , Pipeline([('Scaler', scaler),('LR'  , LogisticRegression(class_weight='balanced'))])))
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
    

models = GetScaledModel('standard')
names, results = cv_score(X_train,Y_train,models,['precision_cl_1',
                                                  'precision_cl_2',
                                                  'recall_cl_1',
                                                  'recall_cl_2',
                                                  'specificity_cl_1',
                                                  'specificity_cl_2'])

ScoreCard = concat_model_score(names, results, ScoreCard)

models = GetScaledModel('minmax')
names, results = cv_score(X_train,Y_train,models,['precision_cl_1',
                                                  'precision_cl_2',
                                                  'recall_cl_1',
                                                  'recall_cl_2',
                                                  'specificity_cl_1',
                                                  'specificity_cl_2'])

ScoreCard = concat_model_score(names, results, ScoreCard)

# Observations - Decision Trees are overfitting along with other Tree based methods such as RF, Boosting methods
# Seems that there are not much observations of the minority class to develop a good model
# LDA looks better of all of them as Precision vs recall for both individual classes is highest
##### The aboove method has various problems such as  - 
# Problem 1 - High Class imbalance
# Problem 2 - The dataset contains outliers to a great deal which will impact 
#             the parametric model results hence need to deal with outliers


# Solution 1 - Perform SMOTE to improve the lesser class instances to enable model learn better.

# Solution 2
# We cannot simply remove outliers as they might be fraudulent transations, however since we need to 
# remove the class imbalance, we can try removing the outliers from the majority class
# Try new features in case of highly skewed data such as quantity fields in the begining. Use Z-score 
# determine outliers.

































# Anamoly detection for Frauds

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

classifiers = {
    "Isolation Forest":IsolationForest(n_estimators=100, max_samples=len(X), 
                                       contamination=0.2,random_state=SEED, verbose=0),
    "Local Outlier Factor":LocalOutlierFactor(n_neighbors=10, algorithm='auto', 
                                              leaf_size=30, metric='minkowski',
                                              p=2, metric_params=None, contamination=.2),
    "Support Vector Machine":OneClassSVM(kernel='rbf', degree=3, gamma=0.1,nu=0.05, 
                                         max_iter=-1, random_state=SEED)   
}

# Using X_train dataset
for i, (clf_name,clf) in enumerate(classifiers.items()):
    #Fit the data and tag outliers
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X_train)
        scores_prediction = clf.negative_outlier_factor_
    elif clf_name == "Support Vector Machine":
        clf.fit(X_train)
        y_pred = clf.predict(X_train)
    else:    
        clf.fit(X_train)
        scores_prediction = clf.decision_function(X_train)
        y_pred = clf.predict(X_train)
    #Reshape the prediction values to 0 for Valid transactions , 1 for Fraud transactions
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    n_errors = (y_pred != Y_train).sum()
    # Run Classification Metrics
    print("{}: {}".format(clf_name,n_errors))
    print("Accuracy Score :")
    print(accuracy_score(Y_train,y_pred))
    print("Classification Report :")
    print(classification_report(Y_train,y_pred))

# Interpretation - T X_train_bal dataset has less accuracy with same models applied 
# as they have less number of true observations to accurately predict the false observations 

train_dataset = pd.concat([X_train,Y_train],axis=1)

