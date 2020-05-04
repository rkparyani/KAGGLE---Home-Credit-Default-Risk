####################### LIBRARIES ################################
# Data and visualization
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew #for some statistics

pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points
get_ipython().run_line_magic('matplotlib', 'inline')

## for ML Modelling
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC, LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import mean_squared_error,confusion_matrix, accuracy_score, roc_curve, roc_auc_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier

#import statsmodels as sm

###################  EXECUTE OTHER DATA PREPROCESSING STEPS ####################### 
# Variables 
wd = "D:\RITESH\Data Science\GIT WD\KAGGLE - Home-Credit-Default-Risk"

# Set working directory
os.chdir(wd)


# Import User Libraries and Data Pre-Proccessing Scripts
import Model.FunctionLib as f
#import Model.Preprocessing_app_train 
#import Model.Preprocessing_app_test
#import Model.Preprocessing_bureau

################################ IMPORT LATEST DATASET ################################
train_df = pd.read_csv(wd+"\\Output\\application_train_bureau_clean.csv")
train_df.drop(train_df.filter(like='Unnamed').columns,axis=1,inplace=True)

# Change the datatype of categorical and numerical values (NOT REQUIRED)
#f.change_type(train_df,num_feats,count_threshold=5)

# Seperate the categorical and numerical features
num_feats,cat_feats = f.distinct_feats(train_df)
print(len(num_feats),len(cat_feats))
num_feats.remove('TARGET')
num_feats.remove('SK_ID_CURR')

# Get the list of attributes and their properties to start
par_num_df_start, par_cat_df_start = f.get_params(train_df, num_feats, 
                                      cat_feats)


############# FEATURE CORRELATIONS ########## 
 # Code Block to find the correlated features for various features including featues including each category correlations
 # This can be used to derive/impute na values when the correlations are strong with other features using sklearn.Impute Iterativeimputer        
 # Not using this approach for now as there are no strong correlations with missing value columns
 
x_df_dum = pd.get_dummies(train_df)
x_df_Default_dum = x_df_dum[x_df_dum['TARGET']==1]

x_df_dum.columns = x_df_dum.columns.map(f.remove_space)
x_df_Default_dum.columns = x_df_Default_dum.columns.map(f.remove_space)

 # General correlations wrt Correlations in case of default.
x_corr_default = x_df_Default_dum.corr()
x_corr = x_df_dum.corr()

corr_threshold = 0.7
get_highly_corr_feats = f.corr_feats (x_df_dum,x_df_dum.columns,corr_threshold)
get_highly_corr_feats = pd.DataFrame(get_highly_corr_feats)
print('Highly correlated features description more than pearsonsr',corr_threshold)
get_highly_corr_feats


### Create Base Model with default hyperparameters and all features #####
SEED = 7
X =  x_df_dum.drop(train_df[['TARGET','SK_ID_CURR']],axis=1)
Y = x_df_dum[['TARGET']]

X_train, X_test, y_train, y_test =train_test_split(X,Y,
                                                   test_size=0.25,
                                                   random_state=0,
                                                   stratify=x_df_dum['TARGET'])


# Run the baseline models on the unbalanced dataset 
models = f.GetBasedModel()
names,results = f.get_model_performance(X_train, y_train,models,SEED, 'f1_weighted')
f.PlotBoxR().PlotResult(names,results)

basedLineF1Score = f.ScoreDataFrame(names,results,'baseline_f1_Score')


models = f.GetBasedModel()
names,results = f.get_model_performance(X_train, y_train,models,SEED,'accuracy')
f.PlotBoxR().PlotResult(names,results)

basedLineAccuracyScore = f.ScoreDataFrame(names,results,'baseline_accuracy')

# Record Scores
ScoreCard = pd.concat([basedLineAccuracyScore,
                       basedLineF1Score], axis=1)


# Scaled Model on top of baseline
# Standard Scalar
models = f.GetScaledModel('standard')
names,results = f.get_model_performance(X_train, y_train,models,SEED,'f1_weighted')
#f.PlotBoxR().PlotResult(names,results)

# Record Scores
scaledScoreStandard = f.ScoreDataFrame(names,results,'standard_f1_score')
ScoreCard = pd.concat([ScoreCard,
                           scaledScoreStandard], axis=1)


# Minmax scalar
models = f.GetScaledModel('minmax')
names,results = f.get_model_performance(X_train, y_train,models,SEED,'f1_weighted')
f.PlotBoxR().PlotResult(names,results)

# Record Scores
scaledScoreMinMax = f.ScoreDataFrame(names,results,'minmax_f1_score')
ScoreCard = pd.concat([ScoreCard,
                          scaledScoreMinMax], axis=1)


# Scaled Model on top of baseline with models adjusted for Class Weight
# Standard Scalar
models = f.GetScaledModelwithfactorizedCW('standard')
names,results = f.get_model_performance(X_train, y_train,models,SEED,'f1_weighted')
f.PlotBoxR().PlotResult(names,results)

# Record Scores
scaledScoreStandard = f.ScoreDataFrame(names,results,'standard_f1_score')
ScoreCard = pd.concat([ScoreCard,
                           scaledScoreStandard], axis=1)


# Minmax scalar
models = f.GetScaledModelwithfactorizedCW('minmax')
names,results = f.get_model_performance(X_train, y_train,models,SEED,'f1_weighted')
f.PlotBoxR().PlotResult(names,results)

# Record Scores
scaledScoreMinMax = f.ScoreDataFrame(names,results,'minmax_f1_score')
ScoreCard = pd.concat([ScoreCard,
                          scaledScoreMinMax], axis=1)

# Since this is an imbalance class problem: The Models scaled to classweight and then looking into 
# the metrics for minority class along with the majority class in TARGET

models = f.GetScaledModelwithfactorizedCW('standard')
names, results = f.cv_score(X_train,y_train,models,['accuracy',
                                                    'precision_cl_1',
                                                    'precision_cl_2',
                                                    'recall_cl_1',
                                                    'recall_cl_2',
                                                    'specificity_cl_1',
                                                    'specificity_cl_2'])

ScoreCard = f.concat_model_score(names, results, ScoreCard)

# Finding best parameters for the most performing models #########
from scipy.stats import uniform
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# Logistic Regression model
penalty = ['l1', 'l2']

# Create regularization hyperparameter distribution using uniform distribution
C = uniform(loc=0, scale=4)

# Create hyperparameter options
hyperparameters = dict(C=C, penalty=penalty)

model = LogisticRegression()
LR_RandSearch = f.RandomSearch(X_train,y_train,model,hyperparameters)
LR_best_model,LR_best_params = LR_RandSearch.RandomSearch()
Prediction_LR = LR_RandSearch.BestModelPredict(X_train)

# ADABOOST CLASSIFIER

learning_rate_value = [.01,.05,.1,.5,1]
n_estimators_value = [50,100,150,200,250,300]

param_grid = dict(learning_rate=learning_rate_value, n_estimators=n_estimators_value)

model_Ad = AdaBoostClassifier()
Ad_RandSearch = f.RandomSearch(X_train,y_train,model_Ad,param_grid)
Ad_best_model,Ad_best_params = Ad_RandSearch.RandomSearch()
Prediction_Ad = Ad_RandSearch.BestModelPredict(X_train)

# Gradient Boosting Classifier

learning_rate_value = [.01,.05,.1,.5,1]
n_estimators_value = [50,100,150,200,250,300]

param_grid = dict(learning_rate=learning_rate_value, n_estimators=n_estimators_value)

model_GB = GradientBoostingClassifier()
GB_RandSearch = f.RandomSearch(X_train,y_train,model_GB,param_grid)
GB_best_model,GB_best_params = GB_RandSearch.RandomSearch()
Prediction_GB = GB_RandSearch.BestModelPredict(X_train)

# Random Forest Classifier
n_estimators_value = [50,100,150,200,250,300]
criterion_val = ["gini", "entropy"]

param_grid = dict(criterion=criterion_val, n_estimators=n_estimators_value)

model_RF = RandomForestClassifier()
RF_RandSearch = f.RandomSearch(X_train,y_train,model_RF,param_grid)
RF_best_model,RF_best_params = RF_RandSearch.RandomSearch()
Prediction_RF = RF_RandSearch.BestModelPredict(X_train)

# Decision Tree Classifier
criterion_val = ["gini", "entropy"]
max_depth_tree = np.arange(2,20,1)
min_samples_split_1 = np.arange(2,10,1)
min_samples_leaf_1 = np.arange(1,5,1)
#min_weight_fraction_leaf_1 = 0.1

param_grid = dict(criterion=criterion_val, 
                  max_depth = max_depth_tree,
                  min_samples_split=min_samples_split_1,
                  min_samples_leaf = min_samples_leaf_1
                  )

model_CART = RandomForestClassifier()
CART_RandSearch = f.RandomSearch(X_train,y_train,model_CART,param_grid)
CART_best_model,CART_best_params = CART_RandSearch.RandomSearch()
Prediction_CART = CART_RandSearch.BestModelPredict(X_train)

# KNN best model
n_neighbors_1  = np.arange(5,75,5)
param_grid = dict(n_neighbors = n_neighbors_1)

model_KNN = KNeighborsClassifier()
KNN_RandSearch = f.RandomSearch(X_train,y_train,model_KNN,param_grid)
KNN_best_model,KNN_best_params = KNN_RandSearch.RandomSearch()

# Run the best model pipeline
models = f.GetScaledModelwithbestparams('standard',
                                        LR_best_params,
                                        Ad_best_params,
                                        GB_best_params,
                                        RF_best_params,
                                        CART_best_params,
                                        KNN_best_params)

names, results = f.cv_score(X_train,y_train,models,['accuracy',
                                                    'precision_cl_1',
                                                    'precision_cl_2',
                                                    'recall_cl_1',
                                                    'recall_cl_2',
                                                    'specificity_cl_1',
                                                    'specificity_cl_2'])

ScoreCard = f.concat_model_score(names, results, ScoreCard)


############################## ROUGH ############################
#ScoreCard.drop(ScoreCard.iloc[0:,12:],axis=1,inplace=True)

