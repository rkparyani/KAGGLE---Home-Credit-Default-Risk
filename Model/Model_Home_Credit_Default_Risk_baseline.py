####################### LIBRARIES ################################
# Python Standard Libraries
import os
import pandas as pd
import numpy as np
from scipy.stats import uniform

# User defined libraries and Modules
import Model.FunctionLib as f
import Preprocessing.Preprocessing as prep
import Model.dataset_definition as d_def
import Model.model_definition as m_def

###################  DECLARATION AND INSTANTIATION ####################### 
# Variables 
wd = "D:\RITESH\Data Science\GIT WD\KAGGLE - Home-Credit-Default-Risk"
train_df_path = wd+"\\Output\\application_train_clean_final.csv"
feats_ignore = ['TARGET','SK_ID_CURR']
SEED = 7

# Set working directory
os.chdir(wd)

# Instantiate the preprocessing ckass and set the variables
baseline_prep = prep.preprocessing(train_df_path)
baseline_prep.set_feats_ignore(feats_ignore)


###################  DATA PREPROCESSING AND BASIC FEATURE ENGINEERING ####################### 

#import Preprocessing.Preprocessing_app_train_new 
##import Preprocessing.Preprocessing_app_test
#import Preprocessing.Preprocessing_bureau_new
#import Preprocessing.Preprocessing_pos_cash_bal_new
#import Preprocessing.Preprocessing_cc_bal_new
#import Preprocessing.Preprocessing_installment_payments_new
#import Preprocessing.Preprocessing_previous_application_new

###################### DIMENTIONALITY REDUCTION #####################################

# Instantiate the dataset_def class and import dataset into class variable
def_data = d_def.data_def(train_df_path)
def_model = m_def.model_def()


# Create all different required datasets through various defined methods of dimentionality reduction
def_data.create_dataset_remove_corr_feats('TARGET', 1, 0.9, feats_ignore )

# Use one of the diff datasets from dim_reduction class variables
train_df  = def_data.dim_red_by_corr_df

# Change the datatype of categorical and numerical values (NOT REQUIRED)
#f.change_type(train_df,num_feats,count_threshold=5)

# Seperate the categorical and numerical features
# Get the list of attributes and their properties to start
par_num_start, par_cat_start = baseline_prep.define_params(train_df)

 
###################### TRAIN AND TEST SET SPLIT #####################################SEED = 7

X =  train_df.drop(train_df[['TARGET','SK_ID_CURR']],axis=1).copy()
Y = train_df[['TARGET']]

X_train, X_test, y_train, y_test =f.train_test_split(X,Y,
                                                   test_size=0.25,
                                                   random_state=0,
                                                   stratify=train_df['TARGET'])

############# RUN BASE MODELS ##################

# Run the baseline models on the unbalanced dataset 
#models = f.GetBasedModel()
#names,results = f.get_model_performance(X_train, y_train,models, SEED, 'f1_weighted')
#f.PlotBoxR().PlotResult(names,results)
#
#basedLineF1Score = f.ScoreDataFrame(names,results,'baseline_f1_Score')
basedLineF1Score = def_model.baseline_model('f1_weighted', 
                                            SEED, 
                                            'baseline_f1_Score')

#models = f.GetBasedModel()
#names, results = f.get_model_performance(X_train, y_train,models, SEED,'accuracy')
#f.PlotBoxR().PlotResult(names,results)
#
#basedLineAccuracyScore = f.ScoreDataFrame(names,results,'baseline_accuracy')
basedLineAccuracyScore = def_model.baseline_model('accuracy', 
                                                  SEED, 
                                                  'baseline_accuracy')

# Record Scores
ScoreCard = pd.concat([basedLineAccuracyScore,
                       basedLineF1Score], axis=1)


############### RUN SCALED MODELS WITH AND WITHOUT FACTORING FOR CLASS WEIGHT #################
# Scaled Model on top of baseline
# Standard Scalar
#models = f.GetScaledModel('standard')
#names,results = f.get_model_performance(X_train, y_train,models,SEED,'f1_weighted')
##f.PlotBoxR().PlotResult(names,results)
#
## Record Scores
#scaledScoreStandard = f.ScoreDataFrame(names,results,'standard_f1_score')

scaledScoreStandard = def_model.scaled_model('accuracy', 
                                                  SEED, 
                                                  'baseline_accuracy',
                                                  'standard')
ScoreCard = pd.concat([ScoreCard,
                           scaledScoreStandard], axis=1)


# Minmax scalar
#models = f.GetScaledModel('minmax')
#names,results = f.get_model_performance(X_train, y_train,models,SEED,'f1_weighted')
#f.PlotBoxR().PlotResult(names,results)
#
## Record Scores
#scaledScoreMinMax = f.ScoreDataFrame(names,results,'minmax_f1_score')

scaledScoreMinMax = def_model.scaled_model('accuracy', 
                                                  SEED, 
                                                  'baseline_accuracy',
                                                  'minmax')
ScoreCard = pd.concat([ScoreCard,
                          scaledScoreMinMax], axis=1)


# Scaled Model on top of baseline with models adjusted for Class Weight
# Standard Scalar
#models = f.GetScaledModelwithfactorizedCW('standard')
#names,results = f.get_model_performance(X_train, y_train,models,SEED,'f1_weighted')
#f.PlotBoxR().PlotResult(names,results)
#
## Record Scores
#scaledScoreStandard = f.ScoreDataFrame(names,results,'standard_f1_score')

scaledScoreStandard = def_model.scaled_model('f1_weighted', 
                                                  SEED, 
                                                  'standard_f1_score',
                                                  'standard')

ScoreCard = pd.concat([ScoreCard,
                           scaledScoreStandard], axis=1)


# Minmax scalar
#models = f.GetScaledModelwithfactorizedCW('minmax')
#names,results = f.get_model_performance(X_train, y_train,models,SEED,'f1_weighted')
#f.PlotBoxR().PlotResult(names,results)
#
## Record Scores
#scaledScoreMinMax = f.ScoreDataFrame(names,results,'minmax_f1_score')


scaledScoreMinMax = def_model.scaled_model('f1_weighted', 
                                                  SEED, 
                                                  'standard_f1_score',
                                                  'minmax')

ScoreCard = pd.concat([ScoreCard,
                          scaledScoreMinMax], axis=1)

# Since this is an imbalance class problem: The Models scaled to classweight and then looking into 
# the metrics for minority class along with the majority class in TARGET

scaledscoreCW = def_model.scaled_model_with_CW_factor(['accuracy',
                                                    'specificity_cl_1',
                                                    'precision_cl_1',
                                                    'recall_cl_1',
                                                    'specificity_cl_2',
                                                    'precision_cl_2',
                                                    'recall_cl_2',
                                                    'AUC_cl_1',
                                                    'AUC_cl_2',
                                                    'MCC'], 
                                                  SEED, 
                                                  'standard')

#ScoreCard = f.concat_model_score(names, results, ScoreCard)
ScoreCard = pd.concat([ScoreCard,
                          scaledScoreMinMax], axis=1)

####### Finding best parameters for the most performing models #########

# HYPERPARAMETER TUNING WITH RANDOM SEARCH.

# Logistic Regression model
penalty = ['l1', 'l2']

# Create regularization hyperparameter distribution using uniform distribution
C = uniform(loc=0, scale=4)

# Create hyperparameter options
hyperparameters = dict(C=C, penalty=penalty)

model = f.LogisticRegression()
LR_RandSearch = f.RandomSearch(X_train,y_train,model,hyperparameters)
LR_best_model,LR_best_params = LR_RandSearch.RandomSearch()
Prediction_LR = LR_RandSearch.BestModelPredict(X_train)

# ADABOOST CLASSIFIER

learning_rate_value = [.01,.05,.1,.5,1]
n_estimators_value = [50,100,150,200,250,300]

param_grid = dict(learning_rate=learning_rate_value, n_estimators=n_estimators_value)

model_Ad = f.AdaBoostClassifier()
Ad_RandSearch = f.RandomSearch(X_train,y_train,model_Ad,param_grid)
Ad_best_model,Ad_best_params = Ad_RandSearch.RandomSearch()
Prediction_Ad = Ad_RandSearch.BestModelPredict(X_train)

# Gradient Boosting Classifier

learning_rate_value = [.01,.05,.1,.5,1]
n_estimators_value = [50,100,150,200,250,300]

param_grid = dict(learning_rate=learning_rate_value, n_estimators=n_estimators_value)

model_GB = f.GradientBoostingClassifier()
GB_RandSearch = f.RandomSearch(X_train,y_train,model_GB,param_grid)
GB_best_model,GB_best_params = GB_RandSearch.RandomSearch()
Prediction_GB = GB_RandSearch.BestModelPredict(X_train)

# Random Forest Classifier
n_estimators_value = [50,100,150,200,250,300]
criterion_val = ["gini", "entropy"]

param_grid = dict(criterion=criterion_val, n_estimators=n_estimators_value)

model_RF = f.RandomForestClassifier()
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

model_CART = f.RandomForestClassifier()
CART_RandSearch = f.RandomSearch(X_train,y_train,model_CART,param_grid)
CART_best_model,CART_best_params = CART_RandSearch.RandomSearch()
Prediction_CART = CART_RandSearch.BestModelPredict(X_train)

# KNN best model
n_neighbors_1  = np.arange(5,75,5)
param_grid = dict(n_neighbors = n_neighbors_1)

model_KNN = f.KNeighborsClassifier()
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
                                                    'specificity_cl_1',
                                                    'precision_cl_1',
                                                    'recall_cl_1',
                                                    'specificity_cl_2',
                                                    'precision_cl_2',
                                                    'recall_cl_2',
                                                    'AUC_cl_1',
                                                    'AUC_cl_2',
                                                    'MCC'])

ScoreCard = f.concat_model_score(names, results, ScoreCard)

results_df = pd.DataFrame(results, index = names)


# HYPER PARAMETER TUNING WITH GRID SEARCH for best performing models

# Logistic Regression model
penalty = ['l1', 'l2']

# Create regularization hyperparameter distribution using uniform distribution
C = uniform(loc=0, scale=4)

# Create hyperparameter options
hyperparameters = dict(C=C, penalty=penalty)

model = f.LogisticRegression()
LR_GridSearch = f.GridSearch(X_train,y_train,model,hyperparameters)
LR_best_model,LR_best_params = LR_RandSearch.GridSearch()
Prediction_LR = LR_GridSearch.BestModelPredict(X_train)


# ADABOOST CLASSIFIER

learning_rate_value = [.001, .01, .05, .1, .5, 1]
n_estimators_value = [25, 50, 100, 150, 200, 250, 300, 400, 500]

param_grid = dict(learning_rate=learning_rate_value, n_estimators=n_estimators_value)

model_Ad = f.AdaBoostClassifier()
Ad_GridSearch = f.GridSearch(X_train,y_train,model_Ad,param_grid)
Ad_best_model,Ad_best_params = Ad_RandSearch.GridSearch()
Prediction_Ad = Ad_GridSearch.BestModelPredict(X_train)


# Gradient Boosting Classifier

learning_rate_value = [.001, .01, .05, .1, .5, 1]
n_estimators_value = [25, 50, 100, 150, 200, 250, 300, 400, 500]

param_grid = dict(learning_rate=learning_rate_value, n_estimators=n_estimators_value)

model_GB = f.GradientBoostingClassifier()
GB_GridSearch = f.GridSearch(X_train,y_train,model_GB,param_grid)
GB_best_model, GB_best_params = GB_GridSearch.GridSearch()
Prediction_GB = GB_GridSearch.BestModelPredict(X_train)


# Run the best model pipeline
models = f.GetScaledModelwithbestparams('standard',
                                        LR_best_params,
                                        Ad_best_params,
                                        GB_best_params,
                                        RF_best_params,
                                        CART_best_params,
                                        KNN_best_params)

names, results = f.cv_score(X_train,y_train,models,['accuracy',
                                                    'specificity_cl_1',
                                                    'precision_cl_1',
                                                    'recall_cl_1',
                                                    'specificity_cl_2',
                                                    'precision_cl_2',
                                                    'recall_cl_2',
                                                    'AUC_cl_1',
                                                    'AUC_cl_2',
                                                    'MCC'])



################# OUTCOME ##########################

ScoreCard.to_csv(wd+'\\Output\\ScoreCard_Train_DS1.csv')

train_df.columns.tolist()

############################## ROUGH ############################
#ScoreCard.drop(ScoreCard.iloc[0:,21:],axis=1,inplace=True)


############    Reference Code   ###############

# Some useful parameters which will come in handy later on
ntrain = train.shape[0]
ntest = test.shape[0]
SEED = 0 # for reproducibility
NFOLDS = 5 # set folds for out-of-fold prediction
kf = KFold(ntrain, n_folds= NFOLDS, random_state=SEED)



# Put in our parameters for said classifiers
# Random Forest parameters
rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
     'warm_start': True, 
     #'max_features': 0.2,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0
}

# Extra Trees Parameters
et_params = {
    'n_jobs': -1,
    'n_estimators':500,
    #'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 500,
    'learning_rate' : 0.75
}

# Gradient Boosting parameters
gb_params = {
    'n_estimators': 500,
     #'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0
}

# Support Vector Classifier parameters 
svc_params = {
    'kernel' : 'linear',
    'C' : 0.025
    }


# Create 5 objects that represent our 4 models
rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)


