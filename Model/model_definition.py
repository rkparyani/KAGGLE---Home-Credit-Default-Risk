
from Preprocessing.Preprocessing import preprocessing as prep
import Model.FunctionLib as f
import pandas as pd


class model_def(object):
    X_train = None
    y_train = None
    SEED = 7
    
    def set_train_data(self, X_train, y_train, SEED):
        self.X_train = X_train
        self.y_train = y_train
        self.SEED = SEED
            
    def baseline_model(self, scoring, SEED, result_col_nm):
        # Run the baseline models on the unbalanced dataset 
        models = f.GetBasedModel()
        names,results = f.get_model_performance(self.X_train, self.y_train,models, SEED, scoring)
        f.PlotBoxR().PlotResult(names,results)
        
        _score = f.ScoreDataFrame(names,results,result_col_nm)
        return _score
    
    def scaled_model(self, scoring, SEED, result_col_nm, scalar):
        models = f.GetScaledModel(scalar)
        names,results = f.get_model_performance(self.X_train, self.y_train,models,SEED,scoring)
        _score = f.ScoreDataFrame(names, results, result_col_nm)
        return _score
    
    def scaled_model_with_CW_factor(self, scoring, SEED, scalar):
        models = f.GetScaledModelwithfactorizedCW(scalar)
        names, results = f.cv_score(self.X_train,self.y_train,models, scoring, SEED)
        _score = f.cv_metrics(names, results)
        return _score


    def get_oof(clf, x_train, y_train, x_test):
        oof_train = np.zeros((ntrain,))
        oof_test = np.zeros((ntest,))
        oof_test_skf = np.empty((NFOLDS, ntest))
    
        for i, (train_index, test_index) in enumerate(kf):
            x_tr = x_train[train_index]
            y_tr = y_train[train_index]
            x_te = x_train[test_index]
    
            clf.train(x_tr, y_tr)
    
            oof_train[test_index] = clf.predict(x_te)
            oof_test_skf[i, :] = clf.predict(x_test)
    
        oof_test[:] = oof_test_skf.mean(axis=0)
        return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)




# Class to extend the Sklearn classifier
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    def fit(self,x,y):
        return self.clf.fit(x,y)
    
    def feature_importances(self,x,y):
        print(self.clf.fit(x,y).feature_importances_)
    
# Class to extend XGboost classifer        