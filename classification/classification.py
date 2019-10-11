from sklearn.decomposition import PCA
import pandas as pd
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
import wittgenstein as lw
from sklearn import tree
import graphviz
from sklearn.metrics import confusion_matrix
import os
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.pipeline import make_pipeline


class CLASSIFICATION:
    # this function runs selected algorithm on training set and return prediction label on test set
    def run_classification(self, X_train, y_train, X_test, classification_algorithm):

        y_pred = None
        
        ################ Model Training ########################
        X_train['label'] = y_train
        
        df_pos_class_count = len(X_train[X_train.label==1])
        df_neg_class_count = len(X_train[X_train.label==0])


        print('***********balancing***********')
        df_not_match = X_train[df.label==0]
        df_match = X_train[df.label==1]

        if len(df_not_match) < len(df_match):
            df_not_matched_upsampled = resample(df_not_match, replace=True, n_samples=max(df_pos_class_count,df_neg_class_count), random_state=123) 
            df_upsampled = pd.concat([df_not_matched_upsampled, df_match])
            X_train = df_upsampled.reset_index(drop=True)
        elif len(df_not_match) < len(df_match):
            df_matched_upsampled = resample(df_match, replace=True, n_samples=max(df_pos_class_count,df_neg_class_count), random_state=123) 
            df_upsampled = pd.concat([df_matched_upsampled, df_not_match])
            X_train = df_upsampled.reset_index(drop=True)

        
        y_train = X_train['label']
        X_train.drop(columns = ['label'], inplace=True)

        if classification_algorithm == 'NB':
            print('************Naive Bayes***********')
            param_grid = [{"gaussiannb__priors":[None], 'gaussiannb__var_smoothing': [1e-09]}]
            
            pipe_lr = make_pipeline(GaussianNB())
            gs = GridSearchCV(estimator=pipe_lr, param_grid=param_grid, scoring='f1', cv=10)
            
            print('************Fitting Model*************')
            gs.fit(X_train, y_train)
            print('************Model Fitted*************')
            
            print('Best score:', gs.best_score_)
            print('Best parameters:', gs.best_params_)
            print('Cross Validation Results: ', gs.cv_results_['mean_test_score'])
            
            best_model = gs.best_estimator_
            y_pred = best_model.predict(X_test)
        
        elif classification_algorithm == 'DT':
            print('************Decision Tree***********')
            depths = [10]
            num_leafs = [50]
            
            param_grid = [{'decisiontreeclassifier__max_depth':depths,
            'decisiontreeclassifier__criterion':['gini'],
            'decisiontreeclassifier__splitter':['random'],
            'decisiontreeclassifier__min_samples_leaf':num_leafs}]
            
            pipe_tree = make_pipeline(DecisionTreeClassifier(random_state=0))
            gs = GridSearchCV(estimator=pipe_tree, param_grid=param_grid, scoring='f1', cv=10)
            
            print('************Fitting Model*************')
            gs.fit(X_train, y_train)
            print('************Model Fitted*************')
            
            best_model = gs.best_estimator_
            y_pred = best_model.predict(X_test)

        

        elif classification_algorithm == 'LR':
            print('************Logistic Regression***********')
            param_grid = [{"logisticregression__C":[1.0], "logisticregression__penalty":["l1"]}]
            
            pipe_lr = make_pipeline(LogisticRegression(random_state=0))
            gs = GridSearchCV(estimator=pipe_lr, param_grid=param_grid, scoring='f1', cv=10)
            
            print('************Fitting Model*************')
            gs.fit(X_train, y_train)
            print('************Model Fitted*************')
            
            best_model = gs.best_estimator_
            y_pred = best_model.predict(X_test)

        elif classification_algorithm == 'SVM':
            print('************SVM***********')
            param_grid = [{"svc__kernel":['linear'], 'svc__C': [1]}]
            
            pipe_lr = make_pipeline(SVC(random_state=0))
            gs = GridSearchCV(estimator=pipe_lr, param_grid=param_grid, scoring='f1', cv=10)
            
            print('************Fitting Model*************')
            gs.fit(X_train, y_train)
            print('************Model Fitted******** *****')
            
            best_model = gs.best_estimator_
            y_pred = best_model.predict(X_test)

        
        elif classification_algorithm == 'RF':
            print('************Random Forest***********')
            param_grid = [{"randomforestclassifier__n_estimators":[100], 'randomforestclassifier__max_depth': [4]}]
            
            pipe_lr = make_pipeline(RandomForestClassifier(random_state=0))
            gs = GridSearchCV(estimator=pipe_lr, param_grid=param_grid, scoring='f1', cv=10)
            
            print('************Fitting Model*************')
            gs.fit(X_train, y_train)
            print('************Model Fitted*************')
            
            best_model = gs.best_estimator_
            y_pred = best_model.predict(X_test)

        else:
            print('************XGBoost***********')
            param_grid = [{"xgbclassifier__n_estimators":[100], 'xgbclassifier__learning_rate': [0.1]}]
            
            pipe_lr = make_pipeline(XGBClassifier(random_state=0))
            gs = GridSearchCV(estimator=pipe_lr, param_grid=param_grid, scoring='f1', cv=10)
            
            print('************Fitting Model*************')
            gs.fit(X_train, y_train)
            print('************Model Fitted*************')
            
            best_model = gs.best_estimator_
            y_pred = best_model.predict(X_test)
        
        return y_pred