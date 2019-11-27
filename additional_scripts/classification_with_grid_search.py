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
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.pipeline import make_pipeline
import os



def calculate_f1_score(y_true, y_pred):
  return f1_score(y_true, y_pred)


f1_scorer = make_scorer(calculate_f1_score, greater_is_better=True)



for file in os.listdir('final_datasets/'):
  print('***Processing: ', file , '  *****************')
  file_name = file
  df = pd.read_pickle('final_datasets/' + file_name)
  df_results = pd.DataFrame(columns=['model', 'tn','fp','fn','tp','precision','recall','f1'])



  print(df.head())

  print(df['label'].value_counts())


  print(df.columns)



  Y = df.label
  Y = Y.astype('int')
  X = df.drop(['entity_id_wiki_1','entity_id_wiki_2','label'], axis=1)

  #print(X.head())

  X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1234)


  print('Training Labels: ', y_train.value_counts())
  print('Testing Labels: ', y_test.value_counts())


  X_train['label'] = y_train

  df_pos_class_count = len(X_train[X_train.label==1])
  df_neg_class_count = len(X_train[X_train.label==0])


  print('***********balancing***********')
  df_not_match = X_train[df.label==0]
  df_match = X_train[df.label==1]

  df_not_matched_upsampled = resample(df_not_match, replace=True, n_samples=max(df_pos_class_count,df_neg_class_count), random_state=123) 
  df_upsampled = pd.concat([df_not_matched_upsampled, df_match])
  X_train = df_upsampled.reset_index(drop=True)

  #print(X_train['label'].value_counts())
  #print(y_train.value_counts())

  y_train = X_train['label']
  X_train.drop(columns = ['label'], inplace=True)

  




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


  my_model = gs.best_estimator_


  y_pred = my_model.predict(X_test)

  print(my_model.score(X_train, y_train))
  print(my_model.score(X_test, y_test))

  # How's our accuracy?
  print('Accuracy: ', accuracy_score(y_test, y_pred))
  print('Precision: ', precision_score(y_test, y_pred))
  print('Recall: ', recall_score(y_test, y_pred))
  print('F-1: ', f1_score(y_test, y_pred))
  print('Confusion Matrix:', confusion_matrix(y_test, y_pred))

  conf_matrix = confusion_matrix(y_test, y_pred)
  precision = precision_score(y_test, y_pred)
  recall = recall_score(y_test, y_pred)
  f1 = f1_score(y_test, y_pred)

  df_results = df_results.append({'model' : 'Naive Bayes' , 'tn' : conf_matrix[0,0], 'fp' : conf_matrix[0,1], 
  'fn' : conf_matrix[1,0], 'tp' : conf_matrix[1,1], 'precision': precision, 'recall': recall ,'f1': f1} , ignore_index=True)




  print('************Decision Tree***********')
  #clf_gini = DecisionTreeClassifier(criterion = "gini", splitter = 'random', random_state = 0, max_depth=4, min_samples_leaf=50)
  from sklearn.pipeline import make_pipeline

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


  print('Best score:', gs.best_score_)
  print('Best parameters:', gs.best_params_)
  print('Cross Validation Results: ', gs.cv_results_['mean_test_score'])


  my_model = gs.best_estimator_


  y_pred = my_model.predict(X_test)

  print(my_model.score(X_train, y_train))
  print(my_model.score(X_test, y_test))

  # How's our accuracy?
  print('Accuracy: ', accuracy_score(y_test, y_pred))
  print('Precision: ', precision_score(y_test, y_pred))
  print('Recall: ', recall_score(y_test, y_pred))
  print('F-1: ', f1_score(y_test, y_pred))
  print('Confusion Matrix:', confusion_matrix(y_test, y_pred))


  conf_matrix = confusion_matrix(y_test, y_pred)
  precision = precision_score(y_test, y_pred)
  recall = recall_score(y_test, y_pred)
  f1 = f1_score(y_test, y_pred)

  df_results = df_results.append({'model' : 'Decision Trees' , 'tn' : conf_matrix[0,0], 'fp' : conf_matrix[0,1], 
  'fn' : conf_matrix[1,0], 'tp' : conf_matrix[1,1], 'precision': precision, 'recall': recall ,'f1': f1} , ignore_index=True)




  #tree.export_graphviz(clf_gini, out_file='tree.dot')

  '''
  dot_data = tree.export_graphviz(clf_gini, out_file=None,
                                          label='all',
                                          filled = True,
                                          leaves_parallel=False,
                                          impurity=True,
                                          node_ids=True,
                                          proportion=False,
                                          rounded=True,
                                          special_characters=True
          )

  graph = graphviz.Source(dot_data)

  graph.render("WDC-OSM_Tree", view=True, format='png')
  '''

  print('************Logistic Regression***********')
  param_grid = [{"logisticregression__C":[1.0], "logisticregression__penalty":["l1"]}]


  pipe_lr = make_pipeline(LogisticRegression(random_state=0))

  gs = GridSearchCV(estimator=pipe_lr, param_grid=param_grid, scoring='f1', cv=10)




  print('************Fitting Model*************')
  gs.fit(X_train, y_train)
  print('************Model Fitted*************')


  print('Best score:', gs.best_score_)
  print('Best parameters:', gs.best_params_)
  print('Cross Validation Results: ', gs.cv_results_['mean_test_score'])


  my_model = gs.best_estimator_


  y_pred = my_model.predict(X_test)

  print(my_model.score(X_train, y_train))
  print(my_model.score(X_test, y_test))

  # How's our accuracy?
  print('Accuracy: ', accuracy_score(y_test, y_pred))
  print('Precision: ', precision_score(y_test, y_pred))
  print('Recall: ', recall_score(y_test, y_pred))
  print('F-1: ', f1_score(y_test, y_pred))
  print('Confusion Matrix:', confusion_matrix(y_test, y_pred))

  conf_matrix = confusion_matrix(y_test, y_pred)
  precision = precision_score(y_test, y_pred)
  recall = recall_score(y_test, y_pred)
  f1 = f1_score(y_test, y_pred)

  df_results = df_results.append({'model' : 'Logistic Regression' , 'tn' : conf_matrix[0,0], 'fp' : conf_matrix[0,1], 
  'fn' : conf_matrix[1,0], 'tp' : conf_matrix[1,1], 'precision': precision, 'recall': recall ,'f1': f1} , ignore_index=True)

  print('************SVM***********')
  param_grid = [{"svc__kernel":['linear'], 'svc__C': [1]}]


  pipe_lr = make_pipeline(SVC(random_state=0))

  gs = GridSearchCV(estimator=pipe_lr, param_grid=param_grid, scoring='f1', cv=10)


  print('************Fitting Model*************')
  gs.fit(X_train, y_train)
  print('************Model Fitted******** *****')


  print('Best score:', gs.best_score_)
  print('Best parameters:', gs.best_params_)
  print('Cross Validation Results: ', gs.cv_results_['mean_test_score'])


  my_model = gs.best_estimator_


  y_pred = my_model.predict(X_test)

  print(my_model.score(X_train, y_train))
  print(my_model.score(X_test, y_test))

  # How's our accuracy?
  print('Accuracy: ', accuracy_score(y_test, y_pred))
  print('Precision: ', precision_score(y_test, y_pred))
  print('Recall: ', recall_score(y_test, y_pred))
  print('F-1: ', f1_score(y_test, y_pred))
  print('Confusion Matrix:', confusion_matrix(y_test, y_pred))


  conf_matrix = confusion_matrix(y_test, y_pred)
  precision = precision_score(y_test, y_pred)
  recall = recall_score(y_test, y_pred)
  f1 = f1_score(y_test, y_pred)

  df_results = df_results.append({'model' : 'SVM' , 'tn' : conf_matrix[0,0], 'fp' : conf_matrix[0,1], 
  'fn' : conf_matrix[1,0], 'tp' : conf_matrix[1,1], 'precision': precision, 'recall': recall ,'f1': f1} , ignore_index=True)

  print('************Random Forest***********')
  param_grid = [{"randomforestclassifier__n_estimators":[100], 'randomforestclassifier__max_depth': [4]}]


  pipe_lr = make_pipeline(RandomForestClassifier(random_state=0))

  gs = GridSearchCV(estimator=pipe_lr, param_grid=param_grid, scoring='f1', cv=10)


  print('************Fitting Model*************')
  gs.fit(X_train, y_train)
  print('************Model Fitted*************')


  print('Best score:', gs.best_score_)
  print('Best parameters:', gs.best_params_)
  print('Cross Validation Results: ', gs.cv_results_['mean_test_score'])


  my_model = gs.best_estimator_


  y_pred = my_model.predict(X_test)

  print(my_model.score(X_train, y_train))
  print(my_model.score(X_test, y_test))

  # How's our accuracy?
  print('Accuracy: ', accuracy_score(y_test, y_pred))
  print('Precision: ', precision_score(y_test, y_pred))
  print('Recall: ', recall_score(y_test, y_pred))
  print('F-1: ', f1_score(y_test, y_pred))
  print('Confusion Matrix:', confusion_matrix(y_test, y_pred))

  conf_matrix = confusion_matrix(y_test, y_pred)
  precision = precision_score(y_test, y_pred)
  recall = recall_score(y_test, y_pred)
  f1 = f1_score(y_test, y_pred)

  df_results = df_results.append({'model' : 'Random Forest' ,  'tn' : conf_matrix[0,0], 'fp' : conf_matrix[0,1], 
  'fn' : conf_matrix[1,0], 'tp' : conf_matrix[1,1], 'precision': precision, 'recall': recall ,'f1': f1} , ignore_index=True)

  print('************XGBoost***********')
  param_grid = [{"xgbclassifier__n_estimators":[100], 'xgbclassifier__learning_rate': [0.1]}]


  pipe_lr = make_pipeline(XGBClassifier(random_state=0))

  gs = GridSearchCV(estimator=pipe_lr, param_grid=param_grid, scoring='f1', cv=10)


  print('************Fitting Model*************')
  gs.fit(X_train, y_train)
  print('************Model Fitted*************')


  print('Best score:', gs.best_score_)
  print('Best parameters:', gs.best_params_)
  print('Cross Validation Results: ', gs.cv_results_['mean_test_score'])


  my_model = gs.best_estimator_


  y_pred = my_model.predict(X_test)

  print(my_model.score(X_train, y_train))
  print(my_model.score(X_test, y_test))

  # How's our accuracy?
  print('Accuracy: ', accuracy_score(y_test, y_pred))
  print('Precision: ', precision_score(y_test, y_pred))
  print('Recall: ', recall_score(y_test, y_pred))
  print('F-1: ', f1_score(y_test, y_pred))
  print('Confusion Matrix:', confusion_matrix(y_test, y_pred))

  conf_matrix = confusion_matrix(y_test, y_pred)
  precision = precision_score(y_test, y_pred)
  recall = recall_score(y_test, y_pred)
  f1 = f1_score(y_test, y_pred)

  df_results = df_results.append({'model' : 'XGBoost' ,  'tn' : conf_matrix[0,0], 'fp' : conf_matrix[0,1], 
  'fn' : conf_matrix[1,0], 'tp' : conf_matrix[1,1], 'precision': precision, 'recall': recall ,'f1': f1} , ignore_index=True)

  df_results.to_excel('results/' + file_name.replace('.pkl','.xlsx'), index=False)