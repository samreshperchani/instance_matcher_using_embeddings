import pandas as pd
from sklearn.decomposition import TruncatedSVD
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



def calculate_f1_score(y_true, y_pred):
  return f1_score(y_true, y_pred)


f1_scorer = make_scorer(calculate_f1_score, greater_is_better=True)

for file in os.listdir('data_revised_negative_example/'):
        print('***Processing: ', file , '  *****************')
        file_name = file
        df_results = pd.DataFrame(columns=['model', 'tn','fp','fn','tp','precision','recall','f1'])


        df = pd.read_pickle('data_revised_negative_example/' + file_name)
        df = df.drop_duplicates(subset=['entity_id_wiki_1','entity_id_wiki_2'])
        df = df.reset_index(drop=True)


        vector_length = 300
        reduced_dimensions = 100


        vector_col_name_doc2vec_ent1 = ['doc2vec_ent1_' + str(i) for i in range(0,vector_length) ]
        df[vector_col_name_doc2vec_ent1]= pd.DataFrame(df.doc2vec_vector_ent1.tolist(), index=df.index)
        df.drop(columns=['doc2vec_vector_ent1'], inplace=True)

        vector_col_name_doc2vec_ent2 = ['doc2vec_ent2_' + str(i) for i in range(0,vector_length) ]
        df[vector_col_name_doc2vec_ent2]= pd.DataFrame(df.doc2vec_vector_ent2.tolist(), index=df.index)
        df.drop(columns=['doc2vec_vector_ent2'], inplace=True)

        vector_col_name_word2vec_ent1 = ['word2vec_ent1_' + str(i) for i in range(0,vector_length) ]
        df[vector_col_name_word2vec_ent1]= pd.DataFrame(df.word2vec_vector_ent1.tolist(), index=df.index)
        df.drop(columns=['word2vec_vector_ent1'], inplace=True)

        vector_col_name_word2vec_ent2 = ['word2vec_ent2_' + str(i) for i in range(0,vector_length) ]
        df[vector_col_name_word2vec_ent2]= pd.DataFrame(df.word2vec_vector_ent2.tolist(), index=df.index)
        df.drop(columns=['word2vec_vector_ent2'], inplace=True)

        vector_col_name_rdf2vec_ent1 = ['rdf2vec_ent1_' + str(i) for i in range(0,vector_length) ]
        df[vector_col_name_rdf2vec_ent1]= pd.DataFrame(df.rdf2vec_vector_ent1.tolist(), index=df.index)
        df.drop(columns=['rdf2vec_vector_ent1'], inplace=True)

        vector_col_name_rdf2vec_ent2 = ['rdf2vec_ent2_' + str(i) for i in range(0,vector_length) ]
        df[vector_col_name_rdf2vec_ent2]= pd.DataFrame(df.rdf2vec_vector_ent2.tolist(), index=df.index)
        df.drop(columns=['rdf2vec_vector_ent2'], inplace=True)





        Y = df.label
        Y = Y.astype('int')
        X = df.drop(['label'], axis=1)


        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1234)
        training_indices = X_train.index
        testing_indices = X_test.index


        print(len(X_train))
        print(len(X_test))
        print(len(X_test['entity_id_wiki_2'].unique()))



        ########################### svd on DOC2Vec #############################################
        vector_col_name_doc2vec = ['doc2vec_ent' + str(i) for i in range(0,vector_length)]
        vector_col_name_doc2vec.append('entity_id')
        df_doc2vec_ent1_train = X_train[vector_col_name_doc2vec_ent1 + ['entity_id_wiki_1']]
        df_doc2vec_ent1_train.columns = vector_col_name_doc2vec

        df_doc2vec_ent2_train = X_train[vector_col_name_doc2vec_ent2 + ['entity_id_wiki_2']]
        df_doc2vec_ent2_train.columns = vector_col_name_doc2vec

        df_doc2vec_entities_train = pd.concat([df_doc2vec_ent1_train, df_doc2vec_ent2_train])
        df_doc2vec_entities_train = df_doc2vec_entities_train.reset_index(drop=True)
        df_doc2vec_entities_train_vectors = df_doc2vec_entities_train.drop(columns=['entity_id'])

        print(df_doc2vec_entities_train.head())

        svd_doc2vec = TruncatedSVD(n_components=reduced_dimensions, random_state = 2019)
        svd_doc2vec.fit(df_doc2vec_entities_train_vectors)

        X_svd_doc2vec_train = svd_doc2vec.transform(df_doc2vec_entities_train_vectors)
        df_svd_doc2vec_train = pd.DataFrame.from_records(X_svd_doc2vec_train)
        df_svd_doc2vec_train.columns = ['doc2vec_svd_ent' + str(i) for i in range(0,reduced_dimensions)]
        df_doc2vec_entities_train = df_doc2vec_entities_train[['entity_id']]
        df_svd_doc2vec_train = pd.merge(df_svd_doc2vec_train, df_doc2vec_entities_train, how='inner', right_index=True, left_index=True)


        df_doc2vec_ent1_test = X_test[vector_col_name_doc2vec_ent1 + ['entity_id_wiki_1']]
        df_doc2vec_ent1_test.columns = vector_col_name_doc2vec
        df_doc2vec_ent2_test = X_test[vector_col_name_doc2vec_ent2+ ['entity_id_wiki_2']]
        df_doc2vec_ent2_test.columns = vector_col_name_doc2vec
        df_doc2vec_entities_test = pd.concat([df_doc2vec_ent1_test, df_doc2vec_ent2_test])
        df_doc2vec_entities_test = df_doc2vec_entities_test.reset_index(drop=True)
        df_doc2vec_entities_test_vectors = df_doc2vec_entities_test.drop(columns=['entity_id'])

        X_svd_doc2vec_test = svd_doc2vec.transform(df_doc2vec_entities_test_vectors)
        df_svd_doc2vec_test = pd.DataFrame.from_records(X_svd_doc2vec_test)
        df_svd_doc2vec_test.columns = ['doc2vec_svd_ent' + str(i) for i in range(0,reduced_dimensions)]
        df_doc2vec_entities_test = df_doc2vec_entities_test[['entity_id']]

        print('After svd Test: ', len(df_svd_doc2vec_test))

        df_svd_doc2vec_test = pd.merge(df_svd_doc2vec_test, df_doc2vec_entities_test, how='inner', right_index=True, left_index=True)


        ########################### svd on Word2Vec #############################################
        vector_col_name_word2vec = ['word2vec_ent' + str(i) for i in range(0,vector_length)]
        vector_col_name_word2vec.append('entity_id')
        df_word2vec_ent1_train = X_train[vector_col_name_word2vec_ent1 + ['entity_id_wiki_1']]
        df_word2vec_ent1_train.columns = vector_col_name_word2vec

        df_word2vec_ent2_train = X_train[vector_col_name_word2vec_ent2 + ['entity_id_wiki_2']]
        df_word2vec_ent2_train.columns = vector_col_name_word2vec

        df_word2vec_entities_train = pd.concat([df_word2vec_ent1_train, df_word2vec_ent2_train])
        df_word2vec_entities_train = df_word2vec_entities_train.reset_index(drop=True)
        df_word2vec_entities_train_vectors = df_word2vec_entities_train.drop(columns=['entity_id'])

        print(df_word2vec_entities_train.head())

        svd_word2vec = TruncatedSVD(n_components=reduced_dimensions, random_state = 2019)
        svd_word2vec.fit(df_word2vec_entities_train_vectors)

        X_svd_word2vec_train = svd_word2vec.transform(df_word2vec_entities_train_vectors)
        df_svd_word2vec_train = pd.DataFrame.from_records(X_svd_word2vec_train)
        df_svd_word2vec_train.columns = ['word2vec_svd_ent' + str(i) for i in range(0,reduced_dimensions)]
        df_word2vec_entities_train = df_word2vec_entities_train[['entity_id']]
        df_svd_word2vec_train = pd.merge(df_svd_word2vec_train, df_word2vec_entities_train, how='inner', right_index=True, left_index=True)


        df_word2vec_ent1_test = X_test[vector_col_name_word2vec_ent1 + ['entity_id_wiki_1']]
        df_word2vec_ent1_test.columns = vector_col_name_word2vec
        df_word2vec_ent2_test = X_test[vector_col_name_word2vec_ent2+ ['entity_id_wiki_2']]
        df_word2vec_ent2_test.columns = vector_col_name_word2vec
        df_word2vec_entities_test = pd.concat([df_word2vec_ent1_test, df_word2vec_ent2_test])
        df_word2vec_entities_test = df_word2vec_entities_test.reset_index(drop=True)
        df_word2vec_entities_test_vectors = df_word2vec_entities_test.drop(columns=['entity_id'])

        X_svd_word2vec_test = svd_word2vec.transform(df_word2vec_entities_test_vectors)
        df_svd_word2vec_test = pd.DataFrame.from_records(X_svd_word2vec_test)
        df_svd_word2vec_test.columns = ['word2vec_svd_ent' + str(i) for i in range(0,reduced_dimensions)]
        df_word2vec_entities_test = df_word2vec_entities_test[['entity_id']]

        print('After svd Test: ', len(df_svd_word2vec_test))

        df_svd_word2vec_test = pd.merge(df_svd_word2vec_test, df_word2vec_entities_test, how='inner', right_index=True, left_index=True)

        ########################### svd on RDF2Vec #############################################
        vector_col_name_rdf2vec = ['rdf2vec_ent' + str(i) for i in range(0,vector_length)]
        vector_col_name_rdf2vec.append('entity_id')
        df_rdf2vec_ent1_train = X_train[vector_col_name_rdf2vec_ent1 + ['entity_id_wiki_1']]
        df_rdf2vec_ent1_train.columns = vector_col_name_rdf2vec

        df_rdf2vec_ent2_train = X_train[vector_col_name_rdf2vec_ent2 + ['entity_id_wiki_2']]
        df_rdf2vec_ent2_train.columns = vector_col_name_rdf2vec

        df_rdf2vec_entities_train = pd.concat([df_rdf2vec_ent1_train, df_rdf2vec_ent2_train])
        df_rdf2vec_entities_train = df_rdf2vec_entities_train.reset_index(drop=True)
        df_rdf2vec_entities_train_vectors = df_rdf2vec_entities_train.drop(columns=['entity_id'])

        print(df_rdf2vec_entities_train.head())

        svd_rdf2vec = TruncatedSVD(n_components=reduced_dimensions, random_state = 2019)
        svd_rdf2vec.fit(df_rdf2vec_entities_train_vectors)

        X_svd_rdf2vec_train = svd_rdf2vec.transform(df_rdf2vec_entities_train_vectors)
        df_svd_rdf2vec_train = pd.DataFrame.from_records(X_svd_rdf2vec_train)
        df_svd_rdf2vec_train.columns = ['rdf2vec_svd_ent' + str(i) for i in range(0,reduced_dimensions)]

        df_rdf2vec_entities_train = df_rdf2vec_entities_train[['entity_id']]
        df_svd_rdf2vec_train = pd.merge(df_svd_rdf2vec_train, df_rdf2vec_entities_train, how='inner', right_index=True, left_index=True)


        df_rdf2vec_ent1_test = X_test[vector_col_name_rdf2vec_ent1 + ['entity_id_wiki_1']]
        df_rdf2vec_ent1_test.columns = vector_col_name_rdf2vec
        df_rdf2vec_ent2_test = X_test[vector_col_name_rdf2vec_ent2+ ['entity_id_wiki_2']]
        df_rdf2vec_ent2_test.columns = vector_col_name_rdf2vec
        df_rdf2vec_entities_test = pd.concat([df_rdf2vec_ent1_test, df_rdf2vec_ent2_test])
        df_rdf2vec_entities_test = df_rdf2vec_entities_test.reset_index(drop=True)
        df_rdf2vec_entities_test_vectors = df_rdf2vec_entities_test.drop(columns=['entity_id'])

        X_svd_rdf2vec_test = svd_rdf2vec.transform(df_rdf2vec_entities_test_vectors)
        df_svd_rdf2vec_test = pd.DataFrame.from_records(X_svd_rdf2vec_test)
        df_svd_rdf2vec_test.columns = ['rdf2vec_svd_ent' + str(i) for i in range(0,reduced_dimensions)]
        df_rdf2vec_entities_test = df_rdf2vec_entities_test[['entity_id']]

        print('After svd Test: ', len(df_svd_rdf2vec_test))

        df_svd_rdf2vec_test = pd.merge(df_svd_rdf2vec_test, df_rdf2vec_entities_test, how='inner', right_index=True, left_index=True)




        X_train = X_train[['entity_id_wiki_1', 'entity_id_wiki_2']]
        X_train = pd.merge(X_train, df_svd_doc2vec_train, how='inner', left_on='entity_id_wiki_1', right_on='entity_id')
        X_train =  X_train.drop(['entity_id'], axis=1)
        X_train = pd.merge(X_train, df_svd_doc2vec_train, how='inner', left_on='entity_id_wiki_2', right_on='entity_id')
        X_train =  X_train.drop(['entity_id'], axis=1)
        X_train = X_train.drop_duplicates(subset=['entity_id_wiki_1', 'entity_id_wiki_2'])

        X_train = pd.merge(X_train, df_svd_word2vec_train, how='inner', left_on='entity_id_wiki_1', right_on='entity_id')
        X_train =  X_train.drop(['entity_id'], axis=1)
        X_train = pd.merge(X_train, df_svd_word2vec_train, how='inner', left_on='entity_id_wiki_2', right_on='entity_id')
        X_train =  X_train.drop(['entity_id'], axis=1)
        X_train = X_train.drop_duplicates(subset=['entity_id_wiki_1', 'entity_id_wiki_2'])

        X_train = pd.merge(X_train, df_svd_rdf2vec_train, how='inner', left_on='entity_id_wiki_1', right_on='entity_id')
        X_train =  X_train.drop(['entity_id'], axis=1)
        X_train = pd.merge(X_train, df_svd_rdf2vec_train, how='inner', left_on='entity_id_wiki_2', right_on='entity_id')
        X_train =  X_train.drop(['entity_id'], axis=1)
        X_train = X_train.drop_duplicates(subset=['entity_id_wiki_1', 'entity_id_wiki_2'])


        print(len(X_train))

        X_test = X_test[['entity_id_wiki_1', 'entity_id_wiki_2']]
        X_test = pd.merge(X_test, df_svd_doc2vec_test, how='inner', left_on='entity_id_wiki_1', right_on='entity_id')
        X_test =  X_test.drop(['entity_id'], axis=1)
        X_test = pd.merge(X_test, df_svd_doc2vec_test, how='inner', left_on='entity_id_wiki_2', right_on='entity_id')
        X_test =  X_test.drop(['entity_id'], axis=1)
        X_test = X_test.drop_duplicates(subset=['entity_id_wiki_1', 'entity_id_wiki_2'])

        X_test = pd.merge(X_test, df_svd_word2vec_test, how='inner', left_on='entity_id_wiki_1', right_on='entity_id')
        X_test =  X_test.drop(['entity_id'], axis=1)
        X_test = pd.merge(X_test, df_svd_word2vec_test, how='inner', left_on='entity_id_wiki_2', right_on='entity_id')
        X_test =  X_test.drop(['entity_id'], axis=1)
        X_test = X_test.drop_duplicates(subset=['entity_id_wiki_1', 'entity_id_wiki_2'])

        X_test = pd.merge(X_test, df_svd_rdf2vec_test, how='inner', left_on='entity_id_wiki_1', right_on='entity_id')
        X_test =  X_test.drop(['entity_id'], axis=1)
        X_test = pd.merge(X_test, df_svd_rdf2vec_test, how='inner', left_on='entity_id_wiki_2', right_on='entity_id')
        X_test =  X_test.drop(['entity_id'], axis=1)
        X_test = X_test.drop_duplicates(subset=['entity_id_wiki_1', 'entity_id_wiki_2'])


        df = df[['entity_id_wiki_1', 'entity_id_wiki_2','label']]

        X_train = pd.merge(X_train, df, how='inner', left_on=['entity_id_wiki_1','entity_id_wiki_2'], right_on=['entity_id_wiki_1','entity_id_wiki_2'])
        y_train = X_train['label']
        y_train=y_train.astype('int')

        X_train = X_train.drop(['label'], axis=1)


        X_test = pd.merge(X_test, df, how='inner', left_on=['entity_id_wiki_1','entity_id_wiki_2'], right_on=['entity_id_wiki_1','entity_id_wiki_2'])
        y_test = X_test['label']
        y_test=y_test.astype('int')
        X_test = X_test.drop(['label'], axis=1)

        ################ Model Training ########################

        X_train = X_train.drop(['entity_id_wiki_1','entity_id_wiki_2'], axis=1)
        print('Null Values: ', X_train.isnull().sum().sum())

        X_test = X_test.drop(['entity_id_wiki_1','entity_id_wiki_2'], axis=1)
        print('Null Values: ', X_test.isnull().sum().sum())


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

        df_results.to_excel('results/svd_nw2_' + file_name.replace('.pkl','.xlsx'), index=False)