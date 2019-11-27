from sklearn.decomposition import TruncatedSVD
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



def calculate_f1_score(y_true, y_pred):
  return f1_score(y_true, y_pred)


f1_scorer = make_scorer(calculate_f1_score, greater_is_better=True)

for file in os.listdir('dataset_ensembles/'):
        print('***Processing: ', file , '  *****************')
        file_name = file
        df_results = pd.DataFrame(columns=['model', 'tn','fp','fn','tp','precision','recall','f1'])


        df = pd.read_pickle('dataset_ensembles/' + file_name)
        df = df.drop_duplicates(subset=['entity_id_wiki_1','entity_id_wiki_2'])
        df = df.reset_index(drop=True)

        vector_length = 300
        reduced_dimensions = 300


        print(len(df))

        Y = df.label
        Y = Y.astype('int')
        X = df.drop(['label'], axis=1)

        #print(X.head())

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1234)
        training_indices = X_train.index
        testing_indices = X_test.index

        print(X_train.head())

        vector_col_name_doc2vec = ['doc2vec_ent_' + str(i) for i in range(0,vector_length) ]
        vector_col_name_word2vec = ['word2vec_ent_' + str(i) for i in range(0,vector_length) ]
        vector_col_name_rdf2vec = ['rdf2vec_ent_' + str(i) for i in range(0,vector_length) ]

        def get_entity_vectors(df):
                df_vec_en_1 = df[['entity_id_wiki_1','doc2vec_vector_ent1','word2vec_vector_ent1','rdf2vec_vector_ent1']]
                df_vec_en_1.rename(columns = {'entity_id_wiki_1': 'entity_id','doc2vec_vector_ent1' : 'doc2vec_vector', 
                'word2vec_vector_ent1' : 'word2vec_vector', 'rdf2vec_vector_ent1' : 'rdf2vec_vector'}, inplace=True)

                df_vec_en_2 = df[['entity_id_wiki_2','doc2vec_vector_ent2','word2vec_vector_ent2','rdf2vec_vector_ent2']]
                df_vec_en_2.rename(columns = {'entity_id_wiki_2': 'entity_id', 'doc2vec_vector_ent2' : 'doc2vec_vector', 
                'word2vec_vector_ent2' : 'word2vec_vector', 'rdf2vec_vector_ent2' : 'rdf2vec_vector'}, inplace=True)


                df_concat = pd.concat([df_vec_en_1, df_vec_en_2])

                df_concat = df_concat.reset_index(drop=True)


                df_concat[vector_col_name_doc2vec]= pd.DataFrame(df_concat.doc2vec_vector.tolist(), index=df_concat.index)
                df_concat.drop(columns=['doc2vec_vector'], inplace=True)



                df_concat[vector_col_name_word2vec]= pd.DataFrame(df_concat.word2vec_vector.tolist(), index=df_concat.index)
                df_concat.drop(columns=['word2vec_vector'], inplace=True)


                df_concat[vector_col_name_rdf2vec]= pd.DataFrame(df_concat.rdf2vec_vector.tolist(), index=df_concat.index)
                df_concat.drop(columns=['rdf2vec_vector'], inplace=True)

                df_concat = df_concat.drop_duplicates(subset=['entity_id'])
                df_concat = df_concat.reset_index(drop=True)

                #df_concat.drop(columns=['doc2vec_vector_ent1','word2vec_vector_ent1','rdf2vec_vector_ent1','doc2vec_vector_ent2','word2vec_vector_ent2','rdf2vec_vector_ent2'], inplace=True)
                
                return df_concat

        #print('Concatenate Length: ', len(df_concat))


        df_train = get_entity_vectors(X_train)
        df_test = get_entity_vectors(X_test)

        df_vectors_train = df_train.drop(columns = ['entity_id'])
        df_vectors_test = df_test.drop(columns = ['entity_id'])

        svd = TruncatedSVD(n_components=reduced_dimensions, random_state = 2019)
        svd.fit(df_vectors_train)

        X_svd_train = svd.transform(df_vectors_train)
        df_svd_train = pd.DataFrame.from_records(X_svd_train)


        X_svd_test = svd.transform(df_vectors_test)
        df_svd_test = pd.DataFrame.from_records(X_svd_test)



        df_train = pd.merge(df_train, df_svd_train, how='inner', left_index=True, right_index=True)
        df_train.drop(columns = vector_col_name_doc2vec + vector_col_name_word2vec + vector_col_name_rdf2vec, inplace=True)

        df_test = pd.merge(df_test, df_svd_test, how='inner', left_index=True, right_index=True)
        df_test.drop(columns = vector_col_name_doc2vec + vector_col_name_word2vec + vector_col_name_rdf2vec, inplace=True)

        X_train.drop(columns=['doc2vec_vector_ent1','word2vec_vector_ent1','rdf2vec_vector_ent1','doc2vec_vector_ent2','word2vec_vector_ent2','rdf2vec_vector_ent2'], inplace=True)
        df_model_train = pd.merge(X_train, df_train, how='inner', left_on='entity_id_wiki_1', right_on='entity_id')
        df_model_train.drop(columns=['entity_id'], inplace=True)

        len(df_model_train)
        print(df_model_train.head())

        df_model_train = pd.merge(df_model_train, df_train, how='inner', left_on='entity_id_wiki_2', right_on='entity_id')
        df_model_train.drop(columns=['entity_id'], inplace=True)
        df_model_train = df_model_train.drop_duplicates(subset=['entity_id_wiki_1','entity_id_wiki_2'])

        print(training_indices)
        df_model_train.index = training_indices
        print(df_model_train.head())




        X_test.drop(columns=['doc2vec_vector_ent1','word2vec_vector_ent1','rdf2vec_vector_ent1','doc2vec_vector_ent2','word2vec_vector_ent2','rdf2vec_vector_ent2'], inplace=True)
        df_model_test = pd.merge(X_test, df_test, how='inner', left_on='entity_id_wiki_1', right_on='entity_id')
        df_model_test.drop(columns=['entity_id'], inplace=True)


        df_model_test = pd.merge(df_model_test, df_test, how='inner', left_on='entity_id_wiki_2', right_on='entity_id')
        df_model_test.drop(columns=['entity_id'], inplace=True)
        df_model_test = df_model_test.drop_duplicates(subset=['entity_id_wiki_1','entity_id_wiki_2'])

        print(testing_indices)
        df_model_test.index = testing_indices
        print(df_model_test.head())

        df_train = pd.concat([df_model_train, y_train], axis=1)
        df_test = pd.concat([df_model_test, y_test], axis=1)

        df_model = pd.concat([df_train, df_test])

        print(df_model.head())

        print(len(df_model))
        #df_model.to_pickle('svd_train_test.pkl')


        df = df[['entity_id_wiki_1', 'entity_id_wiki_2','label']]

        df_model_train = pd.merge(df_model_train, df, how='inner', left_on=['entity_id_wiki_1','entity_id_wiki_2'], right_on=['entity_id_wiki_1','entity_id_wiki_2'])
        y_train = df_model_train['label']
        y_train=y_train.astype('int')

        df_model_train = df_model_train.drop(['label'], axis=1)


        df_model_test = pd.merge(df_model_test, df, how='inner', left_on=['entity_id_wiki_1','entity_id_wiki_2'], right_on=['entity_id_wiki_1','entity_id_wiki_2'])
        y_test = df_model_test['label']
        y_test=y_test.astype('int')

        df_model_test.to_csv('pca_train_test_split.csv', encoding='utf-8')
        df_model_test = df_model_test.drop(['label'], axis=1)

        ################ Model Training ########################

        X_train = df_model_train.drop(['entity_id_wiki_1','entity_id_wiki_2'], axis=1)
        X_test = df_model_test.drop(['entity_id_wiki_1','entity_id_wiki_2'], axis=1)


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

        df_results.to_excel('results/svd_nw1_' + file_name.replace('.pkl','.xlsx'), index=False)