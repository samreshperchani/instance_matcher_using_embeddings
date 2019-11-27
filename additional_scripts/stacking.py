import pandas as pd
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
import numpy as np
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression


model_type = SVC(kernel='linear')#RandomForestClassifier(n_estimators=100, max_depth=2, random_state=522) #DecisionTreeClassifier(criterion = "gini", random_state = None,max_depth=4, min_samples_leaf=50)

def Stacking(model, X, y, n_fold):
        cv = KFold(n_splits=n_fold, random_state=42, shuffle=False)
        model_features= np.empty((0,1),int)
        actual_label = np.empty((0,1),int)

        for train_index, test_index in cv.split(X):
                print("Train Index: ", train_index)
                print("Test Index: ", test_index)
                
                X_train, X_test, y_train, y_test = X.loc[train_index], X.loc[test_index], y.loc[train_index], y.loc[test_index]
                model.fit(X_train, y_train)

                print(type(model.predict(X_test)))
                model_features = np.append(model_features,model.predict(X_test))
                actual_label = np.append(actual_label,(y_test))
                #actual_label.append(y_test)

        return model_features, actual_label

'''
def Stacking(model,train,y,test,n_fold):
   folds=StratifiedKFold(n_splits=n_fold,random_state=1)
   test_pred=np.empty((test.shape[0],1),float)
   train_pred=np.empty((0,1),float)

   for train_indices,val_indices in folds.split(train,y.values):
      x_train,x_val=train.iloc[train_indices],train.iloc[val_indices]
      y_train,y_val=y.iloc[train_indices],y.iloc[val_indices]

      model.fit(X=x_train,y=y_train)
      train_pred=np.append(train_pred,model.predict(x_val))
      test_pred=np.append(test_pred,model.predict(test))
    return test_pred.reshape(-1,1),train_pred
'''

df = pd.read_pickle('dataset/dataset_all_entities_concat.pkl')

doc2vec_cols = ['entity_id_wiki_1' , 'entity_id_wiki_2', 'label'] + ['doc2vec_ent1_' +str(i) for i in range(0,300)] + ['doc2vec_ent2_' + str(i) for i in range(0,300)] 
word2vec_cols = ['entity_id_wiki_1' , 'entity_id_wiki_2', 'label'] + ['word2vec_ent1_' +str(i) for i in range(0,300)] + ['word2vec_ent2_' + str(i) for i in range(0,300)] 
rdf2vec_cols = ['entity_id_wiki_1' , 'entity_id_wiki_2', 'label'] + ['rdf2vec_ent1_' +str(i) for i in range(0,300)] + ['rdf2vec_ent2_' + str(i) for i in range(0,300)] 

df_doc2vec = df[doc2vec_cols]
df_word2vec = df[word2vec_cols]
df_rdf2vec = df[rdf2vec_cols]


print("********************DOC2Vec*****************")

Y = df_doc2vec.label
Y = Y.astype('int')
X = df_doc2vec.drop(['entity_id_wiki_1','entity_id_wiki_2','label'], axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=1234)

print(len(X_train), ' ', len(X_test), ' ', len(y_train), ' ', len(y_test) )


#print('Training Labels: ', y_train.value_counts())
#print('Testing Labels: ', y_test.value_counts())



X_train['label'] = y_train

df_pos_class_count = len(X_train[X_train.label==1])
df_neg_class_count = len(X_train[X_train.label==0])


print('***********balancing***********')
df_not_match = X_train[X_train.label==0]
df_match = X_train[X_train.label==1]

df_not_matched_upsampled = resample(df_not_match, replace=True, n_samples=max(df_pos_class_count,df_neg_class_count), random_state=123) 
df_upsampled = pd.concat([df_not_matched_upsampled, df_match])
X_train = df_upsampled.reset_index(drop=True)

#print(X_train['label'].value_counts())
#print(y_train.value_counts())

y_train = X_train['label']
X_train.drop(columns = ['label'], inplace=True)



print('************Logistic Regression***********')
model1 = model_type #GaussianNB() #DecisionTreeClassifier(criterion = "gini", random_state = None,max_depth=4, min_samples_leaf=50)

model_features_1 ,actual_label = Stacking(model=model1, n_fold=10, X=X, y=Y)

#train_pred1 = pd.DataFrame(train_pred1)
#test_pred1 = pd.DataFrame(test_pred1)

model_features_1 = pd.DataFrame(model_features_1, columns=['model_1'])


model1.fit(X_train, y_train)

pred_model_1 = pd.DataFrame(model1.predict(X_test), columns =['model_1'])



# How's our accuracy?
print('Accuracy: ', accuracy_score(y_test, pred_model_1))
print('Precision: ', precision_score(y_test, pred_model_1))
print('Recall: ', recall_score(y_test, pred_model_1))
print('F-1: ', f1_score(y_test, pred_model_1))

print('Confusion Matrix:', confusion_matrix(y_test, pred_model_1))



print("'''''''''******************************************Word2Vec***********************")
#print(df_word2vec['label'].value_counts())


#print(df_word2vec.columns)


Y = df_word2vec.label
Y = Y.astype('int')
X = df_word2vec.drop(['entity_id_wiki_1','entity_id_wiki_2','label'], axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=1234)
print(len(X_train), ' ', len(X_test), ' ', len(y_train), ' ', len(y_test) )


print('Training Labels: ', y_train.value_counts())
print('Testing Labels: ', y_test.value_counts())


X_train['label'] = y_train

df_pos_class_count = len(X_train[X_train.label==1])
df_neg_class_count = len(X_train[X_train.label==0])


print('***********balancing***********')
df_not_match = X_train[df_word2vec.label==0]
df_match = X_train[df_word2vec.label==1]

df_not_matched_upsampled = resample(df_not_match, replace=True, n_samples=max(df_pos_class_count,df_neg_class_count), random_state=123) 
df_upsampled = pd.concat([df_not_matched_upsampled, df_match])
X_train = df_upsampled.reset_index(drop=True)

#print(X_train['label'].value_counts())
#print(y_train.value_counts())

y_train = X_train['label']
X_train.drop(columns = ['label'], inplace=True)



print('************Logistic Regression***********')
model2 = model_type #DecisionTreeClassifier(criterion = "gini", random_state = None,max_depth=4, min_samples_leaf=50)
#model2.fit(X_train, y_train)

#y_pred = model1.predict(X_test)

model_features_2 ,actual_label = Stacking(model=model2, n_fold=10, X=X, y=Y)

#train_pred1 = pd.DataFrame(train_pred1)
#test_pred1 = pd.DataFrame(test_pred1)


model_features_2 = pd.DataFrame(model_features_2, columns=['model_2'])



model2.fit(X_train, y_train)

pred_model_2 = pd.DataFrame(model2.predict(X_test), columns =['model_2'])




# How's our accuracy?
print('Accuracy: ', accuracy_score(y_test, pred_model_2))
print('Precision: ', precision_score(y_test, pred_model_2))
print('Recall: ', recall_score(y_test, pred_model_2))
print('F-1: ', f1_score(y_test, pred_model_2))

print('Confusion Matrix:', confusion_matrix(y_test, pred_model_2))


print("*********************RDF2Vec******************************")
#print(df_rdf2vec['label'].value_counts())


#print(df_rdf2vec.columns)


Y = df_rdf2vec.label
Y = Y.astype('int')
X = df_rdf2vec.drop(['entity_id_wiki_1','entity_id_wiki_2','label'], axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=1234)
print(len(X_train), ' ', len(X_test), ' ', len(y_train), ' ', len(y_test) )


print('Training Labels: ', y_train.value_counts())
print('Testing Labels: ', y_test.value_counts())


X_train['label'] = y_train

df_pos_class_count = len(X_train[X_train.label==1])
df_neg_class_count = len(X_train[X_train.label==0])


print('***********balancing***********')
df_not_match = X_train[X_train.label==0]
df_match = X_train[X_train.label==1]

df_not_matched_upsampled = resample(df_not_match, replace=True, n_samples=max(df_pos_class_count,df_neg_class_count), random_state=123) 
df_upsampled = pd.concat([df_not_matched_upsampled, df_match])
X_train = df_upsampled.reset_index(drop=True)

#print(X_train['label'].value_counts())
#print(y_train.value_counts())

y_train = X_train['label']

X_train.drop(columns = ['label'], inplace=True)


print('************Logistic Regression***********')
model3 = model_type #DecisionTreeClassifier(criterion = "gini", random_state = None,max_depth=4, min_samples_leaf=50)
#model3.fit(X_train, y_train)

#y_pred = model1.predict(X_test)

model_features_3 ,actual_label = Stacking(model=model2, n_fold=10, X=X, y=Y)

#train_pred1 = pd.DataFrame(train_pred1)
#test_pred1 = pd.DataFrame(test_pred1)


model_features_3 = pd.DataFrame(model_features_3, columns=['model_3'])


actual_labels_train = pd.DataFrame(actual_label)


model3.fit(X_train, y_train)

pred_model_3 = pd.DataFrame(model3.predict(X_test), columns =['model_3'])



# How's our accuracy?
print('Accuracy: ', accuracy_score(y_test, pred_model_3))
print('Precision: ', precision_score(y_test, pred_model_3))
print('Recall: ', recall_score(y_test, pred_model_3))
print('F-1: ', f1_score(y_test, pred_model_3))

print('Confusion Matrix:', confusion_matrix(y_test, pred_model_3))

print("******************Ensembles**********************")

model_ensembles = model_type

model_ensembles.fit(pd.concat([model_features_1, model_features_2, model_features_3],axis=1) ,actual_labels_train)

y_pred = model_ensembles.predict(pd.concat([pred_model_1, pred_model_2, pred_model_3],axis=1))

# How's our accuracy?
print('Accuracy: ', accuracy_score(y_test, y_pred))
print('Precision: ', precision_score(y_test, y_pred))
print('Recall: ', recall_score(y_test, y_pred))
print('F-1: ', f1_score(y_test, y_pred))

print('Confusion Matrix:', confusion_matrix(y_test, y_pred))

