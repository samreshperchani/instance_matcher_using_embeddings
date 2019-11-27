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

df = pd.read_pickle('dataset/doc2vec_dataset.pkl')

print(df.head())

print(df['label'].value_counts())


print(df.columns)


# Separate majority and minority classes
#df_not_match = df[df.label==0]
#df_match = df[df.label==1]

# Upsample minority class
#df_not_matched_upsampled = resample(df_not_match, replace=True, n_samples=9670, random_state=123) 
#df_not_matched_upsampled = df_not_match

# Combine majority class with upsampled minority class
#df_upsampled = pd.concat([df_not_matched_upsampled, df_match])
#df_upsampled = df_upsampled.reset_index(drop=True)


#df_upsampled = df_upsampled.reset_index(drop=True)

#print(df_upsampled['label'].value_counts())

#print(df_upsampled.head())

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



print('***********Dummy Classifier*********')
nb_model = DummyClassifier(strategy='stratified', random_state=None, constant=None)
nb_model.fit(X_train, y_train)

y_pred = nb_model.predict(X_test)

# How's our accuracy?
print('Accuracy: ', accuracy_score(y_test, y_pred))
print('Precision: ', precision_score(y_test, y_pred))
print('Recall: ', recall_score(y_test, y_pred))
print('F-1: ', f1_score(y_test, y_pred))

print('Confusion Matrix:', confusion_matrix(y_test, y_pred))

'''
print('**************** Rule based Classifier***********')
clf = lw.RIPPER()
train = pd.concat([X_train, y_train], axis=1)
train['label'] = train['label'].map(str)
print(train.head())
clf.fit(train, class_feat='label', pos_class=1, random_state=42)

precision = clf.score(X_test, y_test, precision_score)
recall = clf.score(X_test, y_test, recall_score)
cond_count = clf.ruleset_.count_conds()
print(f'precision: {precision} recall: {recall} conds: {cond_count}')


print(type(clf.ruleset_))
print(clf.ruleset_)
clf.ruleset_.out_pretty()

'''


print('************Decision Tree***********')
clf_gini = DecisionTreeClassifier(criterion = "gini", splitter = 'random', random_state = 0, max_depth=4, min_samples_leaf=50)
clf_gini.fit(X_train, y_train)

y_pred = clf_gini.predict(X_test)

print(clf_gini.score(X_train, y_train))
print(clf_gini.score(X_test, y_test))

# How's our accuracy?
print('Accuracy: ', accuracy_score(y_test, y_pred))
print('Precision: ', precision_score(y_test, y_pred))
print('Recall: ', recall_score(y_test, y_pred))
print('F-1: ', f1_score(y_test, y_pred))
print('Confusion Matrix:', confusion_matrix(y_test, y_pred))


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
#train model
clf_1 = LogisticRegression(random_state=0).fit(X_train, y_train)

# Predict on training set
pred_y_1 = clf_1.predict(X_test)

print(clf_1.score(X_train, y_train))
print(clf_1.score(X_test, y_test))


# How's our accuracy?
print('Accuracy: ', accuracy_score(y_test, pred_y_1))
print('Precision: ', precision_score(y_test, pred_y_1))
print('Recall: ', recall_score(y_test, pred_y_1))
print('F-1: ', f1_score(y_test, pred_y_1))
print('Confusion Matrix:', confusion_matrix(y_test, pred_y_1))


print('************SVM***********')

svclassifier = SVC(kernel='linear', random_state=0)
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)

# How's our accuracy?
print('Accuracy: ', accuracy_score(y_test, y_pred))
print('Precision: ', precision_score(y_test, y_pred))
print('Recall: ', recall_score(y_test, y_pred))
print('F-1: ', f1_score(y_test, y_pred))

print('Confusion Matrix:', confusion_matrix(y_test, y_pred))



print('***********XGBoost*********')
xg_model = XGBClassifier(random_state=0)
xg_model.fit(X_train, y_train)

y_pred = xg_model.predict(X_test)

# How's our accuracy?
print('Accuracy: ', accuracy_score(y_test, y_pred))
print('Precision: ', precision_score(y_test, y_pred))
print('Recall: ', recall_score(y_test, y_pred))
print('F-1: ', f1_score(y_test, y_pred))
print('Confusion Matrix:', confusion_matrix(y_test, y_pred))



print('***********Random Forest*********')
clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# How's our accuracy?
print('Accuracy: ', accuracy_score(y_test, y_pred))
print('Precision: ', precision_score(y_test, y_pred))
print('Recall: ', recall_score(y_test, y_pred))
print('F-1: ', f1_score(y_test, y_pred))
print('Confusion Matrix:', confusion_matrix(y_test, y_pred))


print('***********Naive Bayes*********')
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

y_pred = nb_model.predict(X_test)

# How's our accuracy?
print('Accuracy: ', accuracy_score(y_test, y_pred))
print('Precision: ', precision_score(y_test, y_pred))
print('Recall: ', recall_score(y_test, y_pred))
print('F-1: ', f1_score(y_test, y_pred))

print('Confusion Matrix:', confusion_matrix(y_test, y_pred))




'''
print("Cross Validation")
from sklearn.model_selection import cross_val_score
xg_model = LogisticRegression()
scores = cross_val_score(xg_model, X, Y, cv=10,  scoring='f1')
print(scores)
'''