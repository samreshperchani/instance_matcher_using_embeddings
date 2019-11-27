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

df = pd.read_pickle('dataset/training_ds_v4_no_pca.pkl')

'''
all_columns = []

for col in df.columns:
    if col.startswith('rdf2vec'):
        all_columns.append(col)

all_columns.append('entity_id_wiki_1')
all_columns.append('entity_id_wiki_2')
all_columns.append('label')

df = df[all_columns]
'''

print(df['label'].value_counts())



# Separate majority and minority classes
df_not_match = df[df.label==0]
df_match = df[df.label==1]

# Upsample minority class
df_not_matched_upsampled = resample(df_not_match, replace=True, n_samples=9670, random_state=123) 
#df_not_matched_upsampled = df_not_match

# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_not_matched_upsampled, df_match])
df_upsampled = df_upsampled.reset_index(drop=True)

def shuffle(df, n=1, axis=0):
    df = df.copy()
    for _ in range(n):
        df.apply(np.random.shuffle, axis=axis)
    return df

df_upsampled = shuffle(df_upsampled)
df_upsampled = df_upsampled.reset_index(drop=True)

print(df_upsampled['label'].value_counts())

print(df_upsampled.head())

Y = df_upsampled.label
Y = Y.astype('int')
X = df_upsampled.drop(['entity_id_wiki_1','entity_id_wiki_2','label'], axis=1)

#print(X.head())

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=None)

print('Training Labels: ', y_train.value_counts())
print('Testing Labels: ', y_test.value_counts())



print('************Logistic Regression***********')
#train model
clf_1 = LogisticRegression().fit(X_train, y_train)

# Predict on training set
pred_y_1 = clf_1.predict(X_test)

# How's our accuracy?
print('Accuracy: ', accuracy_score(y_test, pred_y_1))
print('Precision: ', precision_score(y_test, pred_y_1))
print('Recall: ', recall_score(y_test, pred_y_1))
print('F-1: ', f1_score(y_test, pred_y_1))


print('************SVM***********')
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)

# How's our accuracy?
print('Accuracy: ', accuracy_score(y_test, y_pred))
print('Precision: ', precision_score(y_test, y_pred))
print('Recall: ', recall_score(y_test, y_pred))
print('F-1: ', f1_score(y_test, y_pred))



print('************Decision Tree***********')
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = None,max_depth=3, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)

y_pred = clf_gini.predict(X_test)

# How's our accuracy?
print('Accuracy: ', accuracy_score(y_test, y_pred))
print('Precision: ', precision_score(y_test, y_pred))
print('Recall: ', recall_score(y_test, y_pred))
print('F-1: ', f1_score(y_test, y_pred))

print('***********XGBoost*********')
xg_model = XGBClassifier()
xg_model.fit(X_train, y_train)

y_pred = xg_model.predict(X_test)

# How's our accuracy?
print('Accuracy: ', accuracy_score(y_test, y_pred))
print('Precision: ', precision_score(y_test, y_pred))
print('Recall: ', recall_score(y_test, y_pred))
print('F-1: ', f1_score(y_test, y_pred))


print('***********Random Forest*********')
clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=None)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# How's our accuracy?
print('Accuracy: ', accuracy_score(y_test, y_pred))
print('Precision: ', precision_score(y_test, y_pred))
print('Recall: ', recall_score(y_test, y_pred))
print('F-1: ', f1_score(y_test, y_pred))


print('********** K-Fold Cross Validation***********')
'''
kf = KFold(n_splits=10, random_state=None, shuffle=False)


for train_index, test_index in kf.split(X):
    #print('TRAIN:', train_index, 'TEST:', test_index)
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]

    print('************Logistic Regression***********')
    #train model
    clf_1 = LogisticRegression().fit(X_train, y_train)

    # Predict on training set
    pred_y_1 = clf_1.predict(X_test)

    # How's our accuracy?
    print('Accuracy: ', accuracy_score(y_test, pred_y_1))
    print('Precision: ', precision_score(y_test, pred_y_1))
    print('Recall: ', recall_score(y_test, pred_y_1))
    print('F-1: ', f1_score(y_test, pred_y_1))

'''
#print(X.loc[[8066,8067]])

#clf = SVC(kernel='linear', C=1)
#scores = cross_val_score(clf, X, Y, cv=10)
#print(scores)
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
'''

scoring = ['precision_macro', 'recall_macro']
clf = LogisticRegression()
scores = cross_validate(clf, X, Y, cv=10, scoring=scoring)
print(scores)
print(sorted(scores.keys()))
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

print(scores['test_recall_macro'])
print(scores['test_precision_macro'])

'''