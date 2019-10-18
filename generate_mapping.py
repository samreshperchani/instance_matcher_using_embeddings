from ensemble.ensemble_learning import ENSEMBLE_LEARNING
from classification.classification import CLASSIFICATION
import pandas as pd
from oaei.oaei import OAEI

el = ENSEMBLE_LEARNING()
cl = CLASSIFICATION()
xml_file_generator = OAEI()

X_train, y_train, X_test = el.get_dataset()


print(len(X_train),  ' ', len(y_train), ' ', len(X_test))

print(X_train.head())
print(X_test.head())


df_train = pd.concat([X_train, y_train], axis=1)
df_test = X_test

X_train = X_train.drop(columns = ['entity_id_wiki_1','entity_id_wiki_2'])
X_test = X_test.drop(columns = ['entity_id_wiki_1','entity_id_wiki_2'])


y_pred = cl.run_classification(X_train, y_train, X_test, 'DT')

print('******************* Final Predictions *****************')
print(y_pred)

y_test = pd.DataFrame(y_pred)
y_test.columns=['label']
print(y_test.head())

df_test = pd.concat([df_test, y_test], axis=1)

df_train = df_train[['entity_id_wiki_1','entity_id_wiki_2','label']]
df_test = df_test[['entity_id_wiki_1','entity_id_wiki_2','label']]

df_train_match = df_train[df_train['label'] == 1]
df_test_match = df_test[df_test['label'] == 1]

df_match = pd.concat([df_test_match])

print('******************* Final Matches *****************')
print(df_match.head())

xml_file_generator.generate_file_oaei_format(df_match.head(150))

print(y_pred)


