import os
from doc2vec.doc2vec import DOC2Vec
from rdf2vec.rdf2vec import RDF2Vec
from word2vec.word2vec import WORD2Vec
from dbkwik.dbkwik_utils import DBKWIK_UTILS
from ensemble.ensemble_learning import ENSEMBLE_LEARNING
from classification.classification import CLASSIFICATION
import config
import pandas as pd
from oaei.oaei import OAEI

print(config.BASE_DIR)

# extract path to base directory
BASE_DIR = config.BASE_DIR

# extract path to data directory
DATA_DIR = config.DATA_DIR

# model to use for classification
CLASSIFICATION_MODEL= config.CLASSIFICATION_MODEL


oaei_file_formater = OAEI()

print('***** Extracting Data Dumps ************')
#os.system('python utilities/untar_dumps.py')


print('***** Executing RDF2Vec model ************')
#os.system('python rdf2vec/main.py')

print('***** Executing DOC2Vec model ************')
#os.system('python doc2vec/main.py')

print('***** Executing Word2Vec model ************')
#os.system('python word2vec/main.py')


dbkwik_utils = DBKWIK_UTILS()

#df_test_set = dbkwik_utils.extract_instance_vectors('130814~en~gameofthrones', '1622892~en~thrones-of-game')

df_training_set = dbkwik_utils.get_training_set()

print(df_training_set['label'].unique())

df_test_set = dbkwik_utils.get_test_set('130814~en~gameofthrones', '1622892~en~thrones-of-game')

ensemble_ds = ENSEMBLE_LEARNING()
print('getting esemble dataset')
# get ensemble dataset
X_train, y_train, X_test = ensemble_ds.get_dataset(df_training_set, df_test_set)

print(X_train.head())
print(y_train.head())
print(X_test.head())

print('running classifier')
# get ensemble dataset
classifier = CLASSIFICATION()
y_pred = classifier.run_classification(X_train, y_train, X_test, CLASSIFICATION_MODEL)
y_pred = pd.DataFrame(y_pred)
y_pred.columns = ['label']


df_test_set = pd.concat([X_test, y_pred], axis=1)

print(df_test_set.head())

print('***************Final set************')
df_final_set = df_test_set[df_test_set['label']==1.0]

print('***************Generating OAEI File************')
oaei_file = oaei_file_formater.generate_file_oaei_format(df_final_set)

print('***************Writing output to file************')
oaei_file.write("dbkwik_output.xml", encoding='utf-8',  xml_declaration=True, method = 'xml')