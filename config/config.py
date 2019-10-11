import os
from pathlib import Path

# path to base directory
BASE_DIR = str(Path(os.path.dirname(os.path.abspath(__file__))).parent)

# directory where data is present
DATA_DIR = 'data'

# directory where processed xml dumps are present
DATA_DUMPS_DIR = 'data_dumps'

# directory where processed dumps are present
PROCESSED_DUMPS_DIR = 'processed_dumps'

# directory where untared xml dumps are present
UNTARED_DUMPS_DIR = 'extracted_dumps'

# directory where extracted text will be stored
EXTRACTED_TEXT_DIR = 'extracted_text'

# directory where processed dumps are present
EXTRACTED_PROCESSED_DUMPS_DIR = 'extracted_processed_dumps'

# directory where doc2vec model and code is present
DOC2VEC_DIR = 'doc2vec'

# directory where rdf2vec model and code is present
RDF2VEC_DIR = 'rdf2vec'

# directory where word2vec model and code is present
WORD2VEC_DIR = 'word2vec'

# path to stats directory
STATS_DIR = 'stats_parts'

# path to labels directory
LABELS_DIR = 'labels_full_en_ds/labels/'

# gold standard directory
GS_DIR = 'gold_standard'

# gold standard entities labels
GS_ENTITIES_LABELS = 'gold_standard_entities_labels'


#correspondances directory
CORRESPONDANCES_DIR = 'correspondances'

# gold standard labels file
GS_ENTITIES_LABELS_FILE = 'gs_entities_labels_wo_lyrics.pkl'

# directory where knowledge graphs (combinations of all ttl files should be stored)
KG_DIR = 'knowledge_graphs' 

# labels mapping directory
LB_MAP_DIR = 'labels_mapping'

# folder where knowledge graphs after revision of labels will be saved
REVISED_KG_DIR = 'revised_kgs'

# folder when final merged knowledge graph will be saved for random walks
MERGED_KG_RW = 'kg_for_random_walks'

# file of the final merged graph for random walks
MERGED_KG_RW_FILE_NAME = 'merged_kg.ttl'

#vector length
EMBEDDING_VECTOR_LENGTH = 300

# folder where walks will be stored
WALKS_DIR = 'graph_walks'

############## CONFIGURATION FOR DOC2Vec MODEL #################
DOC2VEC_EPOCHS = 6


############## CONFIGURATION FOR WORD2Vec MODEL #################
WORD2VEC_EPOCHS = 6


############## CONFIGURATION FOR RDF2Vec MODEL #################
RDF2VEC_EPOCHS = 6




############### LABELS DATABASE CONFIGURATION (please input DSN name)#####################
INS_LABELS_DB = 'instance_matcher_db'
INS_DUPL_LABELS_DB = 'instance_matcher_db'

CAT_LABELS_DB = 'instance_matcher_db'
CAT_DUPL_LABELS_DB = 'instance_matcher_db'

PROP_LABELS_DB = 'instance_matcher_db'
PROP_DUPL_LABELS_DB = 'instance_matcher_db'

CLASS_LABELS_DB = 'instance_matcher_db'
CLASS_DUPL_LABELS_DB = 'instance_matcher_db'


############# Configuration for Classification model ###############################
# NB = Naive Bayes, DT = Decision Trees, LR= Logistic Regression SVM = Support Vector Machines, RF = Random Forest, XGB = XGBoost

#Classification model to use
CLASSIFICATION_MODEL = 'XGB'


############# Configuration for Ensembles ###############################
# CN = Concatenate, AVG = Average
# PCA1 = Principle Component Analysis (Network 1), SVD1 = Singular Value Decomposition(Network 1), AE1 = Autoencoder Network(Network 1)
# PCA2 = Principle Component Analysis (Network 2), SVD2 = Singular Value Decomposition(Network 2), AE2 = Autoencoder Network(Network 2) 

# Ensemble approach
ENSEMBLE_APPROACH = 'CN'