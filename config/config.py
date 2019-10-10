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


#vector length
EMBEDDING_VECTOR_LENGTH = 300


############## CONFIGURATION FOR DOC2Vec MODEL #################
DOC2VEC_EPOCHS = 6
