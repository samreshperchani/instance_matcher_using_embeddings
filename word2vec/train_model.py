import glob
from gensim.models import Word2Vec
import logging
import datetime
import os
import re
import gensim
import sys
from pathlib import Path
import shutil
import subprocess

# set configuration file path
config_path = os.path.dirname(os.getcwd()) + '/config' 

# add config file path to system
sys.path.append(config_path)

import config

# base directory of the solution
BASE_DIR = config.BASE_DIR

# path to data directory
DATA_DIR = config.DATA_DIR

# path where extracted text will be stored
EXTRACTED_TEXT_DIR = config.EXTRACTED_TEXT_DIR


print(gensim.models.word2vec.FAST_VERSION)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# count of epochs
WORD2VEC_EPOCHS = config.WORD2VEC_EPOCHS

# length of embedding vector
EMBEDDING_VECTOR_LENGTH = config.EMBEDDING_VECTOR_LENGTH

# path where untared xml dumps are present
UNTARED_DUMPS_DIR = config.UNTARED_DUMPS_DIR


# class to traing word2vec
class Train_Word2Vec:

    # extract text from all xml dumps
    def extract_text(self):
        '''
        if not os.path.exists(BASE_DIR + '/' + DATA_DIR + '/' +EXTRACTED_TEXT_DIR):
            os.mkdir(BASE_DIR + '/' + DATA_DIR + '/' +EXTRACTED_TEXT_DIR)
        # get list of all folders in untared dumps directory
        file_names = os.listdir(BASE_DIR + '/' + DATA_DIR + '/' + UNTARED_DUMPS_DIR + '/')
        
        for file in file_names:
            text_extraction.extract_text_from_xml(file)
            #text_extraction.extract_text_for_all_wikis()
        '''
        #ta = wikimain_parallel.TEXT_EXTRACTION()
        #ta.extract_text()
        cmd = ['python', 'wikimain_parallel.py']
        subprocess.Popen(cmd).wait()

    def train_model(self):
        # build itertor for vocabulary
        class TextLine(object):
            def __init__(self, dirname):
                self.dirname = dirname
                self.iter_count = 0
                # chunksize determines how many lines needs to be read at a time from file
                self.chunksize = 10000
        
            def __iter__(self):
                self.iter_count = self.iter_count + 1
                self.proc_count = 1
                # loop through directory and read text files one by one
                for file in glob.glob(self.dirname + '/' + DATA_DIR + '/' +  EXTRACTED_TEXT_DIR + '/*.txt'):
                    print('Processing ', file , ' iteration ', self.iter_count, 'Processed: ', self.proc_count)
                    with open(file,'r', encoding='utf-8') as text_contents:
                        file_lines = text_contents.readlines(self.chunksize)
                        while file_lines:
                            lines = []
                            # read line by line and replace new line characters
                            for line in file_lines:
                                lines.append(line.replace('\n','').replace('\r\n','').replace('\r','').strip())
                            # splot line into tokens
                            yield " ".join(lines).split(' ')
                            # read another chunk
                            file_lines = text_contents.readlines(self.chunksize)
                        self.proc_count = self.proc_count + 1
                    # close file
                    text_contents.close()
                    


        corpus_data = TextLine(BASE_DIR)

        # set model parameters
        model = Word2Vec(size=EMBEDDING_VECTOR_LENGTH, window=10, min_count=1, workers=2)

        # build model vocabulary
        model.build_vocab(corpus_data)

        print('Training the Model: ........')

        # train model
        model.train(corpus_data, total_examples=model.corpus_count, epochs=WORD2VEC_EPOCHS)

        # remove directory if already present
        if os.path.exists('model'):
            shutil.rmtree('model')

        # create model directory to save model    
        os.mkdir('model')

        # save model
        model.save('model/word2vec.model')