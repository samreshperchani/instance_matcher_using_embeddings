from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import os
import datetime
import pandas as pd
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
DATA_DIR = 'rdf_random_walks_doc2vec'

class LabeledDocs(object):
    def __init__(self, doc_list):
        self.docs_path = doc_list
        self.iter_cnt = 1
        self.proc_cnt = 0
        #self.cols = ['subject', 'tokens']
    def __iter__(self):
        print('Starting Iteration: ', self.iter_cnt)
        self.proc_cnt = 0
        for f_name in os.listdir(DATA_DIR):
            self.proc_cnt = self.proc_cnt + 1
            print(str(datetime.datetime.now()),': Iteration: ', self.iter_cnt,' Count: ', self.proc_cnt ,', Processing Wiki: ', f_name)
            if f_name.endswith('.csv'):
                with open(DATA_DIR + '/' + f_name, 'r', encoding='utf-8') as train_data:
                    lines = train_data.readlines(100000)
                    while lines:
                        for line in lines:
                            
                            dtls = line.replace('\"',"").split(",", 1)
                            if(len(dtls)>1):
                                yield TaggedDocument(words=dtls[1].split("~EON~"), tags=[dtls[0]])
                        lines = train_data.readlines(100000)


                #reader = pd.read_csv(DATA_DIR + '/' + f_name, encoding='utf-8', chunksize=100000, delimiter=',', skiprows=1, names = ['details'])
                #for df in reader:
                #    df = df.fillna('')
                #    for index, row in df.iterrows():
                #        dtls = row['details'].split("~EON~", 1)
                #        if(len(dtls)>1):
                        #tokens = row['tokens']
                            #print(dtls)
                 #           yield TaggedDocument(words=dtls[1].split("~EON~"), tags=[dtls[0]])
                
        self.iter_cnt = self.iter_cnt + 1


corpus_data = LabeledDocs(DATA_DIR)

model = Doc2Vec(vector_size=300, window=10, min_count=1, sample=1e-4, negative=5, workers=27)


print('Process started: ', str(datetime.datetime.now()))

print('building vocabulary...')
model.build_vocab(corpus_data)

print('training model...')
model.train(corpus_data, total_examples=model.corpus_count, epochs=15)

print('Process End: ', str(datetime.datetime.now()))

print('saving model...')
model.save(DATA_DIR + '/' + 'doc2vec_RDF.model')