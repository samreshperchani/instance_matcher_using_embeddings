import gensim
from gensim.models import Doc2Vec
from gensim.similarities.index import AnnoyIndexer
import re

threshold = 0.5
FILE_DIR = 'model'
regexp = re.compile(r'(http://dbkwik.webdatacommons.org/)(.)+(/resource/)')
regex_file = re.compile(r'(http://dbkwik.webdatacommons.org/)(.)+(/resource/File:)')
regex_cat = re.compile(r'(http://dbkwik.webdatacommons.org/)(.)+(/resource/Category:)')


model = Doc2Vec.load(FILE_DIR + '/' + 'doc2vec_commons_9282.model')
total_docs = len(model.docvecs.doctags)
print('Total Docs: ', total_docs)

print('Creating Index')
model_indexer = AnnoyIndexer(model)
print('Index Creation done')


tags = model.docvecs.doctags
filtered_vocab = []
for key in tags:
    if regexp.search(key):#key.startswith('<http://dbkwik.webdatacommons.org/resource'):
        if not regex_file.search(key):
            if not regex_cat.search(key):
                filtered_vocab.append(key)
        
print('Filtered Tags: ', len(filtered_vocab))


count = 0
with open(FILE_DIR + '/' +'corrspondance.csv','wb+') as cor_file:
    for tag in filtered_vocab:
        count += 1
        print('Count: ', count,' Processing Doc: ', tag)
        similarties = model.most_similar(tag, topn=10, indexer = model_indexer)

        for doc_scores in similarties:
            sim_doc = doc_scores[0]
            score = doc_scores[1]

            if regexp.search(sim_doc):#sim_word.startswith('<http://dbkwik.webdatacommons.org/resource'):
                if not regex_file.search(sim_doc):
                    if not regex_cat.search(sim_doc):
                        if score >= threshold:
                            cor_file.write(tag.encode("utf-8") + ','.encode("utf-8") + sim_doc.encode("utf-8") + ','.encode("utf-8") + str(score).encode("utf-8") + '\n'.encode("utf-8"))
cor_file.close()