import gensim
from gensim.models import Doc2Vec
from gensim.similarities.index import AnnoyIndexer
import re
from scipy import spatial

wiki_1 = 'marvelcinematicuniverse.wikia.com'
wiki_2 = 'lyrics.wikia.com'
threshold = 0.75
FILE_DIR = 'model'
regexp = re.compile(r'(http://dbkwik.webdatacommons.org/)(.)+(/resource/)')
regex_file = re.compile(r'(http://dbkwik.webdatacommons.org/)(.)+(/resource/File:)')
regex_cat = re.compile(r'(http://dbkwik.webdatacommons.org/)(.)+(/resource/Category:)')
regex_wiki1 = re.compile("(http://dbkwik.webdatacommons.org/)"+ wiki_1)
regex_wiki2 = re.compile("(http://dbkwik.webdatacommons.org/)"+ wiki_2)


print('Loading Model:')
model = Doc2Vec.load(FILE_DIR + '/' + 'doc2vec_full.model')
total_docs = len(model.docvecs.doctags)
print('Total Docs: ', total_docs)

print('Creating Index')
model_indexer = AnnoyIndexer(model, 500)
print('Index Creation done')


tags = model.docvecs.doctags
filtered_vocab_first_wiki = []
filtered_vocab_second_wiki = []
for key in tags:
    if regexp.search(key):#key.startswith('<http://dbkwik.webdatacommons.org/resource'):
        if not regex_file.search(key):
            if not regex_cat.search(key):
                if regex_wiki1.search(key):
                    filtered_vocab_first_wiki.append(key)
                elif regex_wiki2.search(key):
                    filtered_vocab_second_wiki.append(key)
        
print('Filtered Words Wiki 1: ', len(filtered_vocab_first_wiki))
print('Filtered Words Wiki 2: ', len(filtered_vocab_second_wiki))


count = 0
with open('corrspondance_two_wikis' + wiki_1 + '_' + wiki_2 + '.csv','wb+') as cor_file:
    cor_file.write("wiki_1,wiki_2,sim_score\n".encode('utf-8'))
    for word_wiki_1 in filtered_vocab_first_wiki:
        count += 1
        print('Count: ', count,' Processing Word: ', word_wiki_1)
        vec1 = model.docvecs[word_wiki_1]
        for word_wiki_2 in filtered_vocab_second_wiki:
            vec2 = model.docvecs[word_wiki_2]
            #score = model.similarity(word_wiki_1, word_wiki_2) #indexer = model_indexer)
            score = 1 - spatial.distance.cosine(vec1, vec2)
            if score >= threshold:
                cor_file.write("\"".encode("utf-8") + word_wiki_1.encode("utf-8") + "\",".encode("utf-8") + "\"".encode("utf-8") + word_wiki_2.encode("utf-8") + "\",".encode("utf-8") + str(score).encode("utf-8") + '\n'.encode("utf-8"))
cor_file.close()