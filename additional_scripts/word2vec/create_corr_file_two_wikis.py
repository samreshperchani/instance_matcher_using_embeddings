import gensim
from gensim.models import Word2Vec
from gensim.similarities.index import AnnoyIndexer
import re

wiki_1 = 'gameofthrones'
wiki_2 = 'game--of--thrones'
threshold = 0.75
FILE_DIR = 'model'
regexp = re.compile(r'(http://dbkwik.webdatacommons.org/)(.)+(/resource/)')
regex_file = re.compile(r'(http://dbkwik.webdatacommons.org/)(.)+(/resource/File:)')
regex_cat = re.compile(r'(http://dbkwik.webdatacommons.org/)(.)+(/resource/Category:)')
regex_wiki1 = re.compile("(http://dbkwik.webdatacommons.org/)"+ wiki_1)
regex_wiki2 = re.compile("(http://dbkwik.webdatacommons.org/)"+ wiki_2)



model = Word2Vec.load(FILE_DIR + '/' + 'word2vec_commons.model')
total_words = len(model.wv.vocab)
print('Total Words: ', total_words)

print('Creating Index')
#model_indexer = AnnoyIndexer(model, 2)
print('Index Creation done')


words = model.wv.vocab
filtered_vocab_first_wiki = []
filtered_vocab_second_wiki = []
for key in words.keys():
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
with open(FILE_DIR + '/' +'corrspondance_two_wikis.csv','wb+') as cor_file:
    for word_wiki_1 in filtered_vocab_first_wiki:
        for word_wiki_2 in filtered_vocab_second_wiki:
            count += 1
            print('Count: ', count,' Processing Word: ', word_wiki_1)
            score = model.similarity(word_wiki_1, word_wiki_2) #indexer = model_indexer)
            
            if score >= threshold:
                cor_file.write(word_wiki_1.encode("utf-8") + ','.encode("utf-8") + word_wiki_2.encode("utf-8") + ','.encode("utf-8") + str(score).encode("utf-8") + '\n'.encode("utf-8"))
cor_file.close()