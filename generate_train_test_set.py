import pandas as pd
from rdf2vec.rdf2vec import RDF2Vec
from doc2vec.doc2vec import DOC2Vec
from word2vec.word2vec import Word2Vec


#initialize rdf2vec_model
rdf2vec_model = RDF2Vec()


# get rdf2vec vectors
rdf2vec_wiki_1, rdf2vec_wiki_2 = rdf2vec_model.extract_vectors('130814~en~gameofthrones','1622892~en~thrones-of-game')

print(rdf2vec_wiki_1.head())

print(rdf2vec_wiki_2.head())



#initialize doc2vec_model
doc2vec_model = DOC2Vec()

#get doc2vec vectors
doc2vec_wiki_1, doc2vec_wiki_2 = doc2vec_model.extract_vectors('130814~en~gameofthrones','1622892~en~thrones-of-game')
print(doc2vec_wiki_1.head())
print(doc2vec_wiki_2.head())


#get word2vec vectors
doc2vec_wiki_1, doc2vec_wiki_2 = rdf2vec_model.extract_vectors('130814~en~gameofthrones','1622892~en~thrones-of-game')
#print(doc2vec_wiki_1.head())
#print(doc2vec_wiki_2.head())
