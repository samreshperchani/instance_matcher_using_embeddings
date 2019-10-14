from doc2vec import DOC2Vec


# create object of a class
doc2vec = DOC2Vec()

'''
# call function to pre-process long abstracts
train_model.pre_process_long_abstracts()

# call function to train model
train_model.train_model()
'''


doc2vec.extract_vectors('130814~en~gameofthrones','1622892~en~thrones-of-game')


