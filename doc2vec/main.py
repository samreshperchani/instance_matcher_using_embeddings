from doc2vec import DOC2Vec


if __name__ == '__main__':
    
    # create object of a class
    doc2vec = DOC2Vec()
    
    # call function to pre-process long abstracts
    doc2vec.pre_process_long_abstracts()
    
    # call function to train model
    doc2vec.train_model()



#doc2vec.extract_vectors('130814~en~gameofthrones','1622892~en~thrones-of-game')


