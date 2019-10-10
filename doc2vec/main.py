from train_model import Train_DOC2Vec


# create object of a class
train_model = Train_DOC2Vec()

# call function to pre-process long abstracts
train_model.pre_process_long_abstracts()

# call function to train model
train_model.train_model()


