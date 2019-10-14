from train_model import DOC2VEC


# create object of a class
train_model = DOC2VEC()

'''
# call function to pre-process long abstracts
train_model.pre_process_long_abstracts()

# call function to train model
train_model.train_model()
'''


train_model.extract_vectors('130814~en~gameofthrones','1622892~en~thrones-of-game')


