from train_model import WORD2VEC


# create object of a class
train_model = WORD2VEC()

'''
# extract text
train_model.extract_text()

# call function to train model
train_model.train_model()
'''

train_model.extract_vectors('130814~en~gameofthrones','1622892~en~thrones-of-game')

