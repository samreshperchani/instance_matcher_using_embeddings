from ensemble_learning import ENSEMBLE_LEARNING


el = ENSEMBLE_LEARNING()

X_train, y_train, X_test = el.get_dataset()

print(len(X_train),  ' ', len(y_train), ' ', len(X_test))

print(X_train.head())
print(X_test.head())