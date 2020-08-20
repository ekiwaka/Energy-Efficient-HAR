import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from HAPT import split_HAPT
from numpy.random import seed
from keras.models import load_model
from sklearn.metrics import accuracy_score

# from sklearn.metrics import confusion_matrix

moderated = str(50)
pca_dims = 20

seed(2020)
X_train, X_test, y_train, y_test = split_HAPT.main(pca_dims)

y_train = y_train - 1
y_test = y_test - 1

# model_path = 'model/' + moderated + 'Hz/' + 'pca_testing_bigger_cnn' + str(pca_dims) + '.hdf5'
model_path = 'model/' + moderated + 'Hz/' + 'pca_big' + str(pca_dims) + '_pruned.hdf5'
model = load_model(model_path)
# model = load_model('compressed_true_0.9_bigs.h5')

pred_train = model.predict(np.expand_dims(X_train, axis=2), batch_size=32)
print("------ TRAIN ACCURACY ------")
print((accuracy_score(y_train, np.argmax(pred_train, axis=1))))
# print((confusion_matrix(y_train, np.argmax(pred_train, axis=1))))

pred_test = model.predict(np.expand_dims(X_test, axis=2), batch_size=32)
print("------ TEST ACCURACY ------")
print((accuracy_score(y_test, np.argmax(pred_test, axis=1))))
# print((confusion_matrix(y_test, np.argmax(pred_test, axis=1))))
