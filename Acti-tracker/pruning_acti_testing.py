import os
from copy import deepcopy
import math
import split_acti
from keras.models import load_model
from keras.optimizers import Adam
from keras.utils import to_categorical
import numpy as np
from numpy.random import seed
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout
from sklearn.metrics import accuracy_score
from numpy import linalg as LA
import tensorflow as tf


rate = 1
pca_dims = 30
X_train, X_test, y_train, y_test = split_acti.main(pca_dims, rate, test_data_ratio=0.3)

y = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for compress_rate in y:
    # compress_rate = 0.3
    X_train, X_test, y_train, y_test = split_acti.main(pca_dims, rate, test_data_ratio=0.3)
    moderated = str(int(20 // rate))

    # print(X_train.shape)
    y_train = y_train - 1
    y_test = y_test - 1

    model_path = 'model/' + moderated + 'Hz/' + 'pruned_pca=' + str(pca_dims) + '_compress_rate=' + str(compress_rate) + '.hdf5'
    model2 = load_model(model_path)
    print(model2.summary())

    pred_test = model2.predict(np.expand_dims(X_test, axis=2), batch_size=32)
    # print("------ TEST ACCURACY ------")
    testing_acc = (accuracy_score(y_test, np.argmax(pred_test, axis=1)))

    print(compress_rate, testing_acc)