# import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import split_acti
import tensorflow
import csv
import time
from numpy.random import seed
import pickle
from sklearn.metrics import accuracy_score
from keras.models import load_model


if __name__ == '__main__':

    seed(2020)

    moderated = str(20)
    pca_dims = 30
    compress_rate = 0.9

    X_train, X_test, y_train, y_test = split_acti.main(pca_dims, 1, 0.3)

    # model = pickle.load(open('model/' + moderated + 'Hz/pruned/' + 'pickled_' + 'decompressed_pca=' + str(
    #     pca_dims) + '_compress_rate=' + str(compress_rate) + '.hdf5', 'rb'))

    model = load_model('model/' + moderated + 'Hz/pruned/' + 'decompressed_pca=' + str(
        pca_dims) + '_compress_rate=' + str(compress_rate) + '.hdf5')

    print(model.summary())

    y_train = y_train - 1
    y_test = y_test - 1
    # print(moderated)
    # print(pca_dims,compress_rate)

    pred_test = model.predict(np.expand_dims(X_test, axis=2), batch_size=32)
    print("------ TEST ACCURACY ------")
    testing_acc = (accuracy_score(y_test, np.argmax(pred_test, axis=1)))
    print(testing_acc)
    # print((confusion_matrix(y_test, np.argmax(pred_test, axis=1))))

    pred_train = model.predict(np.expand_dims(X_train, axis=2), batch_size=32)
    print("------ TRAIN ACCURACY ------")
    training_acc = (accuracy_score(y_train, np.argmax(pred_train, axis=1)))
    print(training_acc)
    # print((confusion_matrix(y_train, np.argmax(pred_train, axis=1))))

    # model.dense_3.shape()




    #
    # print("---------------------------------------------------------------------------------------------------")
    # n_classes = 6
    #
    # # Convert to one hot encoding vector
    # y_train_dynamic_oh = np.eye(n_classes)[y_train]
    #
    # model.fit(np.expand_dims(X_train, axis=2), y_train_dynamic_oh,
    #           batch_size=32, epochs=30, verbose=2, validation_split=0.2)
    #
    # print(model.summary())
    #
    # # print(moderated)
    # # print(pca_dims,compress_rate)
    #
    # pred_test = model.predict(np.expand_dims(X_test, axis=2), batch_size=32)
    # print("------ TEST ACCURACY ------")
    # testing_acc = (accuracy_score(y_test, np.argmax(pred_test, axis=1)))
    # print(testing_acc)
    # # print((confusion_matrix(y_test, np.argmax(pred_test, axis=1))))
    #
    # pred_train = model.predict(np.expand_dims(X_train, axis=2), batch_size=32)
    # print("------ TRAIN ACCURACY ------")
    # training_acc = (accuracy_score(y_train, np.argmax(pred_train, axis=1)))
    # print(training_acc)
    # # print((confusion_matrix(y_train, np.argmax(pred_train, axis=1))))

    # print(model.get_weights())
