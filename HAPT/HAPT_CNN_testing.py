import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow
import argparse
import csv
import numpy as np
from keras.models import load_model
import split_HAPT
import keras.backend as K
import time
from numpy.random import seed
import pickle
from sklearn.metrics import accuracy_score


# from sklearn.metrics import confusion_matrix


def main(rate, pca_dims, compress_rate, sparse_rate, sensor):
    seed(2020)
    X_train, X_test, y_train, y_test = split_HAPT.main(pca_dims, rate, 0.3, sensor)
    moderated = str(int(50 // rate))
    # print(moderated, pca_dims, compress_rate)

    if sensor == 'acc':
        sensor = 'acc_'

    start_time = time.time()

    model = pickle.load(open('model/' + moderated + 'Hz/pickled/' + 'pickled_' + 'acc_decompressed_pca=' + str(
                                    pca_dims) + '_compress_rate=' + str(compress_rate) + '_sparse_rate=' + str(
                                    sparse_rate) + '.hdf5', 'rb'))

    # model = pickle.load(open('model/' + moderated + 'Hz/pruned/' + 'acc_decompressed_pca=' + str(
    #                                 pca_dims) + '_compress_rate=' + str(compress_rate) + '.hdf5', 'rb'))

    # model = load_model('model/' + moderated + 'Hz/pruned/' + 'acc_decompressed_pca=' + str(
    #                                 pca_dims) + '_compress_rate=' + str(compress_rate) + '.hdf5')

    load_time = time.time() - start_time

    # y_train = y_train - 1
    y_test = y_test - 1
    # print(moderated)
    # print(pca_dims,compress_rate)
    start_time = time.time()

    pred_test = model.predict(np.expand_dims(X_test, axis=2), batch_size=32)

    inference_time = time.time() - start_time
    testing_acc = (accuracy_score(y_test, np.argmax(pred_test, axis=1)))
    # print("------ TEST ACCURACY ------")
    # print(testing_acc)
    # # print((confusion_matrix(y_test, np.argmax(pred_test, axis=1))))
    #
    # print("------ INFERENCE TIME ------")
    # print(inference_time)

    # print("Accuracy, Load Time, Inference Time")
    # print(testing_acc, load_time, inference_time)

    # pred_train = model.predict(np.expand_dims(X_train, axis=2), batch_size=32)
    # print("------ TRAIN ACCURACY ------")
    # training_acc = (accuracy_score(y_train, np.argmax(pred_train, axis=1)))
    # print(training_acc)

    # print((confusion_matrix(y_train, np.argmax(pred_train, axis=1))))

    del model

    print(moderated, pca_dims, compress_rate, sparse_rate, testing_acc, inference_time)

    with open('testing_results_HAPT.txt', 'a', newline='') as f_out:
        writer = csv.writer(f_out)
        # writer.writerow(['Sampling rate', 'PCA dims', 'Time'])
        writer.writerow([moderated, pca_dims, compress_rate, testing_acc, inference_time])
    f_out.close()
    K.clear_session()


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='CNN testing on HAPT')
    # parser.add_argument("--pca", default=30, type=int, help="pca dimensions with gyro: [20, 30, 40, 50, 60] "
    #                                                         "without gyro: [15, 18, 21, 24, 27, 30]")
    # parser.add_argument("--rate", default=1, type=float, help="Sampling rate [10, 5, 2.5, 2, 1.25, 1]")
    # parser.add_argument("--c_rate", default=0.3, type=float, help="Compression rate [0.1 - 0.9] with an increment of 0.1")
    # parser.add_argument("--sensor", default='', type=str, help="Input acc for just accelerometer. "
    #                                                            "No input for accelerometer + gyroscope")
    #
    # args = parser.parse_args()
    #
    # main(args.pca, args.rate, args.c_rate, args.sensor)

    with open('testing_results_HAPT.txt', 'w', newline='') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(['Sampling rate', 'PCA dims', 'Compress rate', 'Accuracy', 'Inference time'])
        f_out.close()

    x = [15, 18, 21, 24, 27, 30]
    # x = [24, 27, 30]
    y = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # y = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # y2 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    s_rate = [10, 5, 2.5, 2, 1.25, 1]
    # s_rate = [1]

    for rate in s_rate:
        for pca_dims in x:
            for compress_rate in y:
                # for sparse_rate in y:
                main(rate, pca_dims, compress_rate, 0.1, 'acc')

