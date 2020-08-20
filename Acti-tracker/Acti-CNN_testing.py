import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import argparse
import split_acti
import pickle
import time
from numpy.random import seed
from sklearn.metrics import accuracy_score
import tensorflow


def main(pca_dims, compress_rate, s_rate):
    rate = 1
    seed(2020)
    X_train, X_test, y_train, y_test = split_acti.main(pca_dims, rate, 0.3)
    moderated = str(int(20 // rate))

    start_time = time.time()

    model = pickle.load(open('model/' + moderated + 'Hz/pickled/' + 'pickled_' + 'decompressed_pca=' + str(
                    pca_dims) + '_compress_rate=' + str(compress_rate) + '_sparse_rate=' + str(s_rate) + '.hdf5', 'rb'))

    load_time = time.time() - start_time

    # y_train = y_train - 1
    y_test = y_test - 1
    # print(y_test.shape)

    start_time = time.time()
    pred_test = model.predict(np.expand_dims(X_test, axis=2), batch_size=32)
    # print("------ TEST ACCURACY ------")
    testing_acc = (accuracy_score(y_test, np.argmax(pred_test, axis=1)))
    # print(testing_acc)
    # print((confusion_matrix(y_test, np.argmax(pred_test, axis=1))))
    inference_time = (time.time() - start_time)

    print("Accuracy, Load Time, Inference Time")
    print(testing_acc, load_time, inference_time)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN testing on WISDM')
    parser.add_argument("--pca", default=30, type=int, help="pca dimensions: [7, 8, 10, 15, 20, 30]")
    parser.add_argument("--c_rate", default=0.3, type=float, help="Compression rate [0.1 - 0.9] with an increment of 0.1")
    parser.add_argument("--s_rate", default=0.3, type=float, help="Sparsity rate [0.1 - 0.9] with an increment of 0.1")

    args = parser.parse_args()

    main(args.pca, args.c_rate, args.s_rate)



