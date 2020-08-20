# import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import split_HAPT
import csv
import time
from numpy.random import seed
import pickle
from sklearn.metrics import accuracy_score


if __name__ == '__main__':
    # # x = [0, 15, 18, 21, 24, 27]
    # x = [0, 20, 30, 40, 50]
    # y = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # s_rate = [10, 5, 2.5, 2, 1.25, 1]

    # with open('testing_results_pickle_test.txt', 'w', newline='') as f_out:
    #     writer = csv.writer(f_out)
    #     writer.writerow(
    #         ['Sampling rate', 'PCA dims', 'Compression rate', 'Training accuracy', 'Testing accuracy', 'Time'])
    # f_out.close()

# for rate in s_rate:
# # rate = 1
# for pca_dims in x:
# for compress_rate in y:

    rate = 1
    pca_dims = 30
    compress_rate = 0.3

    seed(2020)
    X_train, X_test, y_train, y_test = split_HAPT.main(pca_dims, rate, 0.3, '')
    moderated = str(int(50 // rate))
    if pca_dims == 0:
        pca_dims = 60

    y_train = y_train - 1
    y_test = y_test - 1

    start_time = time.time()

    model = pickle.load(open('model/' + moderated + 'Hz/pruned/' + 'pickled_' + 'decompressed_pca=' + str(
        pca_dims) + '_compress_rate=' + str(compress_rate) + '.hdf5', 'rb'))

    # print(moderated)
    # print(pca_dims,compress_rate)

    print("SEEEEEEEEEEEEEEEEEEE")

    pred_test = model.predict(np.expand_dims(X_test, axis=2), batch_size=32)
    print("------ TEST ACCURACY ------")
    testing_acc = (accuracy_score(y_test, np.argmax(pred_test, axis=1)))
    print(testing_acc)
    # print((confusion_matrix(y_test, np.argmax(pred_test, axis=1))))
    t = (time.time() - start_time)

    pred_train = model.predict(np.expand_dims(X_train, axis=2), batch_size=32)
    print("------ TRAIN ACCURACY ------")
    training_acc = (accuracy_score(y_train, np.argmax(pred_train, axis=1)))
    print(training_acc)
    # print((confusion_matrix(y_train, np.argmax(pred_train, axis=1))))

    i=0

    while i < len(X_test):
        start_time = time.time()
        lewl = model.predict_on_batch(np.expand_dims(X_test[i:i+100], axis=2))
        bas = time.time() - start_time
        training_acc = (accuracy_score(y_test[i:i+100], np.argmax(lewl, axis=1)))
        print(training_acc, bas)
        i+=101

    # with open('testing_results_pickle_test.txt', 'a', newline='') as f_out:
    #     writer = csv.writer(f_out)
    #     # writer.writerow(['Sampling rate', 'PCA dims', 'Time'])
    #     writer.writerow([moderated, pca_dims, compress_rate, training_acc, testing_acc, t])
    # f_out.close()
