import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import gc
import argparse
import csv
import split_acti
import pickle
import time
from numpy.random import seed
from sklearn.metrics import accuracy_score
import tensorflow



time.sleep(2)
rate = 1
seed(2020)
# x = [7] #, 8, 10, 20, 30]
# # x = [0, 20, 30, 40, 50]
# # y = [0.1]
y = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# # y = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
# # m = 0
#
#

s_rate = 0.1
pca_dims = 7
compress_rate = 0.6
# for compress_rate in y:
#     # time.sleep(1)

X_train, X_test, y_train, y_test = split_acti.main(pca_dims, rate, 0.3)
moderated = str(int(20 // rate))

# start_time = time.time()


model = pickle.load(open('model/' + moderated + 'Hz/pickled/' + 'pickled_' + 'decompressed_pca=' + str(
                pca_dims) + '_compress_rate=' + str(compress_rate) + '_sparse_rate=' + str(s_rate) + '.hdf5', 'rb'))

# load_time = time.time() - start_time

# y_train = y_train - 1
y_test = y_test - 1
# print(y_test.shape)


start_time = time.process_time()
pred_test = model.predict(np.expand_dims(X_test, axis=2), batch_size=32)
pred_time = time.process_time() - start_time
# print("------ TEST ACCURACY ------")
testing_acc = (accuracy_score(y_test, np.argmax(pred_test, axis=1)))
# print(testing_acc)
# print((confusion_matrix(y_test, np.argmax(pred_test, axis=1))))
# inference_time = (time.time() - start_time)

# print("Accuracy, Load Time, Inference Time")
# print(testing_acc, load_time, inference_time)

print(pca_dims, compress_rate, testing_acc, pred_time)

with open('testing_results_final_wisdm.txt', 'a', newline='') as f_out:
    writer = csv.writer(f_out)
    # writer.writerow(['Sampling rate', 'PCA dims', 'Time'])
    writer.writerow([pca_dims, compress_rate, testing_acc, pred_time])
f_out.close()

gc.collect()




# if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='CNN testing on WISDM')
    # parser.add_argument("--pca", default=30, type=int, help="pca dimensions: [7, 8, 10, 15, 20, 30]")
    # parser.add_argument("--c_rate", default=0.3, type=float, help="Compression rate [0.1 - 0.9] with an increment of 0.1")
    # parser.add_argument("--s_rate", default=0.1, type=float, help="Sparsity rate [0.1 - 0.9] with an increment of 0.1")
    #
    # args = parser.parse_args()
    #
    # main(args.pca, args.c_rate, args.s_rate)

    # with open('testing_results_final_wisdm.txt', 'w', newline='') as f_out:
    #     writer = csv.writer(f_out)
    #     writer.writerow(['Sampling rate', 'PCA dims', 'Compress rate', 'Accuracy', 'Prediction time'])
    #     f_out.close()
    #
    #
    #
    # # x = [30] #[7, 8, 10, 15, 20, 30]
    # x = [7, 8, 10, 20, 30]
    # # x = [0, 20, 30, 40, 50]
    # # y = [0.1]
    # y = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # # y = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    # # m = 0
    #
    #
    # for pca_dims in x:
    #     # time.sleep(1)
    #     for compress_rate in y:
    #         main(pca_dims, compress_rate, 0.1)
    #         # time.sleep(2)

    # pca_dims = 7
    # compress_rate = 0.1
    #
    # while m <3:
    #     main(pca_dims, compress_rate, 0.1)
    #     m += 1

    # main(7, 0.1, 0.1)
    # main(7, 0.2, 0.1)
    # main(7, 0.3, 0.1)
    # main(7, 0.4, 0.1)
    # main(7, 0.5, 0.1)
    # main(7, 0.6, 0.1)
    # main(7, 0.7, 0.1)
    # main(7, 0.8, 0.1)
    # main(7, 0.9, 0.1)
    # main(8, 0.1, 0.1)
    # main(8, 0.2, 0.1)
    # main(8, 0.3, 0.1)
    # main(8, 0.4, 0.1)
    # main(8, 0.5, 0.1)
    # main(8, 0.6, 0.1)
    # main(8, 0.7, 0.1)
    # main(8, 0.8, 0.1)
    # main(8, 0.9, 0.1)
    # main(10, 0.1, 0.1)
    # main(10, 0.2, 0.1)
    # main(10, 0.3, 0.1)
    # main(10, 0.4, 0.1)
    # main(10, 0.5, 0.1)
    # main(10, 0.6, 0.1)
    # main(10, 0.7, 0.1)
    # main(10, 0.8, 0.1)
    # main(10, 0.9, 0.1)
    # main(20, 0.1, 0.1)
    # main(20, 0.2, 0.1)
    # main(20, 0.3, 0.1)
    # main(20, 0.4, 0.1)
    # main(20, 0.5, 0.1)
    # main(20, 0.6, 0.1)
    # main(20, 0.7, 0.1)
    # main(20, 0.8, 0.1)
    # main(20, 0.9, 0.1)
    # main(30, 0.1, 0.1)
    # main(30, 0.2, 0.1)
    # main(30, 0.3, 0.1)
    # main(30, 0.4, 0.1)
    # main(30, 0.5, 0.1)
    # main(30, 0.6, 0.1)
    # main(30, 0.7, 0.1)
    # main(30, 0.8, 0.1)
    # main(30, 0.9, 0.1)
    # # #
