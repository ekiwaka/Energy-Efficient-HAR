import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pickle

from numpy.random import seed
from keras.models import load_model



if __name__ == '__main__':

    # x = [0, 15, 18, 21, 24, 27]
    # y = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # s_rate = [10, 5, 2.5, 2, 1.25, 1]
    #
    # for rate in s_rate:
    #     for pca_dims in x:
    #         for compress_rate in y:
    #             seed(2020)
    #             moderated = str(int(50 // rate))
    #             print(moderated, pca_dims, compress_rate)
    #             if pca_dims == 0:
    #                 pca_dims = 30
    #
    #             model_path = 'HAPT/model/' + moderated + 'Hz/pruned/' + 'acc_decompressed_pca=' + str(
    #                 pca_dims) + '_compress_rate=' + str(compress_rate) + '.hdf5'
    #
    #             model = load_model(model_path)
    #             pickle.dump(model,
    #                         open('HAPT/model/' + moderated + 'Hz/pruned/' + 'pickled_' + 'acc_decompressed_pca=' + str(
    #                             pca_dims) + '_compress_rate=' + str(compress_rate) + '.hdf5', 'wb'))
    #             del model







    # x = [0, 15, 18, 21, 24, 27]
    x = [7, 8, 10, 15, 20, 0]
    y = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


    rate = 1
    for pca_dims in x:
        for compress_rate in y:
            seed(2020)
            moderated = str(int(20 // rate))
            print(moderated, pca_dims, compress_rate)
            if pca_dims == 0:
                pca_dims = 30

            model_path = 'Acti-tracker/model/' + moderated + 'Hz/pruned/' + 'decompressed_pca=' + str(
                pca_dims) + '_compress_rate=' + str(compress_rate) + '.hdf5'

            model = load_model(model_path)
            pickle.dump(model, open('Acti-tracker/model/' + moderated + 'Hz/pruned/' + 'pickled_' + 'decompressed_pca=' + str(
                pca_dims) + '_compress_rate=' + str(compress_rate) + '.hdf5', 'wb'))
            del model