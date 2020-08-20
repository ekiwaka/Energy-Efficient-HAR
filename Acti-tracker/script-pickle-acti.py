import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pickle

from numpy.random import seed
from keras.models import load_model



if __name__ == '__main__':

    # x = [30, 15, 18, 21, 24, 27]
    x = [7, 8, 10, 20]
    # x = [0, 20, 30, 40, 50]
    y = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


    rate = 1
    for pca_dims in x:
        for c_rate in y:
            for compress_rate in y:
                seed(2020)
                moderated = str(int(20 // rate))
                print(pca_dims, c_rate, compress_rate)
                if pca_dims == 0:
                    pca_dims = 30

                model_path = 'model/' + moderated + 'Hz/pickle_test2/' + 'decompressed_pca=' + str(
                    pca_dims) + '_compress_rate=' + str(c_rate) + '_sparse_rate=' + str(compress_rate) + '.hdf5'

                model = load_model(model_path)
                pickle.dump(model, open('model/' + moderated + 'Hz/pickle_test2/' + 'pickled_' + 'decompressed_pca=' + str(
                    pca_dims) + '_compress_rate=' + str(c_rate) + '_sparse_rate=' + str(compress_rate) + '.hdf5', 'wb'))
                del model