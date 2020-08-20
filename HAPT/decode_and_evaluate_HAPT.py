from HAPT import split_HAPT
from compression import decode_weights
from keras.models import load_model
from keras.optimizers import Adam
from numpy.random import seed
import pickle
import tensorflow

if __name__ == '__main__':

    # x = [30, 15, 18, 21, 24, 27]
    x = [27]
    y = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # y = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # y2 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    s_rate = [10] #[10, 5, 2.5, 2, 1.25, 1]

    for rate in s_rate:
    # rate = 1
        for pca_dims in x:
            for c_rate in y:
                for compress_rate in y:
                    seed(2020)
                    X_train, X_test, y_train, y_test = split_HAPT.main(pca_dims, rate, 0.3, 'acc')
                    moderated = str(int(50 // rate))
                    if pca_dims == 0:
                        pca_dims = 30
                    # X_train, X_test, y_train, y_test = split.main(pca_dims)

                    y_train = y_train - 1
                    y_test = y_test - 1
                    # print(y_train.shape)
                    print(rate, pca_dims, c_rate, compress_rate)

                    model_path = 'model/' + moderated + 'Hz/' + 'pruned_acc_pca=' + str(pca_dims) + '_compress_rate=' + str(
                        c_rate) + '.hdf5'
                    model = load_model(model_path)

                    weight_file = 'model/' + moderated + 'Hz/pruned/' + 'acc_compressed_' + 'pca=' + str(
                        pca_dims) + '_compress_rate=' + str(c_rate) + '_sparse_rate=' + str(
                        compress_rate) + '.h5'

                    # decode
                    weights = decode_weights(weight_file)
                    for i in range(len(model.layers)):
                        if model.layers[i].name in weights:
                            weight = [w for w in weights[model.layers[i].name]]
                            model.layers[i].set_weights(weight)

                    adam = Adam(lr=0.0004, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
                    model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])

                    model.save(
                        'model/' + moderated + 'Hz/pruned/' + 'acc_decompressed_pca=' + str(
                            pca_dims) + '_compress_rate=' + str(c_rate) + '_sparse_rate=' + str(
                            compress_rate) + '.hdf5')

                    pickle.dump(model,
                                open('model/' + moderated + 'Hz/pickled/' + 'pickled_' + 'acc_decompressed_pca=' + str(
                                    pca_dims) + '_compress_rate=' + str(c_rate) + '_sparse_rate=' + str(
                                    compress_rate) + '.hdf5', 'wb'))

                    del model