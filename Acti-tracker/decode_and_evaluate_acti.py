import split_acti
from compression import decode_weights
from keras.models import load_model
from keras.optimizers import Adam
from numpy.random import seed

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--pca', type=int, nargs='+',
    #                     help='select the degree of PCA')
    # parser.add_argument('--compression', type=int, nargs='+',
    #                     help='select the compression rate')
    # args = parser.parse_args()
    #
    # pca_dims = args.pca
    # compression = args.compression

    # x = [30, 7, 8, 10, 15, 20]
    x = [15, 20]
    y = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    s_rate = 1  # [10, 5, 2.5, 2, 1.25, 1]

    # for rate in s_rate:
    rate = 1
    for pca_dims in x:
        for c_rate in y:
            for compress_rate in y:
                seed(2020)
                X_train, X_test, y_train, y_test = split_acti.main(pca_dims, rate, test_data_ratio=0.3)
                moderated = str(int(20 // rate))
                if pca_dims == 0:
                    pca_dims = 30
                # X_train, X_test, y_train, y_test = split.main(pca_dims)

                y_train = y_train - 1
                y_test = y_test - 1
                # print(y_train.shape)
                print(pca_dims, c_rate, compress_rate)

                # print(("test_dynamic shape: ", X_test.shape))

                # print(X_train.shape)
                # print(y_test.shape)

                # model_path = 'model/' + moderated + 'Hz/' + 'pca=' + str(pca_dims) + '.hdf5'
                # model = load_model(model_path)

                model_path = 'model/' + moderated + 'Hz/' + 'pruned_pca=' + str(pca_dims) + '_compress_rate=' + str(
                    c_rate) + '.hdf5'
                model = load_model(model_path)

                # model.summary()

                # pred_train = model.predict(np.expand_dims(X_train, axis=2), batch_size=32)
                # print("------ TRAIN ACCURACY ------")
                # print((accuracy_score(y_train, np.argmax(pred_train, axis=1))))
                # # print((confusion_matrix(y_train, np.argmax(pred_train, axis=1))))
                #
                # pred_test = model.predict(np.expand_dims(X_test, axis=2), batch_size=32)
                # print("------ TEST ACCURACY ------")
                # print((accuracy_score(y_test, np.argmax(pred_test, axis=1))))
                # # print((confusion_matrix(y_test, np.argmax(pred_test, axis=1))))

                weight_file = 'model/' + moderated + 'Hz/pruned/' + 'compressed_' + 'pca=' + str(pca_dims) + '_compress_rate=' + str(c_rate) + '_sparse_rate=' + str(
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
                    'model/' + moderated + 'Hz/pruned/' + 'decompressed_pca=' + str(pca_dims) + '_compress_rate=' + str(c_rate) + '_sparse_rate=' + str(
                    compress_rate) + '.hdf5')

                # 'compressed_' + 'decompressed_pca=' + str(pca_dims) + '_compress_rate=' + str(c_rate) + '_sparse_rate=' + str(
                #     compress_rate)

                # pred_train = model.predict(np.expand_dims(X_train, axis=2), batch_size=32)
                # print("------ TRAIN ACCURACY ------")
                # print((accuracy_score(y_train, np.argmax(pred_train, axis=1))))
                # # print((confusion_matrix(y_train, np.argmax(pred_train, axis=1))))
                #
                # pred_test = model.predict(np.expand_dims(X_test, axis=2), batch_size=32)
                # print("------ TEST ACCURACY ------")
                # print((accuracy_score(y_test, np.argmax(pred_test, axis=1))))
                # # print((confusion_matrix(y_test, np.argmax(pred_test, axis=1))))
