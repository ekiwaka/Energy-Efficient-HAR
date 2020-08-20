import os
from copy import deepcopy

from HAPT import split_HAPT
from compression import prune_weights, save_compressed_weights
from keras.models import load_model
from numpy.random import seed


def training():
    # pca_dims = 0
    # compress_rate = 0.85

    # x = [30, 15, 18, 21, 24, 27]
    x = [27] #, 24, 27]
    y = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    s_rate = [2] #, 1.25, 2.5, 10, 5, 2]

    for rate in s_rate:
        for pca_dims in x:
            for c_rate in y:
                for compress_rate in y:
                    seed(2020)
                    X_train, X_test, y_train, y_test = split_HAPT.main(pca_dims, rate, 0.3, 'acc')

                    y_train = y_train - 1
                    masks = {}
                    layer_count = 0

                    moderated = str(int(50 // rate))

                    if pca_dims == 0:
                        pca_dims = 30

                    model_path = 'model/' + moderated + 'Hz/' + 'pruned_acc_pca=' + str(pca_dims) + '_compress_rate=' + str(c_rate) + '.hdf5'
                    model = load_model(model_path)

                    # pred_train = model.predict(np.expand_dims(X_train, axis=2), batch_size=32)
                    # # print("------ TRAIN ACCURACY ------")
                    # accuracy = (accuracy_score(y_train, np.argmax(pred_train, axis=1)))
                    # print(accuracy)

                    seed(2020)
                    # not compress first convolution layer
                    first_conv = True
                    for layer in model.layers:
                        weight = layer.get_weights()
                        if len(weight) >= 2:
                            if not first_conv:
                                w = deepcopy(weight)
                                tmp, mask = prune_weights(w[0], compress_rate=compress_rate)
                                masks[layer_count] = mask
                                w[0] = tmp
                                layer.set_weights(w)
                            else:
                                first_conv = False
                        layer_count += 1

                    # save compressed weights
                    compressed_dir = 'model/' + moderated + 'Hz/pruned/'
                    if not os.path.exists(compressed_dir):
                        os.makedirs(compressed_dir)
                    compressed_name = compressed_dir + 'acc_compressed_' + 'pca=' + str(pca_dims) + '_compress_rate=' + str(c_rate) + '_sparse_rate=' + str(
                    compress_rate)
                    save_compressed_weights(model, compressed_name)
                    print(rate, pca_dims, c_rate, compress_rate)


if __name__ == '__main__':
    # parse = argparse.ArgumentParser()
    # parse.add_argument('--data', type=str, default='c10', help='Supports c10 (CIFAR-10) and c100 (CIFAR-100)')
    # parse.add_argument('--model', type=str, default='resnet')
    # parse.add_argument('--depth', type=int, default=50)
    # parse.add_argument('--growth-rate', type=int, default=12, help='growth rate for densenet')
    # parse.add_argument('--wide-factor', type=int, default=1, help='wide factor for WRN')
    # args = parse.parse_args()

    training()
