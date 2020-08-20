import os
from copy import deepcopy
import math
import split_acti
from keras.models import load_model
from keras.optimizers import Adam
import numpy as np
from numpy.random import seed
from numpy import linalg as LA
import tensorflow as tf
# from tensorflow_model_optimization.sparsity import keras as sparsity


def prune_weights(weight, compress_rate=0.9):
    for i in range(weight.shape[-1]):
        # print(weight.shape[1])
        # print(weight[..., i])
        tmp = deepcopy(weight[..., i])
        tmp2 = deepcopy(weight[1][...,i])
        # print(tmp2)
        tmp = np.abs(tmp)
        # print(len(tmp))
        tmp = tmp[tmp >= 0]
        # print(len(tmp))
        # print(tmp)
        tmp = np.sort(np.array(tmp))
        # compute threshold
        th = tmp[int(tmp.shape[0] * compress_rate)]
        # print(th)
        # print(weight[..., i][np.abs(weight[..., i]) < th])
        weight[..., i][np.abs(weight[..., i]) < th] = 0
    # print(weight)
    mask = deepcopy(weight)
    # print(mask)
    # print(mask)
    # print(mask!=0)
    # print(len(mask[mask != 0]))
    # print(len(mask[mask == 0]))
    mask[mask != 0] = 1
    return weight, mask


def weight_prune_dense_layer(k_weights, b_weights, k_sparsity):
    """
    Takes in matrices of kernel and bias weights (for a dense
      layer) and returns the unit-pruned versions of each
    Args:
      k_weights: 2D matrix of the
      b_weights: 1D matrix of the biases of a dense layer
      k_sparsity: percentage of weights to set to 0
    Returns:
      kernel_weights: sparse matrix with same shape as the original
        kernel weight matrix
      bias_weights: sparse array with same shape as the original
        bias array
    """
    # Copy the kernel weights and get ranked indeces of the abs
    kernel_weights = np.copy(k_weights)
    # print(kernel_weights)
    ind = np.unravel_index(
        np.argsort(
            np.abs(kernel_weights),
            axis=None),
        kernel_weights.shape)

    # Number of indexes to set to 0
    cutoff = int(len(ind[0]) * k_sparsity)
    # The indexes in the 2D kernel weight matrix to set to 0
    sparse_cutoff_inds = (ind[0][0:cutoff], ind[1][0:cutoff])
    kernel_weights[sparse_cutoff_inds] = 0.

    # Copy the bias weights and get ranked indeces of the abs
    bias_weights = np.copy(b_weights)
    # print(bias_weights)
    ind = np.unravel_index(
        np.argsort(
            np.abs(bias_weights),
            axis=None),
        bias_weights.shape)

    # Number of indexes to set to 0
    cutoff = int(len(ind[0]) * k_sparsity)
    # The indexes in the 1D bias weight matrix to set to 0
    sparse_cutoff_inds = (ind[0][0:cutoff])
    bias_weights[sparse_cutoff_inds] = 0.
    # print(bias_weights)

    return kernel_weights, bias_weights


def unit_prune_dense_layer(k_weights, b_weights, k_sparsity):
    """
    Takes in matrices of kernel and bias weights (for a dense
      layer) and returns the unit-pruned versions of each
    Args:
      k_weights: 2D matrix of the
      b_weights: 1D matrix of the biases of a dense layer
      k_sparsity: percentage of weights to set to 0
    Returns:
      kernel_weights: sparse matrix with same shape as the original
        kernel weight matrix
      bias_weights: sparse array with same shape as the original
        bias array
    """

    # Copy the kernel weights and get ranked indeces of the
    # column-wise L2 Norms
    kernel_weights = np.copy(k_weights)
    ind = np.argsort(LA.norm(kernel_weights, axis=0))
    print(ind)

    # Number of indexes to set to 0
    cutoff = int(len(ind) * k_sparsity)
    # The indexes in the 2D kernel weight matrix to set to 0
    sparse_cutoff_inds = ind[0:cutoff]
    kernel_weights[:, sparse_cutoff_inds] = 0.

    # Copy the bias weights and get ranked indeces of the abs
    bias_weights = np.copy(b_weights)
    # The indexes in the 1D bias weight matrix to set to 0
    # Equal to the indexes of the columns that were removed in this case
    # sparse_cutoff_inds
    bias_weights[sparse_cutoff_inds] = 0.

    return kernel_weights, bias_weights


masks = {}
layer_count = 0
# not compress first convolution layer
# first_conv = True
# for layer in model.layers:
#     # print((layer.name))
#     weight = layer.get_weights()
#     if len(weight) >= 2:
#         if not first_conv:
#             # num_lowWeightNeurons = math.floor((40 * layer.output_shape[1]) / 100)
#             # new_OutShape = layer.output_shape[1] - num_lowWeightNeurons
#             # print(new_OutShape)
#             print((layer.name))
#             w = deepcopy(weight)
#             print(w[0])
#             print(w[1])
#             tmp, mask = prune_weights(w[0], compress_rate=compress_rate)
#             masks[layer_count] = mask
#             w[0] = tmp
#             w[0] = w[0] * masks[layer_count]
#
#             # print(w[1])
#             # w[1] = w[1] * masks[layer_count]
#             layer.set_weights(w)
#
#             # print(layer.get_weights())
#         else:
#             first_conv = False
#     layer_count += 1

def sparsify_model(model, x_test, y_test, k_sparsity, pruning='weight'):
    """
    Takes in a model made of dense layers and prunes the weights
    Args:
      model: Keras model
      k_sparsity: target sparsity of the model
    Returns:
      sparse_model: sparsified copy of the previous model
    """
    # Copying a temporary sparse model from our original
    sparse_model = model #tf.keras.models.clone_model(model)
    # sparse_model.set_weights(model.get_weights())

    # Getting a list of the names of each component (w + b) of each layer
    names = [weight.name for layer in sparse_model.layers for weight in layer.weights]
    # print(names)
    # Getting the list of the weights for each component (w + b) of each layer
    weights = sparse_model.get_weights()
    # print(weights)


    # Initializing list that will contain the new sparse weights
    newWeightList = []

    # Iterate over all but the final 2 layers (the softmax)
    for i in range(0, len(weights), 2):

        # print(weights[i])
        # print(weights[i+1])

        if pruning == 'weight':
            kernel_weights, bias_weights = weight_prune_dense_layer(weights[i],
                                                                    weights[i + 1],
                                                                    k_sparsity)
        elif pruning == 'unit':
            kernel_weights, bias_weights = unit_prune_dense_layer(weights[i],
                                                                  weights[i + 1],
                                                                  k_sparsity)
        else:
            print('does not match available pruning methods ( weight | unit )')

        # Append the new weight list with our sparsified kernel weights
        newWeightList.append(kernel_weights)

        # Append the new weight list with our sparsified bias weights
        newWeightList.append(bias_weights)

    # Adding the unchanged weights of the final 2 layers
    # for i in range(len(weights) - 2, len(weights)):
    # for i in range(len(weights) - 2, len(weights)):
    #     unmodified_weight = np.copy(weights[i])
    #     newWeightList.append(unmodified_weight)

    # Setting the weights of our model to the new ones
    sparse_model.set_weights(newWeightList)

    # Re-compiling the Keras model (necessary for using `evaluate()`)
    adam = Adam(lr=0.0004, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    sparse_model.compile(
        loss='mean_squared_error',
        optimizer='adam',
        metrics=['accuracy'])

    # print((sparse_model.summary()))
    #
    # sparse_model.fit(np.expand_dims(x_test, axis=2), y_test,
    #           batch_size=32, epochs=20, verbose=2, validation_split=0.2)
    #
    # print((sparse_model.summary()))

    # Printing the the associated loss & Accuracy for the k% sparsity
    # score = sparse_model.evaluate(np.expand_dims(x_test, axis=2), y_test, verbose=0)
    # print('k% weight sparsity: ', k_sparsity,
    #       '\tTest loss: {:07.5f}'.format(score[0]),
    #       '\tTest accuracy: {:05.2f} %%'.format(score[1] * 100.))


    return sparse_model, weights

# fine_tune_epochs = 30
# for i in range(fine_tune_epochs):
#     model.fit(np.expand_dims(X_train, axis=2), y_train_dynamic_oh,
#               batch_size=32, epochs=1, verbose=2, validation_split=0.2)
# for layer_id in masks:
#     print(layer_id)
#     w = model.layers[layer_id].get_weights()
#     # print(w)
#     w[0] = w[0] * masks[layer_id]
#     print(w)
#     model.layers[layer_id].set_weights(w)
    #
    # model.fit(np.expand_dims(X_train, axis=2), y_train_dynamic_oh,
    #           batch_size=32, epochs=1, verbose=2, validation_split=0.2)



rate = 1
pca_dims = 30
compress_rate = 0.3
X_train, X_test, y_train, y_test = split_acti.main(pca_dims, rate, test_data_ratio=0.3)

y_train = y_train - 1
y_test = y_test - 1

n_classes = 6

# Convert to one hot encoding vector
y_train_dynamic_oh = np.eye(n_classes)[y_train]


moderated = str(int(20 // rate))


model_path = 'model/' + moderated + 'Hz/' + 'pca=' + str(pca_dims) + '.hdf5'
model = load_model(model_path)
print(model.summary())

# pred_train = model.predict(np.expand_dims(X_train, axis=2), batch_size=32)
# # print("------ TRAIN ACCURACY ------")
# accuracy = (accuracy_score(y_train, np.argmax(pred_train, axis=1)))
# print(accuracy)

seed(2020)

k_sparsities = [0.0, 0.25, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.97, 0.99]

# The empty lists where we will store our training results
mnist_model_loss_weight = []
mnist_model_accs_weight = []
mnist_model_loss_unit = []
mnist_model_accs_unit = []
fmnist_model_loss_weight = []
fmnist_model_accs_weight = []
fmnist_model_loss_unit = []
fmnist_model_accs_unit = []

dataset = 'mnist'
pruning = 'weight'
print('\n Weight-pruning\n')
for k_sparsity in k_sparsities:
    sparse_model, wee = sparsify_model(model, X_train,
                                         y_train_dynamic_oh,
                                         k_sparsity=k_sparsity,
                                         pruning=pruning)
    # mnist_model_loss_weight.append(score[0])
    # mnist_model_accs_weight.append(score[1])

    # Save entire model to an H5 file
    sparse_model.save('model/sparse_model_k-{}_{}-pruned.h5'.format(k_sparsity, pruning))
    del sparse_model

pruning = 'unit'
print('\n Unit-pruning\n')
for k_sparsity in k_sparsities:
    sparse_model, wee = sparsify_model(model, X_train,
                                         y_train_dynamic_oh,
                                         k_sparsity=k_sparsity,
                                         pruning=pruning)
    # mnist_model_loss_unit.append(score[0])
    # mnist_model_accs_unit.append(score[1])

    # Save entire model to an H5 file
    sparse_model.save('model/sparse_model_k-{}_{}-pruned.h5'.format(k_sparsity, pruning))
    del sparse_model

# save compressed weights
# compressed_dir = 'model/' + moderated + 'Hz/pruned/'
#
# model.save('tuned_model.hdf5')
#
# model2 = load_model('tuned_model.hdf5')
# print(model2.summary())