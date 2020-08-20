import os
from copy import deepcopy
import math
import split_acti
from keras.models import load_model
from keras.optimizers import Adam, rmsprop
from keras.utils import to_categorical
import numpy as np
from numpy.random import seed
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
from numpy import linalg as LA
import tensorflow as tf
# from tensorflow_model_optimization.sparsity import keras as sparsity


# def prune_weights(weight, compress_rate=0.9):
#     for i in range(weight.shape[-1]):
#         # print(weight.shape[1])
#         # print(weight[..., i])
#         tmp = deepcopy(weight[..., i])
#         tmp2 = deepcopy(weight[1][...,i])
#         # print(tmp2)
#         tmp = np.abs(tmp)
#         # print(len(tmp))
#         tmp = tmp[tmp >= 0]
#         # print(len(tmp))
#         # print(tmp)
#         tmp = np.sort(np.array(tmp))
#         # compute threshold
#         th = tmp[int(tmp.shape[0] * compress_rate)]
#         # print(th)
#         # print(weight[..., i][np.abs(weight[..., i]) < th])
#         weight[..., i][np.abs(weight[..., i]) < th] = 0
#     # print(weight)
#     mask = deepcopy(weight)
#     # print(mask)
#     # print(mask)
#     # print(mask!=0)
#     # print(len(mask[mask != 0]))
#     # print(len(mask[mask == 0]))
#     mask[mask != 0] = 1
#     return weight, mask
#
#
# def weight_prune_dense_layer(k_weights, b_weights, k_sparsity):
#     """
#     Takes in matrices of kernel and bias weights (for a dense
#       layer) and returns the unit-pruned versions of each
#     Args:
#       k_weights: 2D matrix of the
#       b_weights: 1D matrix of the biases of a dense layer
#       k_sparsity: percentage of weights to set to 0
#     Returns:
#       kernel_weights: sparse matrix with same shape as the original
#         kernel weight matrix
#       bias_weights: sparse array with same shape as the original
#         bias array
#     """
#     # Copy the kernel weights and get ranked indeces of the abs
#     kernel_weights = np.copy(k_weights)
#     # print(kernel_weights)
#     ind = np.unravel_index(
#         np.argsort(
#             np.abs(kernel_weights),
#             axis=None),
#         kernel_weights.shape)
#
#     # Number of indexes to set to 0
#     cutoff = int(len(ind[0]) * k_sparsity)
#     # The indexes in the 2D kernel weight matrix to set to 0
#     sparse_cutoff_inds = (ind[0][0:cutoff], ind[1][0:cutoff])
#     kernel_weights[sparse_cutoff_inds] = 0.
#
#     # Copy the bias weights and get ranked indeces of the abs
#     bias_weights = np.copy(b_weights)
#     # print(bias_weights)
#     ind = np.unravel_index(
#         np.argsort(
#             np.abs(bias_weights),
#             axis=None),
#         bias_weights.shape)
#
#     # Number of indexes to set to 0
#     cutoff = int(len(ind[0]) * k_sparsity)
#     # The indexes in the 1D bias weight matrix to set to 0
#     sparse_cutoff_inds = (ind[0][0:cutoff])
#     bias_weights[sparse_cutoff_inds] = 0.
#     # print(bias_weights)
#
#     return kernel_weights, bias_weights
#
#
# def unit_prune_dense_layer(k_weights, b_weights, k_sparsity):
#     """
#     Takes in matrices of kernel and bias weights (for a dense
#       layer) and returns the unit-pruned versions of each
#     Args:
#       k_weights: 2D matrix of the
#       b_weights: 1D matrix of the biases of a dense layer
#       k_sparsity: percentage of weights to set to 0
#     Returns:
#       kernel_weights: sparse matrix with same shape as the original
#         kernel weight matrix
#       bias_weights: sparse array with same shape as the original
#         bias array
#     """
#
#     # Copy the kernel weights and get ranked indeces of the
#     # column-wise L2 Norms
#     kernel_weights = np.copy(k_weights)
#     ind = np.argsort(LA.norm(kernel_weights, axis=0))
#     print(ind)
#
#     # Number of indexes to set to 0
#     cutoff = int(len(ind) * k_sparsity)
#     # The indexes in the 2D kernel weight matrix to set to 0
#     sparse_cutoff_inds = ind[0:cutoff]
#     kernel_weights[:, sparse_cutoff_inds] = 0.
#
#     # Copy the bias weights and get ranked indeces of the abs
#     bias_weights = np.copy(b_weights)
#     # The indexes in the 1D bias weight matrix to set to 0
#     # Equal to the indexes of the columns that were removed in this case
#     # sparse_cutoff_inds
#     bias_weights[sparse_cutoff_inds] = 0.
#
#     return kernel_weights, bias_weights
#
#
# masks = {}
# layer_count = 0
# # not compress first convolution layer
# # first_conv = True
# # for layer in model.layers:
# #     # print((layer.name))
# #     weight = layer.get_weights()
# #     if len(weight) >= 2:
# #         if not first_conv:
# #             # num_lowWeightNeurons = math.floor((40 * layer.output_shape[1]) / 100)
# #             # new_OutShape = layer.output_shape[1] - num_lowWeightNeurons
# #             # print(new_OutShape)
# #             print((layer.name))
# #             w = deepcopy(weight)
# #             print(w[0])
# #             print(w[1])
# #             tmp, mask = prune_weights(w[0], compress_rate=compress_rate)
# #             masks[layer_count] = mask
# #             w[0] = tmp
# #             w[0] = w[0] * masks[layer_count]
# #
# #             # print(w[1])
# #             # w[1] = w[1] * masks[layer_count]
# #             layer.set_weights(w)
# #
# #             # print(layer.get_weights())
# #         else:
# #             first_conv = False
# #     layer_count += 1
#
# def sparsify_model(model, x_test, y_test, k_sparsity, pruning='weight'):
#     """
#     Takes in a model made of dense layers and prunes the weights
#     Args:
#       model: Keras model
#       k_sparsity: target sparsity of the model
#     Returns:
#       sparse_model: sparsified copy of the previous model
#     """
#     # Copying a temporary sparse model from our original
#     sparse_model = model #tf.keras.models.clone_model(model)
#     # sparse_model.set_weights(model.get_weights())
#
#     # Getting a list of the names of each component (w + b) of each layer
#     names = [weight.name for layer in sparse_model.layers for weight in layer.weights]
#     # print(names)
#     # Getting the list of the weights for each component (w + b) of each layer
#     weights = sparse_model.get_weights()
#     # print(weights)
#
#
#     # Initializing list that will contain the new sparse weights
#     newWeightList = []
#
#     # Iterate over all but the final 2 layers (the softmax)
#     for i in range(0, len(weights), 2):
#
#         # print(weights[i])
#         # print(weights[i+1])
#
#         if pruning == 'weight':
#             kernel_weights, bias_weights = weight_prune_dense_layer(weights[i],
#                                                                     weights[i + 1],
#                                                                     k_sparsity)
#         elif pruning == 'unit':
#             kernel_weights, bias_weights = unit_prune_dense_layer(weights[i],
#                                                                   weights[i + 1],
#                                                                   k_sparsity)
#         else:
#             print('does not match available pruning methods ( weight | unit )')
#
#         # Append the new weight list with our sparsified kernel weights
#         newWeightList.append(kernel_weights)
#
#         # Append the new weight list with our sparsified bias weights
#         newWeightList.append(bias_weights)
#
#     # Adding the unchanged weights of the final 2 layers
#     # for i in range(len(weights) - 2, len(weights)):
#     # for i in range(len(weights) - 2, len(weights)):
#     #     unmodified_weight = np.copy(weights[i])
#     #     newWeightList.append(unmodified_weight)
#
#     # Setting the weights of our model to the new ones
#     sparse_model.set_weights(newWeightList)
#
#     # Re-compiling the Keras model (necessary for using `evaluate()`)
#     adam = Adam(lr=0.0004, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#     sparse_model.compile(
#         loss='mean_squared_error',
#         optimizer='adam',
#         metrics=['accuracy'])
#
#     # print((sparse_model.summary()))
#     #
#     # sparse_model.fit(np.expand_dims(x_test, axis=2), y_test,
#     #           batch_size=32, epochs=20, verbose=2, validation_split=0.2)
#     #
#     # print((sparse_model.summary()))
#
#     # Printing the the associated loss & Accuracy for the k% sparsity
#     # score = sparse_model.evaluate(np.expand_dims(x_test, axis=2), y_test, verbose=0)
#     # print('k% weight sparsity: ', k_sparsity,
#     #       '\tTest loss: {:07.5f}'.format(score[0]),
#     #       '\tTest accuracy: {:05.2f} %%'.format(score[1] * 100.))
#
#
#     return sparse_model, weights

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
# compress_rate = 0.5
X_train, X_test, y_train, y_test = split_acti.main(pca_dims, rate, test_data_ratio=0.3)



print(X_train.shape)
y_train = y_train - 1
y_test = y_test - 1

n_classes = 6

# Convert to one hot encoding vector
encoded = np.eye(n_classes)[y_train]
# print(encoded)


moderated = str(int(20 // rate))


model_path = 'model/' + moderated + 'Hz/' + 'pca=' + str(pca_dims) + '.hdf5'
model = load_model(model_path)
print(model.summary())

# dense3 = model.layers[5]
# print(dense3.name)
#
# dense3_w = dense3.get_weights()

c1 = model.layers[0].get_weights()
c2 = model.layers[5].get_weights()





# for compress_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:

compress_rate = 0.8
first_conv = True
layer_count = 2
# first_dense = True
prev = 0

model2 = Sequential()
model2.add(Conv1D(100, 6, input_shape=(pca_dims, 1), activation='relu'))
model2.add(MaxPooling1D(2))
model2.add(Flatten())

model2.layers[0].set_weights(c1)

for layer in model.layers:

    weight = layer.get_weights()
    if len(weight) >= 2:

        if not first_conv and layer.name == 'dense_1':

            if layer.name == 'dense_1':
                input_shape = layer.input_shape[1]
            else:
                input_shape = prev
            # print((layer.name))
            output_shape = layer.output_shape[1]
            # print(input_shape, output_shape)

            layer_count += 1
            # print((layer.name))
            num_lowWeightNeurons = math.floor((compress_rate * output_shape))
            print(num_lowWeightNeurons)
            print("dis",input_shape)
            new_OutShape = output_shape - num_lowWeightNeurons
            # print(new_OutShape)
            # print(len(weight[0]))

            new_w = [] #np.empty(input_shape,)
            new_w2 = []
            qw=0
            for x, i in zip(weight[0], range(len(weight[0]))):
                x_abs = abs(x)
                x_abs = np.sort(x_abs)
                print(len(weight[0]))
                # print("ok",x_abs[num_lowWeightNeurons:])
                if prev != 0:
                    input_shape_2 = abs(input_shape - num_lowWeightNeurons)
                    new_w.append(x_abs[abs(len(weight[0]) - abs(input_shape - num_lowWeightNeurons)):][abs(len(weight[0]) - input_shape_2):])

                    # new_w = np.sort(new_w[i])[input_shape_2:]


                else:
                    new_w.append(x_abs[num_lowWeightNeurons:])
                    input_shape_2 = input_shape
                # print(new_w)
                qw+=1
                # dense3_w_new[i] = [w for w in x if abs(w) in new_w]
            print(len(new_w))
            print("lol",qw)
            print(input_shape_2)
            # print(weight[0])
            # print(weight[0][1])
            # print(len(weight[0][]))

            dense3_w_new = np.empty((input_shape_2, new_OutShape))


            for i in range(input_shape_2):
                dense3_w_new[i] = new_w[i] #[w for w in x if abs(w) in new_w]

            # removing bias of 20% of neurons with low weights
            new_bias = np.empty((new_OutShape,))
            y = weight[1]
            x_abs = abs(weight[1])
            x_abs = np.sort(x_abs)
            new_b = x_abs[num_lowWeightNeurons:]
            new_bias = [w for w in y if abs(w) in new_b]
            new_bias = np.asarray(new_bias)

            weight[0] = dense3_w_new
            weight[1] = new_bias
            # print(weight)
            # print(input_shape)

            # layer.output_shape[1] = new_OutShape
            #
            # layer.set_weights(weight)

            model2.add(Dense(new_OutShape, activation='relu'))
            model2.layers[layer_count].set_weights(weight)

            prev = new_OutShape


            # print(weight[0])
            # print(weight[1])

        else:
            first_conv = False


model2.add(Dense(6, activation='softmax'))
model2.add(Dropout(0.5))
adam = Adam(lr=0.0004, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)
opt = rmsprop(lr=0.0001, decay=1e-6)

model2.compile(
        loss='mean_squared_error',
        optimizer=adam,
        metrics=['accuracy'])

print((model2.summary()))

# new_dir = 'model/' + moderated + 'Hz/weights/'
# if not os.path.exists(new_dir):
#     os.makedirs(new_dir)
# fpath = new_dir + moderated + 'Hz' + '_pca' + str(pca_dims) + '_weights.{epoch:02d}-{val_accuracy:.2f}.hdf5'
#
# cp_cb = ModelCheckpoint(fpath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto', period=1)


model2.fit(np.expand_dims(X_train, axis=2), encoded,
           batch_size=32, epochs=50, verbose=2, validation_split=0.2) #, callbacks=[cp_cb])

model2.save(
    'model/' + moderated + 'Hz/' + 'pruned_pca=' + str(pca_dims) + '_compress_rate=' + str(compress_rate) + '.hdf5')
# print("--- %s seconds ---" % (time.time() - start_time))
del model2
del model

# count = 0
# # while count <2:
# # for compress_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
#     # while count <2:
#
# compress_rate = 0.1
# model2 = load_model('model/' + moderated + 'Hz/' + 'pruned_pca=' + str(pca_dims) + '_compress_rate=' + str(compress_rate) + '.hdf5')
# # y_train_dynamic_oh = np.eye(6)[y_train]
# model2.fit(np.expand_dims(X_train, axis=2), encoded,
#               batch_size=32, epochs=50, verbose=2, validation_split=0.2)
#
# model2.save('model/' + moderated + 'Hz/' + 'pruned_pca=' + str(pca_dims) + '_compress_rate=' + str(compress_rate) + '.hdf5')
# # print("--- %s seconds ---" % (time.time() - start_time))
# del model2

        # count+=1


# pred_test = model2.predict(np.expand_dims(X_test, axis=2), batch_size=32)
# # print("------ TEST ACCURACY ------")
# testing_acc = (accuracy_score(y_test, np.argmax(pred_test, axis=1)))


# num_lowWeightNeurons = math.floor((compress_rate * output_shape))
# new_OutShape = output_shape - num_lowWeightNeurons
# dense3_w_new = np.empty((input_shape, new_OutShape))
# for x, i in zip(dense3_w[0], range(len(dense3_w[0]))):
#     x_abs = abs(x)
#     x_abs = np.sort(x_abs)
#     new_w = x_abs[num_lowWeightNeurons:]
#     # print(len(new_w))
#
#     dense3_w_new[i] = [w for w in x if abs(w) in new_w]

#new weights for the last dense layer
# dense3_w_new = dense3_w
# dense3_w_new[0] = dense_w_new
# dense3_w_new[1] = new_bias

# #plot new weight distribution
# ax = dist_plot(dense3_w_new,plt_range= (-0.2, 0.2))
# ax.set_title('Weights distribution')
# ax.set_xlabel('Weights values')
# ax.set_ylabel('Number of Weights')
# pred_train = model.predict(np.expand_dims(X_train, axis=2), batch_size=32)
# # print("------ TRAIN ACCURACY ------")
# accuracy = (accuracy_score(y_train, np.argmax(pred_train, axis=1)))
# print(accuracy)

seed(2020)

# k_sparsities = [0.0, 0.25, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.97, 0.99]
#
# # The empty lists where we will store our training results
# mnist_model_loss_weight = []
# mnist_model_accs_weight = []
# mnist_model_loss_unit = []
# mnist_model_accs_unit = []
# fmnist_model_loss_weight = []
# fmnist_model_accs_weight = []
# fmnist_model_loss_unit = []
# fmnist_model_accs_unit = []
#
# dataset = 'mnist'
# pruning = 'weight'
# print('\n Weight-pruning\n')
# for k_sparsity in k_sparsities:
#     sparse_model, wee = sparsify_model(model, X_train,
#                                          y_train_dynamic_oh,
#                                          k_sparsity=k_sparsity,
#                                          pruning=pruning)
#     # mnist_model_loss_weight.append(score[0])
#     # mnist_model_accs_weight.append(score[1])
#
#     # Save entire model to an H5 file
#     sparse_model.save('model/sparse_model_k-{}_{}-pruned.h5'.format(k_sparsity, pruning))
#     del sparse_model
#
# pruning = 'unit'
# print('\n Unit-pruning\n')
# for k_sparsity in k_sparsities:
#     sparse_model, wee = sparsify_model(model, X_train,
#                                          y_train_dynamic_oh,
#                                          k_sparsity=k_sparsity,
#                                          pruning=pruning)
#     # mnist_model_loss_unit.append(score[0])
#     # mnist_model_accs_unit.append(score[1])
#
#     # Save entire model to an H5 file
#     sparse_model.save('model/sparse_model_k-{}_{}-pruned.h5'.format(k_sparsity, pruning))
#     del sparse_model

# save compressed weights
# compressed_dir = 'model/' + moderated + 'Hz/pruned/'
#
# model.save('tuned_model.hdf5')
#
# model2 = load_model('tuned_model.hdf5')
# print(model2.summary())