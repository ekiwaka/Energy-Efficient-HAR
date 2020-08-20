import os
from copy import deepcopy
import math
import split_acti
from keras.models import load_model
import numpy as np
from numpy.random import seed
from numpy import linalg as LA
import tensorflow as tf




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

k_sparsities = [0.0, 0.25, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.97, 0.99]

for k_sparsity in k_sparsities:
    pruning = 'weight'
    # sparse_model.save('model/sparse_model_k-{}_{}-pruned.hdf5'.format(k_sparsity, pruning))
    model_path2 = 'model/sparse_model_k-' + str(k_sparsity) + '_' + pruning + '-pruned.h5'
    model2 = load_model(model_path2)
    print(model2.summary())

    pruning = 'unit'
    # sparse_model.save('model/sparse_model_k-{}_{}-pruned.hdf5'.format(k_sparsity, pruning))
    model_path3 = 'model/sparse_model_k-' + str(k_sparsity) + '_' + pruning + '-pruned.h5'
    model3 = load_model(model_path3)
    print(model3.summary())