import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
# import random
from numpy.random import seed
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout  # , GlobalMaxPooling1D
from keras.callbacks import ModelCheckpoint
# from keras.utils import plot_model
import keras.backend as K
# import process_data
import time
from HAPT import split_HAPT

# import sys
# import pyRAPL

pca_dims = 10

# ...
# Instructions to be evaluated.
# ...

# Select sample rate moderator. Default sample rate is 50Hz. Moderated sample rate would be default//rate
# sampling_rate = [10, 5, 2.5, 2, 1.25, 1]
# for rate in sampling_rate:
rate = 1
moderated = str(int(50 // rate))
print(moderated)
X_train, X_test, y_train, y_test = split_HAPT.main(pca_dims)
start_time = time.time()
# Load all train and test data (* dynamic and static data are mixed.)

print(y_test.shape)
# Convert (1, 2, 3) labels to (0, 1, 2)
y_train = y_train - 1
print(y_train.shape)

print(("test_dynamic shape: ", X_test.shape))

print(X_train.shape)
print(y_test.shape)

n_classes = 12

# Convert to one hot encoding vector
y_train_dynamic_oh = np.eye(n_classes)[y_train]
# y_train_dynamic_oh = np.delete(y_train_dynamic_oh, 0, 1)

print(y_train_dynamic_oh.shape)
print(y_train.shape)

print(y_train)
print(y_train_dynamic_oh)

# Fit 1d CNN for dynamic HAR

seed(2020)
model = Sequential()
model.add(Conv1D(100, 12, input_shape=(pca_dims, 1), activation='relu'))
model.add(MaxPooling1D(8))
model.add(Flatten())
model.add(Dense(12, activation='softmax'))
model.add(Dropout(0.5))
#
adam = Adam(lr=0.0004, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])

# model = Sequential()
# model.add(Conv1D(100, 3, input_shape=(30, 1), activation='relu', padding='same'))
# # model.add(MaxPooling1D(12, padding='same'))
# model.add(Conv1D(64, 3, activation='relu', padding='same'))
# # model.add(MaxPooling1D(12, padding='same'))
# model.add(Dropout(0.50))
# model.add(GlobalMaxPooling1D())
# # model.add(MaxPooling1D(12, padding='same'))
# # model.add(Conv1D(50, 12, activation='relu', padding='same'))
# # model.add(MaxPooling1D(12, padding='same'))
# model.add(Dropout(0.25))
# # model.add(Conv1D(100, 12, activation='relu', padding='same'))
# # model.add(Flatten())
# model.add(Dense(100, activation='softmax'))
# model.add(Dense(12, activation='softmax'))
# # model.add(Dropout(0.50))

# adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
# model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# Summarize layers
print((model.summary()))

# Save model image
# if not os.path.exists('fig_har_dyna.png'):
#     model_file = 'fig_har_dyna.png'
#     plot_model(model, to_file=model_file)

new_dir = 'model/' + moderated + 'Hz/weights/'
if not os.path.exists(new_dir):
    os.makedirs(new_dir)
fpath = new_dir + moderated + 'Hz' + '_pca' + str(pca_dims) + '_weights.{epoch:02d}-{val_accuracy:.2f}.hdf5'

cp_cb = ModelCheckpoint(fpath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto', period=1)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# To disable learning, the below code - two lines - is commented.
# To enable learning uncomment the below two lines of code.

model.fit(np.expand_dims(X_train, axis=2), y_train_dynamic_oh,
          batch_size=32, epochs=50, verbose=2, validation_split=0.2, callbacks=[cp_cb])
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
model.save('model/' + moderated + 'Hz/' + 'pca' + str(pca_dims) + '.hdf5')
print("--- %s seconds ---" % (time.time() - start_time))
del model
K.clear_session()
# report.data.head()
#
#
# '''
#
# /usr/bin/python2.7 /home/hcilab/Documents/OSS/sensors2018cnnhar/har/har_dyna_learn_model.py
# /home/hcilab/.local/lib/python2.7/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
#   from ._conv import register_converters as _register_converters
# Using TensorFlow backend.
#
# +++ DATA STATISTICS +++
#
# train_dynamic shape:  (3285, 561)
# test_dynamic shape:  (1387, 561)
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv1d_1 (Conv1D)            (None, 559, 100)          400
# _________________________________________________________________
# max_pooling1d_1 (MaxPooling1 (None, 186, 100)          0
# _________________________________________________________________
# flatten_1 (Flatten)          (None, 18600)             0
# _________________________________________________________________
# dense_1 (Dense)              (None, 3)                 55803
# _________________________________________________________________
# dropout_1 (Dropout)          (None, 3)                 0
# =================================================================
# Total params: 56,203
# Trainable params: 56,203
# Non-trainable params: 0
# _________________________________________________________________
# None
#
#
# Process finished with exit code 0
#
# '''
