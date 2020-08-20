import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import csv
# import random
from numpy.random import seed
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout
from keras.callbacks import ModelCheckpoint
# from keras.utils import plot_model
import keras.backend as K
# import process_data
import time
from HAPT import split_HAPT

# import sys
# import pyRAPL

# pca_dims = 20

# ...
# Instructions to be evaluated.
# ...

# Select sample rate moderator. Default sample rate is 50Hz. Moderated sample rate would be default//rate
s_rate = [2.5, 1.25] #[10, 5, 2.5, 2, 1.25, 1]
p = [30, 15, 18, 21, 24, 27] #[0, 20, 30, 40, 50]

with open('training_results_acc.txt', 'a', newline='') as f_out:
    writer = csv.writer(f_out)
    writer.writerow(['Sampling rate', 'PCA dims', 'Time'])
    f_out.close()

for rate in s_rate:
    for pca_dims in p:
# rate = 1
        moderated = str(int(50 // rate))
        print(moderated)
        # pca_dims = 60
        X_train, X_test, y_train, y_test = split_HAPT.main(pca_dims, rate, 0.3,'acc')
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

        if pca_dims == 0:
            pca_dims = 30

        # Fit 1d CNN for dynamic HAR

        seed(2020)
        model = Sequential()
        model.add(Conv1D(100, 12, input_shape=(pca_dims, 1), activation='relu'))
        model.add(MaxPooling1D(2))
        model.add(Flatten())
        model.add(Dense(12, activation='relu'))
        # model.add(Dense(12, activation='relu'))
        # model.add(Dense(128, activation='relu'))
        model.add(Dense(12, activation='softmax'))
        model.add(Dropout(0.5))
        #
        adam = Adam(lr=0.0004, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])

        # Summarize layers
        print((model.summary()))

        # Save model image
        # if not os.path.exists('fig_har_hapt.png'):
        #     model_file = 'fig_har_hapt.png'
        #     plot_model(model, to_file=model_file)

        new_dir = 'model/' + moderated + 'Hz/weights/'
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        fpath = new_dir + moderated + 'Hz_acc' + '_pca' + str(pca_dims) + '_weights.{epoch:02d}-{val_accuracy:.2f}.hdf5'

        cp_cb = ModelCheckpoint(fpath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto', period=1)

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # To disable learning, the below code - two lines - is commented.
        # To enable learning uncomment the below two lines of code.

        model.fit(np.expand_dims(X_train, axis=2), y_train_dynamic_oh,
                  batch_size=32, epochs=50, verbose=2, validation_split=0.2, callbacks=[cp_cb])
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        model.save('model/' + moderated + 'Hz/' + 'acc_pca=' + str(pca_dims) + '.hdf5')
        print("--- %s seconds ---" % (time.time() - start_time))
        del model

        with open('training_results_acc.txt', 'a', newline='') as f_out:
            writer = csv.writer(f_out)
            # writer.writerow(['Sampling rate', 'PCA dims', 'Time'])
            writer.writerow([moderated, pca_dims, (time.time() - start_time)])
        f_out.close()
        K.clear_session()
