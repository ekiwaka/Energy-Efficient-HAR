from copy import deepcopy

import numpy as np
from HAPT import split_HAPT
# from utils.load_cifar import load_data
# from models import resnet, densenet, inception, vggnet
from compression import prune_weights, save_compressed_weights
from keras.models import load_model
from numpy.random import seed
from sklearn.metrics import accuracy_score

# def save_history(history, result_dir, prefix):
#     loss = history.history['loss']
#     acc = history.history['acc']
#     val_loss = history.history['val_loss']
#     val_acc = history.history['val_acc']
#     nb_epoch = len(acc)
#
#     with open(os.path.join(result_dir, '{}_result.txt'.format(prefix)), 'w') as fp:
#         fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
#         for i in range(nb_epoch):
#             fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
#                 i, loss[i], acc[i], val_loss[i], val_acc[i]))
#
#
# def schedule(epoch):
#     if epoch < 60:
#         return 0.1
#     elif epoch < 120:
#         return 0.01
#     elif epoch < 160:
#         return 0.001
#     else:
#         return 0.0001
#
#
# def training():
batch_size = 32
# epochs = 200
fine_tune_epochs = 30
pca_dims = 20
# lr = 0.1

# prune weights
# save masks for weight layers
X_train, X_test, y_train, y_test = split_HAPT.main(pca_dims)

y_train = y_train - 1

# datagen = DataGenerator()
# data_iter = datagen.flow(X_train, y_train, batch_size=batch_size, shuffle=True)
masks = {}
layer_count = 0

moderated = str(50)

model_path = 'model/' + moderated + 'Hz/' + 'pca_testing_bigger_cnn' + str(pca_dims) + '.hdf5'
model = load_model(model_path)

pred_train = model.predict(np.expand_dims(X_train, axis=2), batch_size=32)
# print("------ TRAIN ACCURACY ------")
accuracy = (accuracy_score(y_train, np.argmax(pred_train, axis=1)))
print(accuracy)

seed(2020)
# not compress first convolution layer
first_conv = True
for layer in model.layers:
    weight = layer.get_weights()
    print(len(weight))
    if len(weight) >= 2:
        if not first_conv:
            w = deepcopy(weight)
            tmp, mask = prune_weights(w[0], compress_rate=0.5)  # args.compress_rate)
            masks[layer_count] = mask
            w[0] = tmp
            layer.set_weights(w)
        else:
            first_conv = False
    layer_count += 1

# score = model.evaluate(np.expand_dims(X_train, axis=2), y_test, verbose=0)
# print('val loss: {}'.format(score[0]))
# print('val acc: {}'.format(score[1]))


n_classes = 12

# Convert to one hot encoding vector
y_train_dynamic_oh = np.eye(n_classes)[y_train]

# fine-tune

# while accuracy >= 0.91:
#     # apply masks
#     for layer_id in masks:
#         w = model.layers[layer_id].get_weights()
#         w[0] = w[0] * masks[layer_id]
#         model.layers[layer_id].set_weights(w)
#         model.fit(np.expand_dims(X_train, axis=2), y_train_dynamic_oh,
#                   batch_size=32, epochs=5, verbose=2, validation_split=0.2)  # , callbacks=[cp_cb])
#         accuracy = (accuracy_score(y_train, np.argmax(pred_train, axis=1)))
# for i in range(fine_tune_epochs):
#     for _ in range(X_train.shape[0] // batch_size):
#         # X, Y = data_iter.next()
#         # train on each batch
#         model.train_on_batch(np.expand_dims(X_train, axis=2), y_train_dynamic_oh)
#         # apply masks
#         for layer_id in masks:
#             w = model.layers[layer_id].get_weights()
#             w[0] = w[0] * masks[layer_id]
#             model.layers[layer_id].set_weights(w)
# score = model.evaluate(X_test, y_test, verbose=0)
# print('val loss: {}'.format(score[0]))
# print('val acc: {}'.format(score[1]))

# save compressed weights
compressed_name = 'compressed_true_0.9_bigs'  # .format(args.model)
# model.save('compressed_new.hdf5')
save_compressed_weights(model, compressed_name)

# if __name__ == '__main__':
#     pca_dims = 30
#     moderated = str(50)
#
#     training()


# model_path = 'compressed.hdf5'
# model = load_model('compressed_new.hdf5')
#
# seed(2020)
# X_train, X_test, y_train, y_test = split.main(pca_dims)
#
# y_train = y_train - 1
# y_test = y_test - 1
# print(moderated)
#
# pred_train = model.predict(np.expand_dims(X_train, axis=2), batch_size=32)
# print("------ TRAIN ACCURACY ------")
# print((accuracy_score(y_train, np.argmax(pred_train, axis=1))))
# # print((confusion_matrix(y_train, np.argmax(pred_train, axis=1))))
#
# pred_test = model.predict(np.expand_dims(X_test, axis=2), batch_size=32)
# print("------ TEST ACCURACY ------")
# print((accuracy_score(y_test, np.argmax(pred_test, axis=1))))
# # print((confusion_matrix(y_test, np.argmax(pred_test, axis=1))))
