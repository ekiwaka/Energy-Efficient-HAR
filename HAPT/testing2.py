# import tensorflow as tf
# from keras.models import load_model
#
# moderated = str(50)
# pca_dims = 30
# compress_rate = 0.3
# model_path = 'model/' + moderated + 'Hz/pruned/' + 'decompressed_pca=' + str(
#                     pca_dims) + '_compress_rate=' + str(compress_rate) + '.hdf5'
#
# # model = load_model(model_path)
# # converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
# # converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
# # tflite_quant_model = converter.convert()
#
# converter = tf.lite.TFLiteConverter.from_keras_model_file(model_path)
# converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
# tfmodel = converter.convert()
# open("model.tflite","wb").write(tfmodel)

# import tensorflow as tf
import time
import pickle
import split_HAPT
import numpy as np
import tflite_runtime.interpreter as tflite
from keras.models import load_model
from sklearn.metrics import accuracy_score

moderated = str(50)
pca_dims = 30
compress_rate = 0.3

# model_path = 'model/' + moderated + 'Hz/pruned/' + 'decompressed_pca=' + str(
#                     pca_dims) + '_compress_rate=' + str(compress_rate) + '.hdf5'
# interpreter = tflite.Interpreter('model.tflite')

# seed(2020)
rate = 1
X_train, X_test, y_train, y_test = split_HAPT.main(pca_dims, rate, 0.3, '')
y_test = y_test - 1

verbose = 1

# moderated = str(int(50 // rate))
print(moderated, pca_dims, compress_rate)

# start_time = time.time()
# model = pickle.load(open('model/' + moderated + 'Hz/pruned/' + 'pickled_' + 'decompressed_pca=' + str(
#     pca_dims) + '_compress_rate=' + str(compress_rate) + '.hdf5', 'rb'))
# load_time = time.time() - start_time
#
# start_time = time.time()
# pred_test = model.predict(np.expand_dims(X_test, axis=2), batch_size=32)
# tftime = time.time() - start_time

# %%
#
start_time = time.time()
interpreter = tflite.Interpreter(model_path="model.tflite")
# flowers_model = tf.lite.keras.experimental.load_from_saved_model('model.tflite')
interpreter.allocate_tensors()

# Get input and output tensors.

input_details = interpreter.get_input_details()
print(interpreter.get_input_details())
output_details = interpreter.get_output_details()

x = input_details[0]['shape'][1]

input_data = np.array(np.expand_dims(X_test, axis=2), dtype=np.float32)
# testing_acc1 = (accuracy_score(y_test, np.argmax(pred_test, axis=1)))
# np.array(input_data, dtype=np.float32)
# start_time = time.time()
ok =[]
hmm =[]
# model = load_model('model.tflite')
count = 0
while count < len(X_test):
    interpreter.set_tensor(input_details[0]['index'], np.expand_dims(input_data[count],axis=0))
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)
    ok.append(results)
    count+=1
tfl_time = time.time() - start_time
hmm = np.array(ok)
testing_acc2 = (accuracy_score(y_test, np.argmax(hmm, axis=1)))

# print(tftime, testing_acc1, tfl_time, testing_acc2)
print(tfl_time, testing_acc2)
# label_map = imagenet.create_readable_names_for_imagenet_labels()
# print("Top 1 Prediction: ", output_data.argmax(),label_map[output_data.argmax()], output_data.max(), k+1)

load_time = time.time() - start_time

# y_train = y_train - 1
# y_test = y_test - 1
# print(moderated)
# print(pca_dims,compress_rate)
start_time = time.time()

# pred_test = model.predict(np.expand_dims(X_test, axis=2), batch_size=32)

inference_time = time.time() - start_time
# testing_acc = (accuracy_score(y_test, np.argmax(pred_test, axis=1)))
# print("------ TEST ACCURACY ------")
# print(testing_acc)
# # print((confusion_matrix(y_test, np.argmax(pred_test, axis=1))))
#
# print("------ INFERENCE TIME ------")
# print(inference_time)
#
# print("Accuracy, Load Time, Inference Time")
# print(testing_acc, load_time, inference_time)