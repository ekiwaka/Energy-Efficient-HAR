import pickle
import csv
import tensorflow

# from keras.models import load_model

# with open('param_list.txt', 'w', newline='') as f_out:
#     writer = csv.writer(f_out)
#     writer.writerow(['Sampling rate', 'PCA dims', 'Compress rate', 'Params'])
# f_out.close()

x = [15, 18, 21, 24, 27, 30]
# x = [7, 8, 10, 20]
# x = [0, 20, 30, 40, 50]
y = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# samp_rate = [10, 5, 2.5, 2, 1.25, 1]
samp_rate = [5, 2.5, 2, 1.25, 1]

s_rate = 0.1

for rate in samp_rate:
    moderated = str(int(50 // rate))
    for pca_dims in x:
        for compress_rate in y:
            # for s_rate in y:

            model = pickle.load(open('model/' + moderated + 'Hz/pickled/' + 'pickled_' + 'acc_decompressed_pca=' + str(pca_dims) + '_compress_rate=' + str(compress_rate) + '_sparse_rate=' + str(
                    s_rate) + '.hdf5', 'rb'))
            params = model.count_params()
            

            print(rate, pca_dims, compress_rate, params)

            with open('param_list.txt', 'a', newline='') as f_out:
                writer = csv.writer(f_out)
                # writer.writerow(['Sampling rate', 'PCA dims', 'Time'])
                writer.writerow([moderated, pca_dims, compress_rate, params])
            f_out.close()

# model1 = load_model('model/' + moderated + 'Hz/pruned/' + 'decompressed_pca=' + str(
#     pca_dims) + '_compress_rate=' + str(compress_rate) + '.hdf5')
# model2= load_model('model/' + moderated + 'Hz/pca=' + str(pca_dims) + '.hdf5')
#
#
# print(model.summary())
# print(model1.summary())