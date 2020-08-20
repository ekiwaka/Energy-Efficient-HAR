# import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow
# import split_HAPT
import pandas as pd
import sklearn
# import data_processing_acti
import scipy
from scipy import signal, stats
# import os
# import csv
import time
from sklearn.decomposition import PCA
from numpy.random import seed
from sklearn.model_selection import train_test_split
import pickle
# from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")


# the path of HAPT_Data_Set dir
ROOT = "HAPT Data Set/"

# config path and intermediate files
DATA_SET_DIR = ROOT + "RawData/"
PROCESSED_DATA_DIR = ROOT + "Processed/"
LABLE_FILE = DATA_SET_DIR + "labels.txt"
INTERMEDIATE_DIR = PROCESSED_DATA_DIR + "intermediate/"

MY_LABELS = ['Walking', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Downstairs']

PLOT_ALL =  False
DO_PCA = True
DO_LP_FILTERING = True
DO_HP_FILTERING = True  # remove gravity

# parameters for pre-processing and features
MOVING_AVERAGE_WINDOW_SIZE = 3  # optimal
BUTTERWORTH_CUTTING_FREQUENCY = 0.2  # filter gravity
BUTTERWORTH_ORDER = 4

FEATURE_WINDOW_SIZE = 20 * 5  # 20Hz, 5seconds
OVERLAP = 0.5  # 50%

# pca_dims = 50
# TESTING_DATASET_SIZE = 0.2


def get_file_name_by_ids(exp_id, user_id):
    if exp_id < 10:
        exp_str = "0" + str(exp_id)
    else:
        exp_str = "" + str(exp_id)
    if user_id < 10:
        user_str = "0" + str(user_id)
    else:
        user_str = "" + str(user_id)
    acc_file = "acc_exp" + exp_str + "_user" + user_str + ".txt"
    gyro_file = "gyro_exp" + exp_str + "_user" + user_str + ".txt"
    return [acc_file, gyro_file]


def cat_acc_and_gyro(exp_id, user_id):
    acc_data = pd.read_csv(PROCESSED_DATA_DIR + get_file_name_by_ids(exp_id, user_id)[0], sep=" ", header=None)
    gyro_data = pd.read_csv(PROCESSED_DATA_DIR + get_file_name_by_ids(exp_id, user_id)[1], sep=" ", header=None)
    data = pd.concat([acc_data, gyro_data], axis=1, sort=False, ignore_index=True)
    data.to_csv(INTERMEDIATE_DIR + str(exp_id) + "_" + str(user_id) + ".txt", sep=" ", index=False, header=None)


def catenate_data():
    data = pd.read_csv(LABLE_FILE, sep=" ", header=None)
    for (idx, row) in data.iterrows():
        # if not os.path.exists(INTERMEDIATE_DIR + str(row[0]) + "_" + str(row[1]) + ".txt"):
        cat_acc_and_gyro(row[0], row[1])

def extract_features():
    label_data = pd.read_csv(LABLE_FILE, sep=" ", header=None)
    new_data = pd.DataFrame()
    sm = []
    for (idx, row) in label_data.iterrows():
        data = pd.read_csv(INTERMEDIATE_DIR + str(row[0]) + "_" + str(row[1]) + ".txt", sep=" ", header=None)
        exp_id = row[0]
        user_id = row[1]
        start = row[3]
        end = row[4]
        label = row[2]
        sm.append(label)
        sub_dataframe = pd.concat([sm, data])
        #window_and_extract_features(data, exp_id, user_id, label, start, end)
        new_data = new_data.append(sub_dataframe)
    print("feature matrix shape before PCA: " + str(new_data.shape))  # shape of raw features
    new_data.to_csv(INTERMEDIATE_DIR + "HAPT_all.txt", sep=" ", index=False, header=None)



def rolling_mean_filter(data):
    # print(file_name)
    # name = os.path.basename(file_name)
    # data = pd.read_csv(file_name, sep=" ", header=None)
    # if not DO_LP_FILTERING:
    #     data.to_csv(PROCESSED_DATA_DIR + name, sep=" ", index=False, header=None)
    #     return
    rolling_mean = data.rolling(window=MOVING_AVERAGE_WINDOW_SIZE).mean().fillna(0)
    #     plt.plot(data.iloc[250:500,0], color="red", label="raw")
    #     plt.plot(rolling_mean.iloc[250:500,0], color="green", label="filtered")
    #     plt.title("Low-pass filter")
    #     plt.xlabel("time")
    #     plt.ylabel("acceleration")
    #     plt.legend(['raw', 'filtered'], loc = 0, ncol = 2)
    #     plt.show()

    # print(rolling_mean)

    return rolling_mean
    # rolling_mean.to_csv(PROCESSED_DATA_DIR + name, sep=" ", index=False, header=None)


def butterworth_filter(data):
    # data = pd.read_csv(file_name, sep=" ", header=None)
    # name = os.path.basename(file_name)
    # if not DO_HP_FILTERING:
    #     data.to_csv(PROCESSED_DATA_DIR + name, sep=" ", index=False, header=None)
    #     return
    #     plt.plot(data.iloc[250:2000,0], color="red", label="raw")
    nyq = 0.5 * 50  # sampling frequency = 20Hz
    normal_cutoff = BUTTERWORTH_CUTTING_FREQUENCY / nyq
    b, a = signal.butter(BUTTERWORTH_ORDER, normal_cutoff, 'high', analog=False)
    data_0 = np.array(data.iloc[:, 0])
    data_1 = np.array(data.iloc[:, 1])
    data_2 = np.array(data.iloc[:, 2])
    out_0 = signal.filtfilt(b, a, data_0)
    out_1 = signal.filtfilt(b, a, data_1)
    out_2 = signal.filtfilt(b, a, data_2)
    data.iloc[:, 0] = out_0
    data.iloc[:, 1] = out_1
    data.iloc[:, 2] = out_2
    #     plt.plot(data.iloc[250:2000,0], color="green", label="filtered")
    #     plt.title("High-pass filter")
    #     plt.xlabel("time")
    #     plt.ylabel("acceleration")
    #     plt.legend(['raw', 'filtered'], loc = 0, ncol = 2)
    #     plt.show()

    return data
    # data.to_csv(PROCESSED_DATA_DIR + name, sep=" ", index=False, header=None)


def sort_func(file_name):
    return int(file_name.split("_")[0])


def calculate_features_for_each_column(column_data):
    mean = column_data.mean()
    max = column_data.max()
    min = column_data.min()
    med = column_data.median()
    skew = column_data.skew()
    kurt = column_data.kurt()
    std = column_data.std()
    # iqr = column_data.quantile(.75) - column_data.quantile(.25)
    # z_crossing = zero_crossing_rate(column_data)
    energy = np.sum(abs(column_data) ** 2) / FEATURE_WINDOW_SIZE
    # print(energy)
    f, p = scipy.signal.periodogram(column_data)
    mean_fre = np.sum(f * p) / np.sum(p)
    # max_energy_fre = np.asscalar(f[pd.DataFrame(p).idxmax()])
    # median_fre = weighted_median(f, p)
    return [mean, max, min, med, skew, kurt, std, energy, mean_fre]


def calculate_features_between_columns(column_data_1, column_data_2):
    series_1 = pd.Series(column_data_1)
    series_2 = pd.Series(column_data_2)
    corr = series_1.corr(series_2)
    return [corr]



# *****************************************#
#         4.feature normalization          #
# *****************************************#
def normalize_data(features):
    # features = pd.read_csv(FEATURE_FILE, sep=" ", header=None)
    # print(features.head(5))
    for column in features.columns[:]:
        # print(column)
        col = features[[column]].values.astype(float)
        # print(col)
        min_max_scaler = sklearn.preprocessing.MinMaxScaler()
        normalized_col = min_max_scaler.fit_transform(col)
        features.iloc[:, column] = normalized_col
    # features.to_csv('feature-check.txt', sep=" ", index=False, header=None)

    return features
    # features.to_csv(NORMALIZED_FEATURE_FILE, sep=" ", index=False, header=None)
    # features.to_csv('PCA=30' + REDUCED_FEATURE_FILE, sep=" ", index=False, header=None)


# *****************************************#
#           5.feature reduction            #
# *****************************************#
def pca(features, pca_dims):
    # features = pd.read_csv(NORMALIZED_FEATURE_FILE, sep=" ", header=None)
    # if not DO_PCA:
    #     features.to_csv(REDUCED_FEATURE_FILE, sep=" ", index=False, header=None)
    #     return
    without_label = features.iloc[:, :]
    pca = PCA().fit(without_label)
    # if PLOT_ALL:
    #     plt.plot(np.cumsum(pca.explained_variance_ratio_))
    #     plt.xlabel('number of components')
    #     plt.ylabel('cumulative explained variance');
    #     plt.show()
    pca = PCA(n_components=pca_dims)
    pca.fit(without_label)
    X_pca = pca.transform(without_label)
    df = pd.DataFrame(X_pca)
    # new_data = pd.concat([features.iloc[:, :3], df], axis=1, sort=False, ignore_index=True)  # add labels
    # datafile = 'test-PCA=' + str(pca_dims) + '.txt'
    # df.to_csv(datafile, sep=" ", index=False, header=None)
    return df




def separate():
    datafile = 'actitracker-new.txt'
    test_x = pd.DataFrame()
    test_y = pd.DataFrame(dtype=int)
    data = pd.read_csv(datafile, sep=" ", header=None)
    # data = data.iloc[:, 1:]  # remove user id and experiment id
    labels = data.iloc[:, 0]
    without_labels = data.iloc[:, 1:]

    seed(2020)
    # split data
    X_train, X_test, y_train, y_test = train_test_split(without_labels, labels, test_size=0.3) #test_data_ratio)
    test_x = test_x.append(X_test)
    print(y_test)

    test_y = test_y.append(y_test)
    test_y = test_y.astype(int)
    test_x.to_csv("X_test.txt", sep=" ", index=False, header=None)
    test_y.T.to_csv("y_test.txt", sep=" ", index=False, header=None)

    print(test_y.info())



def main():
    seed(2020)
    # separate()
    # filter_data()
    # catenate_data()
    # extract_features()

    rate = 1
    pca_dims = 30
    compress_rate = 0.2
    sparse_rate = 0.2

    seed(2020)
    # X_train, X_test, y_train, y_test = split_HAPT.main(pca_dims, rate, 0.3, '')
    moderated = str(int(20 // rate))
    # if pca_dims == 0:
    #     pca_dims = 60

    # y_train = y_train - 1
    # y_test = y_test - 1

    start_time = time.time()

    model = pickle.load(open('model/' + moderated + 'Hz/pickled/' + 'pickled_' + 'decompressed_pca=' + str(
        pca_dims) + '_compress_rate=' + str(compress_rate) + '_sparse_rate=' + str(sparse_rate) + '.hdf5', 'rb'))
        
    load_time = time.time() - start_time
    
    print("Time to load :", load_time)

    X_test_file_1 = pd.read_csv("X_test.txt", sep=" ", header=None)
    y_test_file_1 = pd.read_csv("y_test.txt", header=None)
    # print(y_test_file_1)
    y_test_file_1 = y_test_file_1 - 1
    # y_test_file_1.to_csv("y_test2.txt", header=None)
    # print(y_test_file_1)

    # with open('tsssst.txt', 'w', newline='') as f_out:
    #     writer = csv.writer(f_out)
    # f_out.close()

    # data_processing_acti.main()

    
    i=0
    sec = 20*3 - 1

    while i+sec < len(X_test_file_1):
        start_time = time.time()
        X_test_file = X_test_file_1.iloc[i:i+sec]

        rm = rolling_mean_filter(X_test_file)
        bf = butterworth_filter(rm)

        # data = pd.concat([y_test_file_1, bf], axis=1, sort=False, ignore_index=True)
        # data.to_csv('check.txt', sep=" ", index=False, header=None)

        data = bf
        # data.to_csv('check_2.txt', sep=" ", index=False, header=None)
        # print(data.iloc[98,:])
        s=0

        stride = 10
        s = 0
        start = 0
        window_size = 20
        end = window_size #window_size
        feature_list = []
        label_list = []
        # print(2 * (len(data) // window_size))
        while end <= len(X_test_file):

            s += 1
            row_list = []

            for direction in [0, 1, 2]: #, 3, 4]:  # , 3, 4, 5]:  # x,y,z axis for acc and gyro
                column_data = data.iloc[start:start + stride, direction]
                # print(len(data))
                features = calculate_features_for_each_column(column_data)
                row_list.extend(features)
                # add correlation features
                other_column = -1
                if direction == 2:
                    other_column = 0
                # if direction == 4:
                #     other_column = 2
                else:
                    other_column = direction + 1
                corr = calculate_features_between_columns(column_data,
                                                          data.iloc[start:start + stride, other_column])
                row_list.extend(corr)
            feature_list.append(row_list)
            label = stats.mode(y_test_file_1.iloc[start:start + stride, 0])[0][0]
            label_list.append(label)

            start = int(start + stride * (1 - OVERLAP))
            end = int(end + stride * (1 - OVERLAP))

        result = pd.DataFrame(feature_list)
        result2 = pd.DataFrame(label_list)

        # result.to_csv('feat-check.txt', sep=" ", index=False, header=None)
        # result2.to_csv('label-check.txt', index=False, header=None)

        feat1 = normalize_data(result)
        # feat = pca(feat1,pca_dims)
        feat = feat1
        # print(features)

        # print(moderated)
        # print(pca_dims,compress_rate)

        # print("SEEEEEEEEEEEEEEEEEEE")
        # i+=99

        # pred_test = model.predict(np.expand_dims(X_test, axis=2), batch_size=32)
        # print("------ TEST ACCURACY ------")
        # testing_acc = (accuracy_score(y_test, np.argmax(pred_test, axis=1)))
        # print(testing_acc)
        # # print((confusion_matrix(y_test, np.argmax(pred_test, axis=1))))
        # t = (time.time() - start_time)
        #
        # # pred_train = model.predict(np.expand_dims(X_train, axis=2), batch_size=32)
        # # print("------ TRAIN ACCURACY ------")
        # # training_acc = (accuracy_score(y_train, np.argmax(pred_train, axis=1)))
        # # print(training_acc)
        # # print((confusion_matrix(y_train, np.argmax(pred_train, axis=1))))
        #
        # i=0
        #
        # while i < len(X_test):
        # print(result)

        #start_time = time.time()
        lewl = model.predict(np.expand_dims(feat, axis=2))
        # print("Prediction results shape:", lewl.shape)
        bas = time.time() - start_time
        tf_pred_dataframe = pd.DataFrame(lewl)
        # tf_pred_dataframe.columns = MY_LABELS
        # print(tf_pred_dataframe, result2)
        maxi = tf_pred_dataframe.idxmax(axis=1)
        score = tf_pred_dataframe.max(axis=1)
        fine = pd.concat([tf_pred_dataframe, result2, maxi, score], axis=1)
        print(fine)
        print("Prediction time:", bas)
        # print(tf_pred_dataframe.idxmax(axis=1))
        # print(tf_pred_dataframe[:0])
        # print(np.argmax(lewl, axis=-1))

        #predicted_ids = np.argmax(lewl, axis=-1)
        # predicted_labels = MY_LABELS[predicted_ids]
        #
        # print(predicted_labels, result2)

        # print(tf_pred_dataframe.head())
        # _, accuracy = model.evaluate(np.expand_dims(result, axis=2), result2, batch_size = 20, verbose=0)
        # print(accuracy)
        # training_acc = (accuracy_score(result2, np.argmax(lewl, axis=1)))
        # print(training_acc, bas)

        i= i + sec + 1

        # with open('tsssst.txt', 'a', newline='') as f_out:
        #     writer = csv.writer(f_out)
        #     # writer.writerow(['Sampling rate', 'PCA dims', 'Time'])
        #     writer.writerow([training_acc, bas])
        # f_out.close()


if __name__ == '__main__':
    main()