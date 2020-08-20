# importing libraries and dependecies

import numpy as np
import sklearn
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import csv
from scipy import signal, stats
from numpy.random import seed
from sklearn.decomposition import PCA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



# the path of HAPT_Data_Set dir
ROOT = "Acti-tracker Data Set/"

# config path and intermediate files
DATA_SET_DIR = ROOT + "RawData/"
if not os.path.exists(DATA_SET_DIR):
    os.mkdir(DATA_SET_DIR)

PROCESSED_DATA_DIR = ROOT + "Processed/"
if not os.path.exists(PROCESSED_DATA_DIR):
    os.mkdir(PROCESSED_DATA_DIR)

LABLE_FILE = DATA_SET_DIR + "labels.txt"
INTERMEDIATE_DIR = PROCESSED_DATA_DIR + "intermediate/"
if not os.path.exists(INTERMEDIATE_DIR):
    os.mkdir(INTERMEDIATE_DIR)

# SAMPLED_DIR = "sampled/"
FEATURE_FILE = "features.txt"
NORMALIZED_FEATURE_FILE = "normalized_features.txt"
REDUCED_FEATURE_FILE = "reduced_features.txt"
MY_LABELS = ['Walking', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Downstairs']

# switches
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

pca_dims = 50
TESTING_DATASET_SIZE = 0.2

# setting up a random seed for reproducibility
random_seed = 611
np.random.seed(random_seed)

# matplotlib inline
plt.style.use('ggplot')


# defining function for loading the dataset
def readData(filePath, separator):
    data = pd.read_csv(filePath, sep=separator, header=None, na_values=';')
    return data


def clean_data(dataset):
    dataset_acc = dataset.iloc[:, 3:]
    dataset_acc.to_csv('Acti-tracker Data Set/actitracker_acc.txt', sep=" ", index=False, header=None)

    dataset_labels = dataset.iloc[:,0:2]
    dataset_labels.to_csv('Acti-tracker Data Set/actitracker_labels.txt', sep=" ", index=False, header=None)


def cat_acc_label():
    acc_data = pd.read_csv(PROCESSED_DATA_DIR + 'actitracker_acc.txt', sep=" ", header=None)
    label_data = pd.read_csv('Acti-tracker Data Set/actitracker_labels.txt', sep=" ", header=None)
    data = pd.concat([label_data, acc_data], axis=1, sort=False, ignore_index=True)
    data.to_csv(PROCESSED_DATA_DIR + 'actitracker.txt', sep=" ", index=False, header=None)



# defining the function to plot a single axis data
def plotAxis(axis, x, y, title):
    axis.plot(x, y)
    axis.set_title(title)
    axis.xaxis.set_visible(False)
    axis.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    axis.set_xlim([min(x), max(x)])
    axis.grid(True)


# defining a function to plot the data for a given activity
def plotActivity(activity, data):
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(15, 10), sharex=True)
    plotAxis(ax0, data['timestamp'], data['x-axis'], 'x-axis')
    plotAxis(ax1, data['timestamp'], data['y-axis'], 'y-axis')
    plotAxis(ax2, data['timestamp'], data['z-axis'], 'z-axis')
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.9)
    plt.show()


# low-pass filter
def rolling_mean_filter(file_name):
    # print(file_name)
    name = os.path.basename(file_name)
    data = pd.read_csv(file_name, sep=" ", header=None)
    if not DO_LP_FILTERING:
        data.to_csv(PROCESSED_DATA_DIR + name, sep=" ", index=False, header=None)
        return
    rolling_mean = data.rolling(window=MOVING_AVERAGE_WINDOW_SIZE).mean().fillna(0)
    #     plt.plot(data.iloc[250:500,0], color="red", label="raw")
    #     plt.plot(rolling_mean.iloc[250:500,0], color="green", label="filtered")
    #     plt.title("Low-pass filter")
    #     plt.xlabel("time")
    #     plt.ylabel("acceleration")
    #     plt.legend(['raw', 'filtered'], loc = 0, ncol = 2)
    #     plt.show()
    rolling_mean.to_csv(PROCESSED_DATA_DIR + name, sep=" ", index=False, header=None)


def butterworth_filter(file_name):
    data = pd.read_csv(file_name, sep=" ", header=None)
    name = os.path.basename(file_name)
    if not DO_HP_FILTERING:
        data.to_csv(PROCESSED_DATA_DIR + name, sep=" ", index=False, header=None)
        return
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
    data.to_csv(PROCESSED_DATA_DIR + name, sep=" ", index=False, header=None)


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
#      1.filter all the raw data file      #
# *****************************************#




# segmenting the time series
def segment_signal(data, window_size=100):
    stride = 10
    s = 0
    start = 0
    end = window_size
    feature_list = []
    label_list = []
    print(2 * (len(data) // window_size))
    while end <= 2 * (len(data) // window_size):

        s += 1
        row_list = []

        for direction in [2, 3, 4]:  # , 3, 4, 5]:  # x,y,z axis for acc and gyro
            column_data = data.iloc[start:start + window_size, direction]
            features = calculate_features_for_each_column(column_data)
            row_list.extend(features)
            # add correlation features
            other_column = -1
            # if direction == 2:
            #     other_column = 0
            if direction == 4:
                other_column = 2
            else:
                other_column = direction + 1
            corr = calculate_features_between_columns(column_data,
                                                      data.iloc[start:start + window_size, other_column])
            row_list.extend(corr)
        feature_list.append(row_list)
        label = stats.mode(data.iloc[start: start + window_size, 1])[0][0]
        label_list.append(label)

        start = int(start + stride * (1 - OVERLAP))
        end = int(end + stride * (1 - OVERLAP))
        print(end)

    result = pd.DataFrame(feature_list)
    result2 = pd.DataFrame(label_list)
    result.to_csv(FEATURE_FILE, sep=" ", index=False, header=None)
    result2.to_csv('labels.txt', sep=" ", index=False, header=None)
    # print(feature_list)
    # print("xxxxxxxxxxxxxxxx")
    print(s)
    return result,result2


# *****************************************#
#         4.feature normalization          #
# *****************************************#
def normalize_data():
    features = pd.read_csv(FEATURE_FILE, sep=" ", header=None)
    # print(features.head(5))
    for column in features.columns[:]:
        col = features[[column]].values.astype(float)
        min_max_scaler = sklearn.preprocessing.MinMaxScaler()
        normalized_col = min_max_scaler.fit_transform(col)
        features.iloc[:, column] = normalized_col
    features.to_csv(NORMALIZED_FEATURE_FILE, sep=" ", index=False, header=None)
    features.to_csv('PCA=30' + REDUCED_FEATURE_FILE, sep=" ", index=False, header=None)


# *****************************************#
#           5.feature reduction            #
# *****************************************#
def pca(pca_dims):
    features = pd.read_csv(NORMALIZED_FEATURE_FILE, sep=" ", header=None)
    if not DO_PCA:
        features.to_csv(REDUCED_FEATURE_FILE, sep=" ", index=False, header=None)
        return
    without_label = features.iloc[:, :]
    pca = PCA().fit(without_label)
    if PLOT_ALL:
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance');
        plt.show()
    pca = PCA(n_components=pca_dims)
    pca.fit(without_label)
    X_pca = pca.transform(without_label)
    df = pd.DataFrame(X_pca)
    # new_data = pd.concat([features.iloc[:, :3], df], axis=1, sort=False, ignore_index=True)  # add labels
    datafile = 'PCA=' + str(pca_dims) + REDUCED_FEATURE_FILE
    df.to_csv(datafile, sep=" ", index=False, header=None)


# *****************************************#
#           7.data plotting                #
# *****************************************#

def plot_report(y_test, test_predict, title):
    precision = precision_score(y_test, test_predict, average=None)
    recall = recall_score(y_test, test_predict, average=None)
    f1 = f1_score(y_test, test_predict, average=None)
    plt.tight_layout(pad=0)
    plt.plot(precision, color="red")
    plt.plot(recall, color="green")
    plt.plot(f1, color="blue")
    plt.margins(x=0)
    plt.gcf().subplots_adjust(bottom=0.5)
    plt.title(title)
    plt.legend(["precision", "recall", "f1-score"])
    plt.xticks(np.arange(0, 12, step=1), MY_LABELS, rotation=60, fontsize=6)
    plt.show()


def plot_label_distribution(y_train):
    # print(y_train.value_counts())
    label_distribution = y_train.value_counts().reset_index()
    sorted = label_distribution.sort_values(['index'])
    sorted.set_index('index').plot(kind='bar')
    plt.title("Distribution of labels in training data")
    plt.xlabel("")
    plt.ylabel("number of samples")
    plt.gcf().subplots_adjust(bottom=0.5)
    plt.gca().get_legend().remove()
    plt.xticks(np.arange(0, 12, step=1), MY_LABELS, rotation=60, fontsize=6)
    plt.show()




def main():
    ''' Main Code '''

    seed(2020)

    file = 'Acti-tracker Data Set/actitracker_raw.txt'
    dataset = data = pd.read_csv(file, sep=",", header=None, na_values=';')

    clean_data(dataset)

    label_file = 'Acti-tracker Data Set/actitracker_labels.txt'
    datafile = 'Acti-tracker Data Set/actitracker_acc.txt'

    file2 = 'Acti-tracker Data Set/actitracker_acc.txt'
    rolling_mean_filter(file2)

    # use processed data
    butterworth_filter(PROCESSED_DATA_DIR + 'actitracker_acc.txt')

    cat_acc_label()

    file3 = 'Acti-tracker Data Set/Processed/actitracker.txt'
    dataset = pd.read_csv(file3, sep=" ", header=None, na_values=';')
    result, result2 = segment_signal(dataset)

    normalize_data()
    # print(len(dataset))

    pca(7)
    pca(8)
    pca(10)
    pca(15)
    pca(20)


if __name__ == '__main__':
    main()





# import sys
# import pyRAPL

# pca_dims = 20

# ...
# Instructions to be evaluated.
# ...

# Select sample rate moderator. Default sample rate is 50Hz. Moderated sample rate would be default//rate
# s_rate = [10, 5, 2.5, 2, 1.25, 1]
# p = [0, 20, 30, 40, 50]

# with open('training_results.txt', 'a', newline='') as f_out:
#     writer = csv.writer(f_out)
#     writer.writerow(['Sampling rate', 'PCA dims', 'Time'])
#     f_out.close()
#
# # for rate in s_rate:
# #     for pca_dims in p:
# #         # rate = 1
#
# rate = 1
# pca_dims = 30
# moderated = str(int(50 // rate))
# print(moderated)
# #
# datafile_test = 'normalized_features.txt'
# with open(datafile_test) as f:
#     container = f.readlines()
#
# result = []
# for line in container:
#     tmp1 = line.strip()
#     tmp2 = tmp1.replace('  ', ' ')  # removes inconsistent blank spaces
#     tmp_ary = list(map(float, tmp2.split(' ')))
#     result.append(tmp_ary)
# without_labels = np.array(result)
# # data = pd.read_csv(datafile_test, sep=" ", header=None)
# # without_labels = data.iloc[:, :]
#
# label_file = 'labels.txt'
# with open(label_file) as f:
#     container = f.readlines()
#
# result = []
# for line in container:
#     num_str = line.strip()
#     result.append(int(num_str))
# labels= np.array(result)
# # data2 = pd.read_csv(label_file, sep=" ", header=None)
# # labels = data2.iloc[:, 0]
#
# X_train, X_test, y_train, y_test = train_test_split(without_labels, labels, test_size=0.3)
# start_time = time.time()
# # Load all train and test data (* dynamic and static data are mixed.)
#
# print(y_test.shape)
# # Convert (1, 2, 3) labels to (0, 1, 2)
# y_train = y_train - 1
# y_test = y_test -1
# print(y_train.shape)
#
# print(("test_dynamic shape: ", X_test.shape))
#
# print(X_train.shape)
# print(y_test.shape)
#
# n_classes = 6
#
# # Convert to one hot encoding vector
# y_train_dynamic_oh = np.eye(n_classes)[y_train]
# # y_train_dynamic_oh = np.delete(y_train_dynamic_oh, 0, 1)
#
# print(y_train_dynamic_oh.shape)
# print(y_train.shape)
#
# print(y_train)
# print(y_train_dynamic_oh)
#
# if pca_dims == 0:
#     pca_dims = 60
#
# # Fit 1d CNN for dynamic HAR
#
# seed(2020)
# model = Sequential()
# model.add(Conv1D(100, 6, input_shape=(pca_dims, 1), activation='relu'))
# model.add(MaxPooling1D(8))
# model.add(Flatten())
# model.add(Dense(6, activation='relu'))
# model.add(Dense(6, activation='relu'))
# # model.add(Dense(128, activation='relu'))
# model.add(Dense(6, activation='softmax'))
# model.add(Dropout(0.5))
# #
# adam = Adam(lr=0.0004, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
# model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])
#
# # Summarize layers
# print((model.summary()))
#
# # Save model image
# # if not os.path.exists('fig_har_hapt.png'):
# #     model_file = 'fig_har_hapt.png'
# #     plot_model(model, to_file=model_file)
#
#
# new_dir = 'model/' + moderated + 'Hz/weights/'
# if not os.path.exists(new_dir):
#     os.makedirs(new_dir)
# fpath = new_dir + moderated + 'Hz' + '_pca' + str(pca_dims) + '_weights.{epoch:02d}-{val_accuracy:.2f}.hdf5'
#
# cp_cb = ModelCheckpoint(fpath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto', period=1)
#
# # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# # To disable learning, the below code - two lines - is commented.
# # To enable learning uncomment the below two lines of code.
#
# model.fit(np.expand_dims(X_train, axis=2), y_train_dynamic_oh,
#           batch_size=32, epochs=50, verbose=2, validation_split=0.2, callbacks=[cp_cb])
# # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# model.save('model/' + moderated + 'Hz/' + 'pca=' + str(pca_dims) + '.hdf5')
# print("--- %s seconds ---" % (time.time() - start_time))
# del model
#
# with open('training_results.txt', 'a', newline='') as f_out:
#     writer = csv.writer(f_out)
#     # writer.writerow(['Sampling rate', 'PCA dims', 'Time'])
#     writer.writerow([rate, pca_dims, (time.time() - start_time)])
# f_out.close()
# K.clear_session()
#
# model_path = 'model/' + moderated + 'Hz/pca=' + str(
#     pca_dims) + '.hdf5'
# model = load_model(model_path)
#
# pred_train = model.predict(np.expand_dims(X_train, axis=2), batch_size=32)
# print("------ TRAIN ACCURACY ------")
# training_acc = (accuracy_score(y_train, np.argmax(pred_train, axis=1)))
# print(training_acc)
# # print((confusion_matrix(y_train, np.argmax(pred_train, axis=1))))
#
# pred_test = model.predict(np.expand_dims(X_test, axis=2), batch_size=32)
# print("------ TEST ACCURACY ------")
# testing_acc = (accuracy_score(y_test, np.argmax(pred_test, axis=1)))
# print(testing_acc)
# # print((confusion_matrix(y_test, np.argmax(pred_test, axis=1))))