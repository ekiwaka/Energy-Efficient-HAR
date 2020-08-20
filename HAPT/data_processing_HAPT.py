import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from numpy.random import seed
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, f1_score

# the path of HAPT_Data_Set dir
ROOT = "HAPT Data Set/"

# config path and intermediate files
DATA_SET_DIR = ROOT + "RawData/"
PROCESSED_DATA_DIR = ROOT + "Processed/"
LABLE_FILE = DATA_SET_DIR + "labels.txt"
INTERMEDIATE_DIR = PROCESSED_DATA_DIR + "intermediate/"
SAMPLED_DIR = "sampled/"
FEATURE_FILE = "features_acc.txt"
NORMALIZED_FEATURE_FILE = "normalized_features_acc.txt"
REDUCED_FEATURE_FILE = "reduced_features_acc.txt"
MY_LABELS = ['WALKING', 'UPSTAIRS', 'DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING', 'STAND_TO_SIT', 'SIT_TO_STAND',
             'SIT_TO_LIE', 'LIE_TO_SIT', 'STAND_TO_LIE', 'LIE_TO_STAND']

# switches
PLOT_ALL = False #True  # False
DO_PCA = True
DO_LP_FILTERING = True
DO_HP_FILTERING = True  # remove gravity
DO_CROSS_VALIDATION = True

# parameters for pre-processing and features
MOVING_AVERAGE_WINDOW_SIZE = 3  # optimal
BUTTERWORTH_CUTTING_FREQUENCY = 0.2  # filter gravity
BUTTERWORTH_ORDER = 4

FEATURE_WINDOW_SIZE = 50 * 3  # 50Hz, 3seconds
OVERLAP = 0.5  # 50%

# pca_dims = 50
TESTING_DATASET_SIZE = 0.2


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
    nyq = 0.5 * 50  # sampling frequency = 50Hz
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


def window_and_extract_features(data, exp_id, user_id, label, start, end):
    feature_list = []
    while True:
        if start + FEATURE_WINDOW_SIZE < end:
            row_list = [exp_id, user_id, label]
            for direction in [0, 1, 2]:# , 3, 4, 5]:  # x,y,z axis for acc and gyro
                column_data = data.iloc[start:start + FEATURE_WINDOW_SIZE, direction]
                features = calculate_features_for_each_column(column_data)
                row_list.extend(features)
                # add correlation features
                other_column = -1
                if direction == 2:
                    other_column = 0
                elif direction == 5:
                    other_column = 3
                else:
                    other_column = direction + 1
                corr = calculate_features_between_columns(column_data,
                                                          data.iloc[start:start + FEATURE_WINDOW_SIZE, other_column])
                row_list.extend(corr)
            feature_list.append(row_list)
            start = (int)(start + FEATURE_WINDOW_SIZE * (1 - OVERLAP))
        else:  # if not enough data points in this window, same method to calculate features
            row_list = [exp_id, user_id, label]
            for direction in [0, 1, 2]:#, 3, 4, 5]:  # x,y,z axis for acc and gyro
                column_data = data.iloc[start:end, direction]
                features = calculate_features_for_each_column(column_data)
                row_list.extend(features)
                other_column = -1
                if direction == 2:
                    other_column = 0
                elif direction == 5:
                    other_column = 3
                else:
                    other_column = direction + 1
                corr = calculate_features_between_columns(column_data, data.iloc[start:end, other_column])
                row_list.extend(corr)
            feature_list.append(row_list)
            break
    result = pd.DataFrame(feature_list)
    # print(feature_list)
    return result


# *****************************************#
#      1.filter all the raw data file      #
# *****************************************#
def filter_data():
    files = os.listdir(DATA_SET_DIR)
    for file in files:
        if not file.startswith("labels"):
            file_name = DATA_SET_DIR + file
            # print(file_name)
            rolling_mean_filter(file_name)
            if file.startswith("acc"):  # gravity only exists in acc data
                # use processed data
                butterworth_filter(PROCESSED_DATA_DIR + file)


# *****************************************#
#        2.cat acc and gyro data           #
# *****************************************#
def catenate_data():
    data = pd.read_csv(LABLE_FILE, sep=" ", header=None)
    for (idx, row) in data.iterrows():
        if not os.path.exists(INTERMEDIATE_DIR + str(row[0]) + "_" + str(row[1]) + ".txt"):
            cat_acc_and_gyro(row[0], row[1])


# *****************************************#
#      3.feature extraction and label      #
# *****************************************#
def extract_features():
    label_data = pd.read_csv(LABLE_FILE, sep=" ", header=None)
    new_data = pd.DataFrame()
    for (idx, row) in label_data.iterrows():
        data = pd.read_csv(INTERMEDIATE_DIR + str(row[0]) + "_" + str(row[1]) + ".txt", sep=" ", header=None)
        exp_id = row[0]
        user_id = row[1]
        start = row[3]
        end = row[4]
        label = row[2]
        sub_dataframe = window_and_extract_features(data, exp_id, user_id, label, start, end)
        new_data = new_data.append(sub_dataframe)
    print("feature matrix shape before PCA: " + str(new_data.shape))  # shape of raw features
    new_data.to_csv(FEATURE_FILE, sep=" ", index=False, header=None)


# *****************************************#
#         4.feature normalization          #
# *****************************************#
def normalize_data():
    features = pd.read_csv(FEATURE_FILE, sep=" ", header=None)
    # print(features.head(5))
    for column in features.columns[3:]:
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
    without_label = features.iloc[:, 3:]
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
    new_data = pd.concat([features.iloc[:, :3], df], axis=1, sort=False, ignore_index=True)  # add labels
    datafile = 'PCA=' + str(pca_dims) + REDUCED_FEATURE_FILE
    new_data.to_csv(datafile, sep=" ", index=False, header=None)


# *****************************************#
#           6.sampling rate change         #
# *****************************************#
def sampling_rate(pca_dims, rate):
    moderated = str(int(50 // rate))
    datafile = 'PCA=' + str(pca_dims) + REDUCED_FEATURE_FILE
    outdir = 'sampled/' + moderated + 'Hz/'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    outfile = outdir + 'PCA=' + str(pca_dims) + REDUCED_FEATURE_FILE
    with open(datafile, 'r') as f, open(outfile, 'w', newline='') as f_out:
        # number would determine sampling size
        count = 0
        count2 = 0
        reader = csv.reader(f)
        writer = csv.writer(f_out)
        for row in reader:
            count += 1
            count2 += 1
            if count % 5 != 0:
                if count2 % 2 == 0:
                    writer.writerow(row)
                # container.append(line)

    f.close()
    f_out.close()


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
    seed(2020)
    # filter_data()
    # catenate_data()
    # extract_features()
    # normalize_data()
    # pca(50)
    # pca(40)
    # pca(30)
    # pca(20)
    # pca(10)
    # pca(15)
    # pca(8)
    # pca(7)

    s_rate = [10, 5, 2.5, 2, 1.25, 1]
    p = [30, 12, 15, 18, 21, 24, 27] #[20, 30, 40, 50, 60]

    for i in p:
        # pca(i)
        sampling_rate(i, 2.5)
        # for j in s_rate:
        #     sampling_rate(i,j)

    # sampling_rate(i, j)


if __name__ == '__main__':
    main()
