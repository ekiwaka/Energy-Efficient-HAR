import pandas as pd
from numpy.random import seed
from sklearn.model_selection import train_test_split

# pca_dims = 50
# TESTING_DATASET_SIZE = 0.3

# the path of HAPT_Data_Set dir
ROOT = "HAPT Data Set/"

# config path and intermediate files
# REDUCED_FEATURE_FILE = str(pca_dims) + "reduced_features.txt"
REDUCED_FEATURE_FILE = "reduced_features" #_acc.txt"
NORMALIZED_FEATURE_FILE = "normalized_features" #_acc.txt"
MY_LABELS = ['WALKING', 'UPSTAIRS', 'DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING', 'STAND_TO_SIT', 'SIT_TO_STAND',
             'SIT_TO_LIE', 'LIE_TO_SIT', 'STAND_TO_LIE', 'LIE_TO_STAND']


def main(pca_dims, rate, test_data_ratio, sensor):
    moderated = str(int(50 // rate))
    print("start...............................")
    labels = None
    without_labels = None
    if sensor == 'acc':
        sensor = '_acc'

    datafile = 'sampled/' + moderated + 'Hz/' + 'PCA=' + str(pca_dims) + REDUCED_FEATURE_FILE + sensor + '.txt'

    data = pd.read_csv(datafile, sep=" ", header=None)
    data = data.iloc[:, 2:]  # remove user id and experiment id
    labels = data.iloc[:, 0]
    without_labels = data.iloc[:, 1:]

    seed(2020)
    # split data
    X_train, X_test, y_train, y_test = train_test_split(without_labels, labels, test_size=test_data_ratio)
    # plot_label_distribution(y_train)
    # print(y_test)
    # print(y_test-1)

    return X_train, X_test, y_train, y_test
#
if __name__ == '__main__':
    X_train, X_test, y_train, y_test = main(30,1,0.3,'')
