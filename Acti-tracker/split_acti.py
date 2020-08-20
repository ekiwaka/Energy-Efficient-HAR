import numpy as np
from numpy.random import seed
from sklearn.model_selection import train_test_split

# pca_dims = 50
# TESTING_DATASET_SIZE = 0.3

# the path of HAPT_Data_Set dir
ROOT = "Acti-tracker Data Set/"

# config path and intermediate files
# REDUCED_FEATURE_FILE = str(pca_dims) + "reduced_features.txt"
REDUCED_FEATURE_FILE = "reduced_features" #.txt"
NORMALIZED_FEATURE_FILE = "normalized_features" #.txt"
LABEL_FILE = "labels.txt"
MY_LABELS = ['Walking', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Downstairs']


def main(pca_dims, rate, test_data_ratio):
    moderated = str(int(20 // rate))
    print("start...............................")

    datafile = 'PCA=' + str(pca_dims) + REDUCED_FEATURE_FILE + '.txt'

    with open(datafile) as f:
        container = f.readlines()

    result = []
    for line in container:
        tmp1 = line.strip()
        tmp2 = tmp1.replace('  ', ' ')  # removes inconsistent blank spaces
        tmp_ary = list(map(float, tmp2.split(' ')))
        result.append(tmp_ary)
    without_labels = np.array(result)


    label_file = 'labels.txt'
    with open(label_file) as f:
        container = f.readlines()

    result = []
    for line in container:
        num_str = line.strip()
        result.append(int(num_str))
    labels= np.array(result)

    seed(2020)
    # split data
    X_train, X_test, y_train, y_test = train_test_split(without_labels, labels, test_size=test_data_ratio)
    # plot_label_distribution(y_train)

    return X_train, X_test, y_train, y_test
#
if __name__ == '__main__':
    X_train, X_test, y_train, y_test = main(30,1,0.3)
