# import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import split_HAPT
import pandas as pd
import os
import csv
import time
from numpy.random import seed
import pickle
from sklearn.metrics import accuracy_score


# the path of HAPT_Data_Set dir
ROOT = "HAPT Data Set/"

# config path and intermediate files
DATA_SET_DIR = ROOT + "RawData/"
PROCESSED_DATA_DIR = ROOT + "Processed/"
LABLE_FILE = DATA_SET_DIR + "labels.txt"
INTERMEDIATE_DIR = PROCESSED_DATA_DIR + "intermediate/"


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

def main():
    seed(2020)
    # filter_data()
    catenate_data()
    extract_features()

    # rate = 1
    # pca_dims = 30
    # compress_rate = 0.3
    #
    # seed(2020)
    # X_train, X_test, y_train, y_test = split_HAPT.main(pca_dims, rate, 0.3, '')
    # moderated = str(int(50 // rate))
    # if pca_dims == 0:
    #     pca_dims = 60
    #
    # y_train = y_train - 1
    # y_test = y_test - 1
    #
    # start_time = time.time()
    #
    # model = pickle.load(open('model/' + moderated + 'Hz/pruned/' + 'pickled_' + 'decompressed_pca=' + str(
    #     pca_dims) + '_compress_rate=' + str(compress_rate) + '.hdf5', 'rb'))
    #
    # # print(moderated)
    # # print(pca_dims,compress_rate)
    #
    # print("SEEEEEEEEEEEEEEEEEEE")
    #
    # pred_test = model.predict(np.expand_dims(X_test, axis=2), batch_size=32)
    # print("------ TEST ACCURACY ------")
    # testing_acc = (accuracy_score(y_test, np.argmax(pred_test, axis=1)))
    # print(testing_acc)
    # # print((confusion_matrix(y_test, np.argmax(pred_test, axis=1))))
    # t = (time.time() - start_time)
    #
    # pred_train = model.predict(np.expand_dims(X_train, axis=2), batch_size=32)
    # print("------ TRAIN ACCURACY ------")
    # training_acc = (accuracy_score(y_train, np.argmax(pred_train, axis=1)))
    # print(training_acc)
    # # print((confusion_matrix(y_train, np.argmax(pred_train, axis=1))))
    #
    # i=0
    #
    # while i < len(X_test):
    #     start_time = time.time()
    #     lewl = model.predict_on_batch(np.expand_dims(X_test[i:i+100], axis=2))
    #     bas = time.time() - start_time
    #     training_acc = (accuracy_score(y_test[i:i+100], np.argmax(lewl, axis=1)))
    #     print(training_acc, bas)
    #     i+=101

    # with open('testing_results_pickle_test.txt', 'a', newline='') as f_out:
    #     writer = csv.writer(f_out)
    #     # writer.writerow(['Sampling rate', 'PCA dims', 'Time'])
    #     writer.writerow([moderated, pca_dims, compress_rate, training_acc, testing_acc, t])
    # f_out.close()


if __name__ == '__main__':
    main()