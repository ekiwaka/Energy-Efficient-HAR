import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from HAPT import split_HAPT
import csv
import time
from numpy.random import seed
from keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score

# from sklearn.metrics import confusion_matrix


if __name__ == '__main__':
    # x = [0, 15, 18, 21, 24, 27]
    x = [0, 20, 30, 40, 50]
    y = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    s_rate = [10, 5, 2.5, 2, 1.25, 1]
    MY_LABELS = ['WALKING', 'UPSTAIRS', 'DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING', 'STAND_TO_SIT', 'SIT_TO_STAND',
                 'SIT_TO_LIE', 'LIE_TO_SIT', 'STAND_TO_LIE', 'LIE_TO_STAND']

    with open('testing_results.txt', 'w', newline='') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(
            ['Sampling rate', 'PCA dims', 'Compression rate', 'Training accuracy', 'Testing accuracy', 'Time']) #, 'Report'])
    f_out.close()

    for rate in s_rate:
        for pca_dims in x:
            for compress_rate in y:
                seed(2020)
                X_train, X_test, y_train, y_test = split_HAPT.main(pca_dims, rate, test_data_ratio=0.3)
                moderated = str(int(50 // rate))
                print(moderated, pca_dims, compress_rate)
                if pca_dims == 0:
                    pca_dims = 60
                # moderated= str(50)

                # sampling_rate = [0, 20, 30, 40, 50]
                # for pca_dims in sampling_rate:
                #         seed(2020)
                #         X_train, X_test, y_train, y_test = split.main(pca_dims)
                #
                #         if pca_dims == 0:
                #             pca_dims = 60

                start_time = time.time()

                model_path = 'model/' + moderated + 'Hz/pruned/' + 'decompressed_pca=' + str(
                    pca_dims) + '_compress_rate=' + str(compress_rate) + '.hdf5'
                model = load_model(model_path)

                y_train = y_train - 1
                y_test = y_test - 1
                # print(moderated)
                # print(pca_dims,compress_rate)

                pred_test = model.predict(np.expand_dims(X_test, axis=2), batch_size=32)
                print("------ TEST ACCURACY ------")
                testing_acc = (accuracy_score(y_test, np.argmax(pred_test, axis=1)))
                print(testing_acc)
                report = classification_report(y_test, np.argmax(pred_test, axis=1), target_names=MY_LABELS)
                print(report)
                # print((confusion_matrix(y_test, np.argmax(pred_test, axis=1))))
                t = (time.time() - start_time)

                pred_train = model.predict(np.expand_dims(X_train, axis=2), batch_size=32)
                print("------ TRAIN ACCURACY ------")
                training_acc = (accuracy_score(y_train, np.argmax(pred_train, axis=1)))
                print(training_acc)

                # print((confusion_matrix(y_train, np.argmax(pred_train, axis=1))))

                with open('testing_results.txt', 'a', newline='') as f_out:
                    writer = csv.writer(f_out)
                    # writer.writerow(['Sampling rate', 'PCA dims', 'Time'])
                    writer.writerow([moderated, pca_dims, compress_rate, training_acc, testing_acc, t]) #, report])
                f_out.close()
