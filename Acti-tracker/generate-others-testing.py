import pickle
import csv
import time

import split_acti
from HAPT.data_processing_HAPT import plot_report
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import train_test_split
# from sklearn import metrics
# from sklearn.model_selection import cross_val_score, KFold
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
# from sklearn.metrics.classification import confusion_matrix, f1_score
from sklearn.metrics import accuracy_score, classification_report

PLOT_ALL = False
DO_PCA = True
# DO_LP_FILTERING = True
# DO_HP_FILTERING = False  # remove gravity
DO_CROSS_VALIDATION = False  # True
NUMBER_OF_K_FOLD_CROSS_VALIDATION = 10
NUMBER_OF_CLASSIFIERS_IN_RANDOMFOREST = 120

MY_LABELS = ['Walking', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Downstairs']


# *****************************************#
#          KNN          #
# *****************************************#
def KNN(X_train, X_test, y_train, y_test, moderated, pca_dims):
    print(X_train.shape)
    model = pickle.load(open('model/' + moderated + 'Hz/' + 'pca=' + str(pca_dims) + 'KNN.hdf5', 'rb'))
    test_predict = model.predict(X_test)
    if PLOT_ALL:
        plot_report(y_test, test_predict, "KNN")
    # print("report for KNN: ")
    # report = classification_report(y_test, test_predict, digits=4, target_names=MY_LABELS)
    # print(report)
    acc = accuracy_score(y_test, test_predict)
    print("KNN overall accuracy: " + str(acc))
    # print(confusion_matrix(y_test, test_predict))

    return acc


# *****************************************#
#          Naive Bayes          #
# *****************************************#
def NaiveBayes(X_train, X_test, y_train, y_test, moderated, pca_dims):
    print(X_train.shape)
    model = pickle.load(open('model/' + moderated + 'Hz/' + 'pca=' + str(pca_dims) + 'NB.hdf5', 'rb'))
    test_predict = model.predict(X_test)
    if PLOT_ALL:
        plot_report(y_test, test_predict, "Naive Bayes")
    # print("report for NaiveBayes: ")
    # print(classification_report(y_test, test_predict, digits=4, target_names=MY_LABELS))
    acc = accuracy_score(y_test, test_predict)
    print("NaiveBayes overall accuracy: " + str(acc))

    return acc


# *****************************************#
#          SVM          #
# *****************************************#
def SVM(X_train, X_test, y_train, y_test, moderated, pca_dims):
    print(X_train.shape)
    model = pickle.load(open('model/' + moderated + 'Hz/' + 'pca=' + str(pca_dims) + 'SVM.hdf5', 'rb'))
    test_predict = model.predict(X_test)
    if PLOT_ALL:
        plot_report(y_test, test_predict, "SVM")
    # print("report for SVM: ")
    # print(classification_report(y_test, test_predict, digits=4, target_names=MY_LABELS))
    acc = accuracy_score(y_test, test_predict)
    print("SVM overall accuracy: " + str(acc))
    # print(sklearn.metrics.confusion_matrix(y_test, test_predict))

    return acc


# *****************************************#
#       RandomForest    #
# *****************************************#
def RandomForest(X_train, X_test, y_train, y_test, moderated, pca_dims):
    print(X_train.shape)
    model = pickle.load(open('model/' + moderated + 'Hz/' + 'pca=' + str(pca_dims) + 'RF.hdf5', 'rb'))
    test_predict = model.predict(X_test)
    if PLOT_ALL:
        plot_report(y_test, test_predict, "Random Forest")
    # print("report for RandomForest: ")
    # print(classification_report(y_test, test_predict, digits=4, target_names=MY_LABELS))
    acc = accuracy_score(y_test, test_predict)
    print("RandomForest overall accuracy: " + str(acc))


    """ # https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_],
             axis=0)
    indices = np.argsort(importances)[::-1]
    # Print the feature ranking
    print("Feature ranking:")
    for f in range(X_train.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X_train.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
    plt.xticks(range(X_train.shape[1]), indices)
    plt.xlim([-1, X_train.shape[1]])
    plt.show()
    """

    return acc


def main(pca_dims, rate):
    moderated = str(int(20 // rate))
    X_train, X_test, y_train, y_test = split_acti.main(pca_dims, rate, 0.3, 'acc')

    start_time = time.time()
    acc_knn = KNN(X_train, X_test, y_train, y_test, moderated, pca_dims)
    time_knn = (time.time() - start_time)

    acc_nb = NaiveBayes(X_train, X_test, y_train, y_test, moderated, pca_dims)
    time_nb = (time.time() - time_knn)

    acc_svm = SVM(X_train, X_test, y_train, y_test, moderated, pca_dims)
    time_svm = (time.time() - time_nb)

    acc_rf = RandomForest(X_train, X_test, y_train, y_test, moderated, pca_dims)
    time_rf = (time.time() - time_svm)

    print(pca_dims, moderated)
    print("done!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    with open('testing_others_results.txt', 'a', newline='') as f_out:
        writer = csv.writer(f_out)
        # writer.writerow(['Sampling rate', 'PCA dims', 'Time'])
        writer.writerow([moderated, pca_dims, acc_svm, time_svm, acc_nb, time_nb,
                         acc_knn, time_knn, acc_rf, time_rf])
    f_out.close()


if __name__ == '__main__':

    with open('testing_others_results.txt', 'w', newline='') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(
            ['Sampling rate', 'PCA dims', 'SVM-acc','SVM-time', 'NB-acc','NB-time', 'KNN-acc', 'KNN-time',
             'RF-acc', 'RF-time'])
    f_out.close()
    x = [0, 7, 8, 10, 15, 20]
    rate = 1

    for pca_dims in x:
        main(pca_dims, rate)
