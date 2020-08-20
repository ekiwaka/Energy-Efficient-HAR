import pickle
import argparse
import time

import split_acti
from data_processing_acti import plot_report
from sklearn.metrics import accuracy_score

PLOT_ALL = False


# *****************************************#
#          KNN          #
# *****************************************#
def KNN(X_train, X_test, y_train, y_test, moderated, pca_dims):
    # print(X_train.shape)

    start_time = time.time()
    model = pickle.load(open('model/' + moderated + 'Hz/' + 'pca=' + str(pca_dims) + 'KNN.hdf5', 'rb'))
    load_time = time.time() - start_time

    start_time = time.time()
    test_predict = model.predict(X_test)
    inference_time = time.time() - start_time

    if PLOT_ALL:
        plot_report(y_test, test_predict, "KNN")
    # print("report for KNN: ")
    # report = classification_report(y_test, test_predict, digits=4, target_names=MY_LABELS)
    # print(report)
    acc = accuracy_score(y_test, test_predict)
    print("KNN overall accuracy: " + str(acc))
    # print(confusion_matrix(y_test, test_predict))

    return acc, load_time, inference_time


# *****************************************#
#          Naive Bayes          #
# *****************************************#
def NaiveBayes(X_train, X_test, y_train, y_test, moderated, pca_dims):
    # print(X_train.shape)

    start_time = time.time()
    model = pickle.load(open('model/' + moderated + 'Hz/' + 'pca=' + str(pca_dims) + 'NB.hdf5', 'rb'))
    load_time = time.time() - start_time

    start_time = time.time()
    test_predict = model.predict(X_test)
    inference_time = time.time() - start_time

    if PLOT_ALL:
        plot_report(y_test, test_predict, "Naive Bayes")
    # print("report for NaiveBayes: ")
    # print(classification_report(y_test, test_predict, digits=4, target_names=MY_LABELS))
    acc = accuracy_score(y_test, test_predict)
    print("NaiveBayes overall accuracy: " + str(acc))

    return acc, load_time, inference_time


# *****************************************#
#          SVM          #
# *****************************************#
def SVM(X_train, X_test, y_train, y_test, moderated, pca_dims):
    # print(X_train.shape)

    start_time = time.time()
    model = pickle.load(open('model/' + moderated + 'Hz/' + 'pca=' + str(pca_dims) + 'SVM.hdf5', 'rb'))
    load_time = time.time() - start_time

    start_time = time.time()
    test_predict = model.predict(X_test)
    inference_time = time.time() - start_time

    if PLOT_ALL:
        plot_report(y_test, test_predict, "SVM")
    # print("report for SVM: ")
    # print(classification_report(y_test, test_predict, digits=4, target_names=MY_LABELS))
    acc = accuracy_score(y_test, test_predict)
    print("SVM overall accuracy: " + str(acc))
    # print(sklearn.metrics.confusion_matrix(y_test, test_predict))

    return acc, load_time, inference_time


# *****************************************#
#       RandomForest    #
# *****************************************#
def RandomForest(X_train, X_test, y_train, y_test, moderated, pca_dims):
    # print(X_train.shape)
    start_time = time.time()
    model = pickle.load(open('model/' + moderated + 'Hz/' + 'pca=' + str(pca_dims) + 'RF.hdf5', 'rb'))
    load_time = time.time() - start_time

    start_time = time.time()
    test_predict = model.predict(X_test)
    inference_time = time.time() - start_time

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

    return acc, load_time, inference_time


def main(pca_dims, classifier):
    rate = 1
    moderated = str(int(20 // rate))
    X_train, X_test, y_train, y_test = split_acti.main(pca_dims, rate, 0.3)

    if classifier == 'knn':
        acc, l_time, i_time = KNN(X_train, X_test, y_train, y_test, moderated, pca_dims)

    elif classifier == 'nb':
        acc, l_time, i_time = NaiveBayes(X_train, X_test, y_train, y_test, moderated, pca_dims)

    elif classifier == 'svm':
        acc, l_time, i_time = SVM(X_train, X_test, y_train, y_test, moderated, pca_dims)

    elif classifier == 'rf':
        acc, l_time, i_time = RandomForest(X_train, X_test, y_train, y_test, moderated, pca_dims)

    else:
        print("Not implemented")
        exit(0)

    print("Classifier, Accuracy, Load Time, Inference Time")
    print(classifier, acc, l_time, i_time)





if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Others testing on WISDM')
    parser.add_argument("--pca", default=30, type=int, help="pca dimensions: [7, 8, 10, 15, 20, 30]")
    parser.add_argument("--classifier", default='svm', type=str, help="Other classifiers. "
                                                               "knn, nb, svm, rf")

    args = parser.parse_args()

    main(args.pca, args.classifier)
