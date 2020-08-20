import pickle
import sklearn
import time
import csv

import split_acti
from HAPT.data_processing_HAPT import plot_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

PLOT_ALL = False
DO_PCA = True
# DO_LP_FILTERING = True
# DO_HP_FILTERING = False  # remove gravity
DO_CROSS_VALIDATION = False #True
NUMBER_OF_K_FOLD_CROSS_VALIDATION = 10
NUMBER_OF_CLASSIFIERS_IN_RANDOMFOREST = 120



# *****************************************#
#          KNN          #
# *****************************************#
def KNN(X_train, X_test, y_train, y_test, moderated, pca_dims):
    print(X_train.shape)
    """ # use this to choose K
    k = []
    score = []
    for i in range(0, 50):
        if i%2 != 0:
            print(i)
            model = KNeighborsClassifier(n_neighbors=i)
            scores = sklearn.model_selection.cross_val_score(model, X_train, y_train, cv=KFold(n_splits=NUMBER_OF_K_FOLD_CROSS_VALIDATION, shuffle=True, random_state=0), scoring='accuracy')
            k.append(i)
            score.append(scores.mean())

    plt.plot(k, score)
    plt.xlabel('Number of Neighbors K')
    plt.ylabel('Accuracy')
    plt.show()
    return
    """

    print("################# KNN #################")
    model = KNeighborsClassifier(n_neighbors=9)
    if DO_CROSS_VALIDATION:
        scores = sklearn.model_selection.cross_val_score(model, X_train, y_train, cv=KFold(n_splits=NUMBER_OF_K_FOLD_CROSS_VALIDATION, shuffle=True), scoring='accuracy')
        print("KNN cross-validation Accuracy: %0.2f" % scores.mean())
    print()
    model.fit(X_train, y_train)
    pickle.dump(model, open('model/' + moderated + 'Hz/' + 'pca=' + str(pca_dims) + 'KNN.hdf5', 'wb'))
    del model



# *****************************************#
#          Naive Bayes          #
# *****************************************#
def NaiveBayes(X_train, X_test, y_train, y_test, moderated, pca_dims):
    print(X_train.shape)
    model = GaussianNB()
    print("################# NaiveBayes #################")
    if DO_CROSS_VALIDATION:
        scores = sklearn.model_selection.cross_val_score(model, X_train, y_train, cv=KFold(n_splits=NUMBER_OF_K_FOLD_CROSS_VALIDATION, shuffle=True), scoring='accuracy')
        print("NaiveBayes cross-validation Accuracy: %0.2f" % scores.mean())
    print()
    model.fit(X_train, y_train)
    pickle.dump(model, open('model/' + moderated + 'Hz/' + 'pca=' + str(pca_dims) + 'NB.hdf5', 'wb'))
    del model



# *****************************************#
#          SVM          #
# *****************************************#
def SVM(X_train, X_test, y_train, y_test, moderated, pca_dims):
    print(X_train.shape)
    model = sklearn.svm.SVC(kernel='linear', C=1000, decision_function_shape='ovo')

    """ # gridsearch to find hyperparameters
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                     'C': [0.001, 0.01, 0.1, 1, 10, 25, 50, 100, 1000]},
                    {'kernel': ['sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                     'C': [0.001, 0.01, 0.1, 1, 10, 25, 50, 100, 1000]},
                    {'kernel': ['linear'], 'C': [0.001, 0.01, 0.1, 1, 10, 25, 50, 100, 1000]}
                   ]
    clf = sklearn.model_selection.GridSearchCV(model1, tuned_parameters, cv=KFold(n_splits=10, shuffle=True, random_state=0))
    clf.fit(without_labels, labels)
    print(clf.best_params_)

    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    return
    """
    print("################# SVM #################")
    if DO_CROSS_VALIDATION:
        scores = cross_val_score(model, X_train, y_train, cv=KFold(n_splits=NUMBER_OF_K_FOLD_CROSS_VALIDATION, shuffle=True), scoring='accuracy')
        print("SVM cross-validation Accuracy: %0.2f" % scores.mean())
    print()
    model.fit(X_train, y_train)
    pickle.dump(model, open('model/' + moderated + 'Hz/' + 'pca=' + str(pca_dims) + 'SVM.hdf5', 'wb'))
    del model



# *****************************************#
#       RandomForest    #
# *****************************************#
def RandomForest(X_train, X_test, y_train, y_test, moderated, pca_dims):
    print(X_train.shape)
    model = RandomForestClassifier(n_estimators=NUMBER_OF_CLASSIFIERS_IN_RANDOMFOREST, criterion='entropy', max_features='log2')
    """ # gridsearch to find hyperparameters
    tuned_parameters = {'n_estimators': [50, 80, 100, 120, 150, 200],
                  'max_features': ['auto', 'sqrt', 'log2'],
                  'criterion' :['gini', 'entropy']
                  }
    clf = sklearn.model_selection.GridSearchCV(model, tuned_parameters, cv=KFold(n_splits=10, shuffle=True, random_state=0), scoring='accuracy')
    clf.fit(X_train, y_train)
    print("-------------------------------------")
    print(clf.best_params_)
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    return
    """
    print("################# RandomForest #################")
    if DO_CROSS_VALIDATION:
        scores = cross_val_score(model, X_train, y_train, cv=KFold(n_splits=NUMBER_OF_K_FOLD_CROSS_VALIDATION, shuffle=True), scoring='accuracy')
        print("RandomForest cross-validation Accuracy: %0.2f" % scores.mean())
    print()
    model.fit(X_train, y_train)
    pickle.dump(model, open('model/' + moderated + 'Hz/' + 'pca=' + str(pca_dims) + 'RF.hdf5', 'wb'))
    del model



def main(pca_dims, rate):

    moderated = str(int(20 // rate))
    X_train, X_test, y_train, y_test = split_acti.main(pca_dims, rate, 0.3)

    if pca_dims == 0:
        pca_dims = 30


    start_time = time.time()
    KNN(X_train, X_test, y_train, y_test, moderated, pca_dims)
    time_knn = (time.time() - start_time)

    NaiveBayes(X_train, X_test, y_train, y_test, moderated, pca_dims)
    time_nb = (time.time() - time_knn)

    SVM(X_train, X_test, y_train, y_test, moderated, pca_dims)
    time_svm = (time.time() - time_nb)

    RandomForest(X_train, X_test, y_train, y_test, moderated, pca_dims)
    time_rf = (time.time() - time_svm)

    print(pca_dims, moderated)
    print("done!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    with open('training_others_results.txt', 'a', newline='') as f_out:
        writer = csv.writer(f_out)
        # writer.writerow(['Sampling rate', 'PCA dims', 'Time'])
        writer.writerow([moderated, pca_dims, time_svm, time_nb, time_knn, time_rf])
    f_out.close()


if __name__ == '__main__':

    with open('training_others_results.txt', 'w', newline='') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(
            ['Sampling rate', 'PCA dims', 'SVM', 'NB', 'KNN', 'RF'])
    f_out.close()
    x = [0, 7, 8, 10, 15, 20]
    rate = 1
    for pca_dims in x:
        main(pca_dims, rate)
