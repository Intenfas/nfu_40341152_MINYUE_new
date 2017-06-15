#coding=utf-8
#thank for CHAN_LIN
import pandas as pd
import time
from sklearn import metrics
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# Multinomial Naive Bayes Classifier
def naive_bayes_classifier(train_x, train_y):
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB(alpha=0.01)
    model.fit(train_x, train_y)
    return model


# KNN Classifier
def knn_classifier(train_x, train_y):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    model.fit(train_x, train_y)
    return model


# Logistic Regression Classifier
def logistic_regression_classifier(train_x, train_y):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l2')
    model.fit(train_x, train_y)
    return model


# Random Forest Classifier
def random_forest_classifier(train_x, train_y):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=8)
    model.fit(train_x, train_y)
    return model


# Decision Tree Classifier
def decision_tree_classifier(train_x, train_y):
    from sklearn import tree
    model = tree.DecisionTreeClassifier()
    model.fit(train_x, train_y)
    return model


# GBDT(Gradient Boosting Decision Tree) Classifier
def gradient_boosting_classifier(train_x, train_y):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=200)
    model.fit(train_x, train_y)
    return model


# SVM Classifier
def svm_classifier(train_x, train_y):
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    model.fit(train_x, train_y)
    return model


# SVM Classifier using cross validation
def svm_cross_validation(train_x, train_y):
    from sklearn.grid_search import GridSearchCV
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    grid_search = GridSearchCV(model, param_grid, n_jobs=1, verbose=1)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in best_parameters.items():
        print para, val
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
    model.fit(train_x, train_y)
    return model


def read_data():
    dataset_train = pd.read_csv('analy.csv', low_memory=False)
    features_train = ['country_txt', 'region_txt', 'city',
                      'attacktype1_txt', 'weaptype1_txt']
    #x為特徵值
    x = dataset_train[features_train]
    #預測目標
    #y為目標值
    y = dataset_train["success"]
    x = x.fillna(0)
    #LabelEncoder`是一个可以用来将标签规范化的工具类，它可以将标签的编码值范围限定在[0,n_classes-1]
    #通过fit_transform函数计算各个词语出现的次数
    x['country_txt'] = LabelEncoder().fit_transform(x['country_txt'])
    x['region_txt'] = LabelEncoder().fit_transform(x['region_txt'])
    x['city'] = LabelEncoder().fit_transform(x['city'])
    x['attacktype1_txt'] = LabelEncoder().fit_transform(x['attacktype1_txt'])
    x['weaptype1_txt'] = LabelEncoder().fit_transform(x['weaptype1_txt'])
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.33) #test_size為測試樣本大小
    return train_x, train_y, test_x, test_y


if __name__ == '__main__':
    #data_file = "mnist.pkl.gz"
    thresh = 0.5
    model_save_file = None
    model_save = {}

    test_classifiers = ['NB', 'KNN', 'LR', 'RF', 'DT','GBDT', 'SVM']
    classifiers = {'NB': naive_bayes_classifier, #單純貝式
                   'KNN': knn_classifier, #臨邊分析
                   'LR': logistic_regression_classifier, #邏輯回歸
                   'RF': random_forest_classifier, #隨機森林
                   'DT': decision_tree_classifier, #決策樹
                   'GBDT': gradient_boosting_classifier, #梯度分類法
                   'SVM': svm_classifier #支持向量機
                   }

    print 'reading training and testing data...'
    train_x, train_y, test_x, test_y = read_data()
    num_train, num_feat = train_x.shape
    num_test, num_feat = test_x.shape
    is_binary_class = (len(np.unique(train_y)) == 2)

    print('******************** Data Info *********************')
    print('#training data: %d, #testing_data: %d, dimension: %d' % (num_train, num_test, num_feat))
    for classifier in test_classifiers:
        print('******************* %s ********************' % classifier)
        #start_time = time.time()  ##抓取起始時間
        #model = classifiers[classifier](train_x, train_y)
        #print('training took %fs!' % (time.time() - start_time))  ##計算訓練時間
        #predict = model.predict(test_x)

        #accuracy = metrics.accuracy_score(test_y, predict)
        #print('accuracy: %.2f%%' % (100 * accuracy))
        #print('total took %fs!' % (time.time() - start_time))


        start_time = time.time()
        model = classifiers[classifier](train_x, train_y)
        print 'training took %fs!' % (time.time() - start_time)
        predict = model.predict(test_x)
        if model_save_file != None:
            model_save[classifier] = model
        if is_binary_class:
            precision = metrics.precision_score(test_y, predict)
            recall = metrics.recall_score(test_y, predict)
            print 'precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall)
        accuracy = metrics.accuracy_score(test_y, predict)
        print 'accuracy: %.2f%%' % (100 * accuracy)