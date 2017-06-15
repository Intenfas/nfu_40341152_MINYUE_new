#coding=utf-8
import sys
import os
import time
from sklearn import metrics
import numpy as np
import cPickle as pickle

from sklearn import datasets
iris = datasets.load_iris()
digits = datasets.load_digits()

reload(sys)
sys.setdefaultencoding('utf8')

# Multinomial Naive Bayes Classifier
def naive_bayes_classifier(train_x, train_y):
    from sklearn.naive_bayes import MultinomialNB  ##樸素貝葉斯分類器的多項式模型
    model = MultinomialNB(alpha=0.01)              ##加法（拉普拉斯/ Lidstone）平滑參數（0表示無平滑）
    model.fit(train_x, train_y)                    ##將x,y傳入
    return model
# KNN Classifier
def knn_classifier(train_x, train_y):
    from sklearn.neighbors import KNeighborsClassifier      ##KNN分類演算法
    model = KNeighborsClassifier()
    model.fit(train_x, train_y)
    return model


# Logistic Regression Classifier
def logistic_regression_classifier(train_x, train_y):           ##邏輯回歸分類演算法
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l2')
    model.fit(train_x, train_y)
    return model

# SVM Classifier
def svm_classifier(train_x, train_y):                       ##支持向量機分類法
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)            ##指定算法中要使用的內核類型。是否啟用概率估計。
    model.fit(train_x, train_y)
    return model


#def read_data(data_file):
    #import gzip
    #f = gzip.open(data_file, "rb")                 ##開啟檔案
    #train, val, test = pickle.load(f)
    #f.close()                                      ##關閉檔案
    #f=iris
    #train_x = train[0]
    #train_y = train[1]
    #test_x = test[0]
    #test_y = test[1]
    #return train_x, train_y, test_x, test_y


if __name__ == '__main__':
    data_file = "mnist.pkl.gz"
    thresh = 0.5
    model_save_file = None
    model_save = {}

    test_classifiers = ['NB', 'KNN', 'LR', 'SVM']
    classifiers = {'NB': naive_bayes_classifier,
                   'KNN': knn_classifier,
                   'LR': logistic_regression_classifier,
                   'SVM': svm_classifier,
                   }
    print 'reading training and testing data...'
   # train_x, train_y, test_x, test_y = read_data(data_file)         ##利用變數存取檔案資料
   # num_train, num_feat = train_x.shape                             ##num_train計算列數,num_feat計算行數
   # num_test, num_feat = test_x.shape                               ##num_test計算列數,num_feat計算行數
   # is_binary_class = (len(np.unique(train_y)) == 2)

    train_x = digits.data[0:1000]
    train_y = digits.target[0:1000]
    test_x = digits.data[1000:1797]
    test_y = digits.target[1000:1797]
    num_train,num_feat = train_x.shape
    num_test,num_feat = test_x.shape

    print '******************** Data Info *********************'
    print '#training data: %d, #testing_data: %d, dimension: %d' % (num_train, num_test, num_feat)
    for classifier in test_classifiers:
        print '******************* %s ********************' % classifier
        start_time = time.time()                                            ##抓取起始時間
        model = classifiers[classifier](train_x, train_y)
        print 'training took %fs!' % (time.time() - start_time)              ##計算訓練時間
        predict = model.predict(test_x)

        #if is_binary_class:                                                 ##將70行的結果作判斷
        #    precision = metrics.precision_score(test_y, predict)            ##計算精度
         #   recall = metrics.recall_score(test_y, predict)
         #   print 'precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall)
        accuracy = metrics.accuracy_score(test_y, predict)
        print 'accuracy: %.2f%%' % (100 * accuracy)