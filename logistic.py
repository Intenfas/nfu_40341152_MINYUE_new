# -*- coding: utf-8 -*-
# !/usr/bin/python
#author wepe

from numpy import *
from os import listdir


def loadData(direction):
    # 抓資料夾裡的所有檔案名稱
    trainfileList = listdir(direction)
    # 計算出trainfileList長度
    m = len(trainfileList)
    # 新增一個data矩陣 zeros為Numpy內的函數，用途為創建一個元素均為0的組數 大小為m*1024大小的訓練矩陣
    dataArray = zeros((m, 1024))
    # 新增一個label矩陣 大小為m*1的類別向量
    labelArray = zeros((m, 1))
    for i in range(m):
        # 每個txt文件形成的特徵向量
        returnArray = zeros((1, 1024))
        filename = trainfileList[i]
        # 打開文件，filename為文件名稱
        fr = open('%s/%s' % (direction, filename))
        for j in range(32):
            # 每次讀一行資料
            lineStr = fr.readline()
            for k in range(32):
                returnArray[0, 32 * j + k] = int(lineStr[k])
        # 儲存特徵向量
        dataArray[i, :] = returnArray
        filename0 = filename.split('.')[0]
        label = filename0.split('_')[0]
        # 儲存類別
        labelArray[i] = int(label)  # 存储类别
    return dataArray, labelArray


def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))


# alpha:步長，maxCycles:迭代次數，可以调整
# 梯度下降函數
def gradAscent(dataArray, labelArray, alpha, maxCycles):
    # size=m*n mat為numpy方法，把輸入解釋為矩陣
    dataMat = mat(dataArray)
    # size=m*1
    labelMat = mat(labelArray)
    # shape為numpy方法，計算矩陣長度
    m, n = shape(dataMat)
    # ones為numpy方法，返回一個固定形狀和類型的新組數
    weigh = ones((n, 1))
    for i in range(maxCycles):
        h = sigmoid(dataMat * weigh)
        error = labelMat - h  # size:m*1
        weigh = weigh + alpha * dataMat.transpose() * error
    return weigh


# 分類函數，根據參數weigh對測試樣本進行預測，同時計算錯誤率
def classfy(testdir, weigh):
    dataArray, labelArray = loadData(testdir)
    dataMat = mat(dataArray)
    labelMat = mat(labelArray)
    h = sigmoid(dataMat * weigh)  # size:m*1
    m = len(h)
    error = 0.0
    for i in range(m):
        if int(h[i]) > 0.5:
            print int(labelMat[i]), 'is classfied as: 1'
            if int(labelMat[i]) != 1:
                error += 1
                print 'error'
        else:
            print int(labelMat[i]), 'is classfied as: 0'
            if int(labelMat[i]) != 0:
                error += 1
                print 'error'
    print 'error rate is:', '%.4f' % (error / m)


# 整合上面函數
def digitRecognition(trainDir, testDir, alpha=0.07, maxCycles=10):
    data, label = loadData(trainDir)
    weigh = gradAscent(data, label, alpha, maxCycles)
    classfy(testDir, weigh)


digitRecognition('train', 'test', 0.01, 50)








