#-*- coding:UTF-8 -*-
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation # 匯入交叉驗證函式
import numpy as np # 數值計算函式
import pandas as pd # 讀取表格
import matplotlib.pyplot as plt # 匯入圖表函式

print("************讀取資料*************")

knn_data=pd.DataFrame(pd.read_csv('knn_data.csv'))
print("       前5筆資料")
print(knn_data.head(5))

#Fully Paid數據集的x1
fully_paid_loan=knn_data.loc[(knn_data["loan_status"] == "Fully Paid"),["loan_amt"]]
#Fully Paid數據集的y1
fully_paid_annual=knn_data.loc[(knn_data["loan_status"] == "Fully Paid"),["annual_inc"]]
#Charge Off數據集的x2
charged_off_loan=knn_data.loc[(knn_data["loan_status"] == "Charged Off"),["loan_amt"]]
#Charge Off數據集的y2
charged_off_annual=knn_data.loc[(knn_data["loan_status"] == "Charged Off"),["annual_inc"]]

#設置圖表字體為華文細黑，字號15

plt.rc('font', family='DFKai-SB', size=15)
'''
黑體	SimHei
微軟雅黑	Microsoft YaHei
微軟正黑體	Microsoft JhengHei
新宋體	NSimSun
新細明體	PMingLiU
細明體	MingLiU
標楷體	DFKai-SB
仿宋體	FangSong
楷體	KaiTi
仿宋_GB2312	FangSong_GB2312
楷體_GB2312	KaiTi_GB2312
'''

print("************繪製散點圖*************")

#繪製散點圖，Fully Paid數據集貸款金額x1，用戶年收入y1，設置顏色，標記點樣式和透明度等參數
plt.scatter(fully_paid_loan,fully_paid_annual,color='#9b59b6',marker='^',s=60)
#繪製散點圖，Charge Off數據集貸款金額x2，用戶年收入y2，設置顏色，標記點樣式和透明度等參數
plt.scatter(charged_off_loan,charged_off_annual,color='#3498db',marker='o',s=60)
#添加圖例，顯示位置右上角
plt.legend(['Fully Paid', 'Charged Off'], loc='upper right')
#添加x軸標題
plt.xlabel(u'貸款金額')
#添加y軸標題
plt.ylabel(u'用戶收入')
#添加圖表標題
plt.title(u'貸款金額與用戶收入')
#設置背景網格線顏色，樣式，尺寸和透明度
plt.grid( linestyle='--', linewidth=0.2)
#顯示圖表
plt.show()

print("************設定自變量與因變量*************")

#將貸款金額和用戶收入設為自變量X
X = np.array(knn_data[['loan_amt','annual_inc']])
#將貸款狀態設為因變量Y
Y = np.array(knn_data['loan_status'])
print("自變量和因變量的行數")
print(X.shape,Y.shape)

print("************分割數據*************")

#分割數據
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.4, random_state=0)
print("訓練集數據的行數")
print(X_train.shape,y_train.shape)
print("測試集數據的行數")
print(X_test.shape,y_test.shape)

print("************訓練模型*************")

#將訓練集代入到KNN模型中
print("訓練集帶入模型")
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train,y_train)
print("準確度 : "+str(clf.score(X_test, y_test)))

print("************新數據分類測試*************")

new_data = np.array([[5000,40000]])
print("新數據 : "+str(new_data))
print("分類預測 : "+str(clf.predict(new_data)))
print(clf.classes_)
print(clf.predict_proba(new_data))

