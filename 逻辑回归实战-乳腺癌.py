import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split # 将数据集分割为训练集和测试集
from sklearn.preprocessing import MinMaxScaler # 对特征进行归一化处理
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report # 精确率、召回率、F1 分数

# 读取数据
dataset = pd.read_csv("1.csv")

# 提取X特征
X = dataset.iloc[:, :, -1]
# 提取Y
Y = dataset['target']

# 划分数据集和测试集 train80  test20
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)

# 进行数据的归一化
sc= MinMaxScaler(feature_range=(0,1))
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

# 逻辑回归模型
lr = LogisticRegression()
lr.fit(x_train,y_train)

# 打印模型的参数
print("W:" ,lr.coef_)
print("B:",lr.intercept_)

# 进行测试推理
pre_result = lr.predict(x_test)
print(pre_result)

# 打印预测结果的概率
pre_result_preoba = lr.predict_proba(x_test)

# 获取恶性肿瘤的概率
pre_list = pre_result_preoba[:,1]

thresholds = 0.3

result = []
result_name = []

for i in range(len(pre_list)):
    if pre_list[i] > thresholds:
        result.append(1)
        result_name.append('恶性')
    else:
        result.append(0)
        result_name.append('良性')


# 精确率、召回率、F1 分数
report = classification_report(y_test,result,labels=[0,1] ,target_names=['ok','not'])
