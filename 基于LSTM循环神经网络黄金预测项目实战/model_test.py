import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import load_model

# 解决中文显示的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据集
dataset = pd.read_csv("1.csv",index_col=[0])

# 设置训练集的长度
training_len = 1256 - 200

# 获取测试集数据
test_set = dataset.iloc[training_len:,[0]]

# 对数据集进行归一化处理
sc = MinMaxScaler(feature_range=(0,1))

test_set = sc.fit_transform(test_set)


# 设置测试集特征和测试集标签
x_test = []
y_test = []

# 利用for循环进行测试集特征和标签的制作，提取数据中连续5天作为特征
for i in range(5,len(test_set)):
    x_test.append(test_set[i-5:i,0])
    y_test.append(test_set[i,0])

# 将训练集用list转为array格式
x_test,y_test = np.array(x_test),np.array(y_test)

# 循环神经网络送去训练格式：[样本数，时间步，特征个数]
x_test = np.reshape(x_test,(x_test.shape[0],5,1))

# 导入模型
model = load_model('model.h5')

predicted = model.predict(x_test)

# 进行反归一化
prediction = sc.inverse_transform(predicted)

# 对测试集的标签进行反归一化
real = sc.inverse_transform(test_set[5:])

remse = sqrt(mean_squared_error(real,prediction))
mape = np.mean(np.abs((real - prediction) / prediction))

plt.plot(real, label='真实值')
plt.plot(prediction, label='预测值')
plt.title('基于LSTM神经网络的黄金价格预测')
plt.legend()
plt.show()




