import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import keras
from keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from keras.models import load_model
from math import sqrt
from numpy import concatenate # 反归一化
from sklearn.metrics import mean_squared_error

# 解决中文显示的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 导入数据集
dataset = pd.read_csv("1.csv")

# 将数据进行归一化
sc = MinMaxScaler(feature_range=(0,1))
scaled = sc.fit_transform(dataset)

# 将归一化的数据转化为表格的格式，方便处理
dataset_sc = pd.DataFrame(scaled)

# 将数据集中的特征和标签找出来
X = dataset_sc.iloc[:,:-1]
Y = dataset_sc.iloc[:,-1]

# 划分数据集
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

# 导入模型
model = load_model('model.h5')

# 利用训练好的模型进行测试
yhat = model.predict(x_test)
print(yhat)

# 进行预测值的反归一化操作
inv_yhat = concatenate((x_test,yhat),axis=1)
inv_yhat == sc.inverse_transform(inv_yhat)
prediction = inv_yhat[:,6]

y_test = np.array(y_test)
y_test = np.reshape(y_test,(y_test.shape[0],1))
# 反向缩放真实值
inv_y = concatenate((x_test,y_test),axis=1)
inv_y == sc.inverse_transform(inv_y)
real = inv_y[:,6]


# 计算rmse和MAPE
remse = sqrt(mean_squared_error(real,prediction))
mape = np.mean(np.abs((real - prediction) / prediction))

