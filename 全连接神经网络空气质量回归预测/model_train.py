import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import keras
from keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report

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

# 利用keras搭建神经网络模型
model = keras.Sequential()
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1))

# 对神经网络进行编译（损失函数，评价指标）
model.compiled(loss='mse',optimizer='SGD')

# 进行模型的训练
history = model.fit(x_train,y_train,epochs=100,batch_size=16,verbose = 2,validation_data=(x_test,y_test))
model.save('model.h5')

# 绘制训练集和验证集的loss值对比 回归模型用mse，评价指标中不推荐用accuracy而是loss
plt.plot(history.history['loss'],label='train')
plt.plot(history.history['val_loss'],label='val')
plt.title("全连接神经网络loss值图")
plt.legend()
plt.show()


