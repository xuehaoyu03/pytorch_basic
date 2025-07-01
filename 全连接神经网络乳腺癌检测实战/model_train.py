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

# 提取数据中的特征和标签
X = dataset.iloc[:,: -1]
Y = dataset['target']

# 划分测试集和训练集
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

# 将数据标签转化为one_hot向量格式(让神经网络认识标签)
y_train_one = to_categorical(y_train,2)
y_test_one = to_categorical(y_test,2)

# 特征进行归一化(否则梯度比较低，训练时间比较久)
sc = MinMaxScaler(feature_range=(0,1))
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

# 利用keras框架搭建神经网络学习模型
model = keras.Sequential()
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(2,activation='softmax'))

# 对神经网络进行编译（损失函数，评价指标）
model.compiled(loss='categorical_crossentyopy',optimizer='SGD',metrics=['accuracy'])

# 训练模型(verbose：打印输出台，validation_data：测试集当成训练集)
history = model.fit(x_train,y_train_one,epochs=100,batch_size=16,verbose = 2,validation_data=(x_test,y_test_one))
model.save('model.h5')

# 绘制训练集和验证集的loss值对比
plt.plot(history.history['loss'],label='train')
plt.plot(history.history['val_loss'],label='val')
plt.title("全连接神经网络loss值图")
plt.legend()
plt.show()


# 绘制训练集和验证集准确率的对比图
plt.plot(history.history['accuracy'],label='train')
plt.plot(history.history['val_accuracy'],label='val')
plt.title("全连接神经网络accuracy值图")
plt.legend()
plt.show()