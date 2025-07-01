import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense,LSTM
import keras

# 解决中文显示的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据集
dataset = pd.read_csv("1.csv",index_col=[0])

# 设置训练集的长度
training_len = 1256 - 200

# 获取训练集
training_set = dataset.iloc[:training_len,[0]]

# 获取测试集数据
test_set = dataset.iloc[training_len:,[0]]

# 对数据集进行归一化处理
sc = MinMaxScaler(feature_range=(0,1))
train_set_scaled = sc.fit_transform(training_set)
test_set_scaled = sc.fit_transform(test_set)

# 设置训练集特征和训练集标签
x_train = []
y_train = []

# 设置测试集特征和测试集标签
x_test = []
y_test = []

# 利用for循环进行训练集特征和标签的制作，提取数据中连续5天作为特征
for i in range(5,len(train_set_scaled)):
    x_train.append(train_set_scaled[i-5:i,0])
    y_train.append(train_set_scaled[i,0])

# 将训练集用list转为array格式
x_train,y_train = np.array(x_train),np.array(y_train)

# 循环神经网络送去训练格式：[样本数，时间步，特征个数]
x_train = np.reshape(x_train,(x_train.shape[0],5,1))


# 利用for循环进行测试集特征和标签的制作，提取数据中连续5天作为特征
for i in range(5,len(test_set)):
    x_train.append(test_set[i-5:i,0])
    y_train.append(test_set[i,0])

# 将训练集用list转为array格式
x_test,y_test = np.array(x_train),np.array(y_train)

# 循环神经网络送去训练格式：[样本数，时间步，特征个数]
x_test = np.reshape(x_test,(x_test.shape[0],5,1))

# 搭建网络
model = keras.Sequential()
# return_sequences=True隐藏层是否作为输入
model.add(LSTM(80,return_sequences=True,activation='relu'))
model.add(LSTM(100,return_sequences=False,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1))

# 对网络编译
model.compiled(loss='mse',optimizer=keras.optimizers.SGD(0.01))

# 利用神经网络对训练集进行训练
history = model.fit(x_train,y_train,batch_size=32,epochs=100,validation_data = (x_test,y_test))

model.save('mode.h5')

# 绘制训练集和验证集的loss值对比 回归模型用mse，评价指标中不推荐用accuracy而是loss
plt.plot(history.history['loss'],label='train')
plt.plot(history.history['val_loss'],label='val')
plt.title("LSTM 神经网络loss值图")
plt.legend()
plt.show()
