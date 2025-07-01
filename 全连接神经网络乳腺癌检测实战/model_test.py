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
y_test_one = to_categorical(y_test,2)

# 特征进行归一化(否则梯度比较低，训练时间比较久)
sc = MinMaxScaler(feature_range=(0,1))
x_test = sc.fit_transform(x_test)

# 导入模型
model = load_model('model.h5')

# 利用训练好的模型进行测试
predict = model.predict(x_test)
print(predict)

y_pred = np.argmax(predict,axis=1)
print(y_pred)

# 进行转化
result_name = []

for i in range(len(y_pred)):
    if y_pred[i] == 1:
        result_name.append('恶性')
    else:
        result_name.append('良性')

print(result_name)

# 打印模型的精度和召回
report = classification_report(y_test,y_pred,labels=[0,1],target_names=["良性","恶性"])
