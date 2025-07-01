
5. SoftMax函数
$\mathbf{y}_i = \frac{\mathrm{e}^{\mathbf{z}_i}}{\sum_j \mathrm{e}^{\mathbf{z}_j}} \implies \mathbf{y}_i' = \mathbf{y}_i (1 - \mathbf{y}_i)$

损失函数
1. 回归任务
$均方误差的损失函数：


J(x) = \frac{1}{2m} \sum_{i=1}^{m} (f(x_i) - y_i)^2$
2. 分类任务
$交叉熵损失函数：


J(x) = -\frac{1}{m} \sum_{i=1}^{m} \left( y_i \log\sigma(w^{\mathrm{T}} x_i + b) + (1 - y_i) \log(1 - \sigma(w^{\mathrm{T}} x_i + b)) \right)$
数据类型
深度学习常用的数据类型有DataFrame(Pandas)、 Array(Numpy)、tensor、list、map
import pandas as pd
import numpy as np

# DataFrame(Pandas) iloc切片
df = pd.read_csv('fileName')
a = np.array([1, 2, 3])

# DataFrame -> Array
data = np.array(data)

# Array -> torch
data = torch.from_numpy(data)
data = torch.tensor(data)

# list
image_paths = ['data/001.jpg', 'data/002.jpg']  

# map
model_config = {
    'hidden_dim': 512,
    'num_layers': 6,
    'dropout': 0.1
}

1.全连接网络（FCNN）
 每个神经元都与前一层和后一层的所有神经元相连接，形成一个密集的连接结构。

分类任务：loss + accuracy 回归任务： loss
2.卷积神经网络（CNN）
● 卷积层：用来提取图像的底层特征
● 池化层：防止过拟合，将数据维度减小
● 全连接层：汇总卷积层和池化层得到的图像的底层特征和信息
特征图大小
$计算公式：


OH = \frac{H + 2P - FH}{S} + 1$
$OW = \frac{W + 2P - FW}{S} + 1$
H/W 是输入特征图的高 / 宽；P 是填充（Padding）；FH/FW 是卷积核的高 / 宽；S 是步长
1.卷积运算
单通道

多通道（通达数是由卷积核的数量决定的）

2.池化运算
1. 最大池化运算

2. 平均池化运算

3.循环神经网络（RNN）
与传统的前馈神经网络不同，RNN在处理每个输入时都会保留一个隐藏状态，该隐藏状态会被传递到下一个时间步，以便模型能够记忆之前的信息。权重共享


$数学表达式：


O_t = g(V \cdot h_t)$
$h_t = f(U \cdot X_t + W \cdot h_{t-1})$
基于时间反向传播（BPTT，Backpropagation Through Time ） 
$梯度计算公式：


\frac{\partial L_t}{\partial w} = \sum_{i=0}^{i=t} \frac{\partial L_t}{\partial o_t} \frac{\partial o_t}{\partial h_t} \frac{\partial h_t}{\partial h_i} \frac{\partial h_i}{\partial w}$
$\frac{\partial h_t}{\partial h_i}这项计算的过程：


\frac{\partial h_t}{\partial h_i} = \frac{\partial h_t}{\partial h_{t-1}} \cdot \frac{\partial h_{t-1}}{\partial h_{t-2}} \cdots \frac{\partial h_{i+1}}{\partial h_i} = \prod_{k=i}^{k=t-1} \frac{\partial h_{k+1}}{\partial h_k}$
$同时：


\frac{\partial h_k}{\partial h_{k-1}} = f' \cdot w$
$f' = f(1 - f) \in \left(0, \frac{1}{4}\right) 连乘导致梯度消失和爆炸$
RNN梯度消失和梯度爆炸情况
4.长短时记忆网络（LSTM）
传统的 RNN 在处理长序列数据时面临着严重的梯度消失问题，这使得网络难以学习到长距离的依赖关系。LSTM 作为一种特殊的 RNN 架构应运而生，有效地解决了这一难题。

遗忘门

输入门

细胞状态进行更新：根据遗忘门Ct-1和输入门(当前时刻)的结果，对细胞状态Ct进行更新

输出门
