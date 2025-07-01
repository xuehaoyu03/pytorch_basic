## 神经网络篇
> 模型的训练：导入数据 -> 数据进行归一化处理 -> 特征和标签 -> 划分训练集和验证集 -> 搭建神经网络 -> 进行训练并导出模型 -> 对训练集和验证集评价指标进行图形化
>
> 模型的验证： 导入数据 -> 数据进行归一化处理 -> 特征和标签 -> 划分训练集和验证集 -> 导入模型 -> 进行训练 -> 格式化预测数据 -> （回归任务：反归一化） -> 计算评价指标rmse和mape
>

#### 激活函数
1. Sigmoid 函数

![image](https://cdn.nlark.com/yuque/__latex/1622c082073681d20e5e72c4904914a7.svg)

适用于简单分类任务，缺点：反向传播训练有梯度消失的问题、非0对称

![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1750658328633-006a02e6-f0ca-40f1-97e2-143ab5b3357f.png)

![](C:\Users\xhyu10\AppData\Roaming\Typora\typora-user-images\image-20250618100620592.png)

2. Tanh函数

![image](https://cdn.nlark.com/yuque/__latex/9218a1f9f63350bd92bdf0d6184d7b59.svg)

收敛快，解决了非0对称，缺点：反向传播梯度消失

![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1750658361592-b6046a38-70c8-4074-b9fe-96b6e6c8322d.png)

![](C:\Users\xhyu10\AppData\Roaming\Typora\typora-user-images\image-20250618100914573.png)

3. ReLU函数

![image](https://cdn.nlark.com/yuque/__latex/1ae0cde4ed28679751d232b3b5c5d3ad.svg)

解决了梯度消失问题，计算简单，缺点：训练可能出现神经元死亡

![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1750658384442-5d5aa18c-993e-4be6-887e-af0765616802.png)

![](C:\Users\xhyu10\AppData\Roaming\Typora\typora-user-images\image-20250618101141043.png)

4. Leaky ReLU函数

![image](https://cdn.nlark.com/yuque/__latex/3e8bf45a2fa87da2605a7b024f496db1.svg)

![](C:\Users\xhyu10\AppData\Roaming\Typora\typora-user-images\image-20250618101336163.png)

解决了ReLU的神经元死亡问题

![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1750658417524-ddf6b049-4266-4849-b590-986aa1ac2f91.png)

5. SoftMax函数

![image](https://cdn.nlark.com/yuque/__latex/83a7ec1eb28cfc5a124007e5c8691ca7.svg)

![](C:\Users\xhyu10\AppData\Roaming\Typora\typora-user-images\image-20250618101604996.png)![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1750658446070-071fe85b-9680-463d-b73d-8c60134d832e.png)

#### 损失函数
1. 回归任务

![image](https://cdn.nlark.com/yuque/__latex/87dafc19aad2f8df3e3866053432b554.svg)

2. 分类任务

![image](https://cdn.nlark.com/yuque/__latex/2b42ce1dae4b0500aacdf34cb4af312e.svg)

#### 数据类型
深度学习常用的数据类型有**DataFrame(Pandas)**、 **Array(Numpy)**、**tensor**、**list**、**map**

```python
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
```



### 1.全连接网络（FCNN）
 每个神经元都与前一层和后一层的所有神经元相连接，形成一个密集的连接结构。

![](C:\Users\xhyu10\AppData\Roaming\Typora\typora-user-images\image-20250618095709297.png)![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1750658465700-46fdfbfa-823c-407f-8f53-f135536d689c.png)

分类任务：loss + accuracy 回归任务： loss

### 2.卷积神经网络（CNN）![](C:\Users\xhyu10\AppData\Roaming\Typora\typora-user-images\image-20250618105712212.png)![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1750658497005-f80677c3-afe4-4fb6-a1e6-4c677bce354e.png)
+ 卷积层：用来提取图像的底层特征
+ 池化层：防止过拟合，将数据维度减小
+ 全连接层：汇总卷积层和池化层得到的图像的底层特征和信息

**特征图大小**

![image](https://cdn.nlark.com/yuque/__latex/f775642bf66fb6b7bd45d6c52b41f0e4.svg)

![image](https://cdn.nlark.com/yuque/__latex/53d8e14d7bcf488ed75f920779ec29be.svg)

`H`/`W` 是输入特征图的高 / 宽；`P` 是填充（Padding）；`FH`/`FW` 是卷积核的高 / 宽；`S` 是步长

**1.卷积运算**

**单通道**

![](C:\Users\xhyu10\AppData\Roaming\Typora\typora-user-images\image-20250618110018982.png)![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1750658526992-f53fe3fe-d9f7-4332-ad18-4acf3942490c.png)

**多通道（通达数是由卷积核的数量决定的）**

![](C:\Users\xhyu10\AppData\Roaming\Typora\typora-user-images\image-20250618105937768.png)![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1750658569685-e8151938-995c-4ead-8523-a8753cdf6724.png)

**2.池化运算**

1. 最大池化运算

![](C:\Users\xhyu10\AppData\Roaming\Typora\typora-user-images\image-20250618111737197.png)![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1750658621381-e82e187c-eee0-4ed7-9f9c-930d061eae93.png)

2. 平均池化运算

![](C:\Users\xhyu10\AppData\Roaming\Typora\typora-user-images\image-20250618111810820.png)![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1750658632876-4adfcf5d-e994-465d-b3a8-ea97d00cbd7e.png)

### 3.循环神经网络（RNN）
与传统的前馈神经网络不同，RNN在处理每个输入时都会保留一个隐藏状态，该隐藏状态会被传递到下一个时间步，以便模型能够**记忆**之前的信息。**权重共享**

![](C:\Users\xhyu10\AppData\Roaming\Typora\typora-user-images\image-20250618135436957.png)![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1750658650927-f0eabf80-bb87-423a-b2db-710f7ba16eba.png)![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1750658666426-053acf67-c69b-426d-8365-6375bc130a14.png)

![](C:\Users\xhyu10\AppData\Roaming\Typora\typora-user-images\image-20250618135532849.png)

![image](https://cdn.nlark.com/yuque/__latex/b814443584d61218309a86db43fb0ef1.svg)

![image](https://cdn.nlark.com/yuque/__latex/a29784b6f024161585e3a186dd9d51f1.svg)

**基于时间反向传播（BPTT，Backpropagation Through Time ）** 

![image](https://cdn.nlark.com/yuque/__latex/d3cc48bc7b6cc55289f072f949c3f065.svg)

![image](https://cdn.nlark.com/yuque/__latex/268d87fe63cb5e26b066a6b637925cc6.svg)

![image](https://cdn.nlark.com/yuque/__latex/2ccc2d009b36f045ecbcf4b408ea342f.svg)

![image](https://cdn.nlark.com/yuque/__latex/dd71f6708dd8af9acac2db631fb915c2.svg)

**RNN梯度消失和梯度爆炸情况**

### 4.长短时记忆网络（LSTM）
传统的 RNN 在处理长序列数据时面临着严重的**梯度消失**问题，这使得网络难以学习到长距离的依赖关系。LSTM 作为一种特殊的 RNN 架构应运而生，有效地解决了这一难题。

![](C:\Users\xhyu10\AppData\Roaming\Typora\typora-user-images\image-20250618141311607.png)![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1750658707407-5ac9be8d-6b8b-4af7-a130-5955d930be3a.png)

#### 遗忘门
![](C:\Users\xhyu10\AppData\Roaming\Typora\typora-user-images\image-20250618141540975.png)![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1750658720050-43055781-7dee-4be9-be5c-df454ce5454f.png)

#### 输入门
![](C:\Users\xhyu10\AppData\Roaming\Typora\typora-user-images\image-20250618141818765.png)![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1750658731544-d4344aa8-e8c3-469b-ad58-3afe7a184c90.png)

#### 细胞状态进行更新：根据遗忘门Ct-1和输入门(当前时刻)的结果，对细胞状态Ct进行更新
![](C:\Users\xhyu10\AppData\Roaming\Typora\typora-user-images\image-20250618142631825.png)![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1750658747549-0fa84ab8-4731-4578-9b6e-3170731c76c4.png)

#### 输出门
![](C:\Users\xhyu10\AppData\Roaming\Typora\typora-user-images\image-20250618142725318.png)![](https://cdn.nlark.com/yuque/0/2025/png/29704292/1750658763303-1a62c67c-c978-419c-bd51-216e3be862c6.png)




