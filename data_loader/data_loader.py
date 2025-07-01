from torch.utils.data import Dataset  # 从 PyTorch 的数据工具模块导入 Dataset 类，用于自定义数据集
import os  # 导入 Python 内置的操作系统交互库，用于文件路径、目录操作等
import pandas as pd  # 导入 Pandas 库，简称 pd，用于数据处理（如读取 CSV 文件、DataFrame 操作）
import numpy as np  # 导入 NumPy 库，简称 np，用于数值计算（数组操作、数学运算等）
import torch  # 导入 PyTorch 库，用于深度学习相关功能（张量操作、模型构建等）

class iris_dataloader(Dataset):  # 定义继承自 Dataset 的类，用于构建鸢尾花数据集加载器

    def __init__(self, data_path):  # 类的构造函数，初始化对象时调用，参数 data_path 为数据集文件路径
        self.data_path = data_path  # 将传入的数据集路径保存为对象的属性，方便后续使用

        # 断言检查：判断数据集路径对应的文件是否存在，不存在则抛出错误提示
        assert os.path.exists(self.data_path), "dataset does not exits"

        # 用 Pandas 读取 CSV 文件，指定列名为 [0,1,2,3,4]，将文件内容加载为 DataFrame
        df = pd.read_csv(self.data_path, names=[0, 1, 2, 3, 4])

        # 定义字典，用于将鸢尾花品种名称映射为数字标签
        d = {"Iris-setosa": "0", "Iris-versicolor": "1", "Iris-virginica": "2"}
        # 对 DataFrame 的第 4 列（品种列）应用映射，将名称转为数字标签
        df[4] = df[4].map(d)

        # 提取 DataFrame 中前 4 列作为特征数据（.iloc[:, 0:4] 表示取所有行、第 0 到 3 列 ）
        data = df.iloc[:, 0:4]
        # 提取 DataFrame 中第 4 列作为标签数据（.iloc[:, 4:] 表示取所有行、第 4 列 ）
        label = df.iloc[:, 4]

        # 对特征数据进行 Z 标准化：(数据 - 均值) / 标准差，让特征具有相似的尺度，利于模型训练
        data = (data - np.mean(data)) / np.std(data)

        # 将处理后的特征数据（NumPy 数组）转为 PyTorch 张量，数据类型设为 float32
        self.data = torch.from_numpy(np.array(data, dtype='float32'))
        # 将处理后的标签数据（NumPy 数组）转为 PyTorch 张量，数据类型设为 int64
        self.label = torch.from_numpy(np.array(label, dtype='int64'))

        # 计算并保存数据集样本数量（标签的长度就是样本数 ）
        self.data_num = len(label)
        # 打印当前数据集的大小（样本数量 ）
        print("当前数据集的大小：", self.data_num)