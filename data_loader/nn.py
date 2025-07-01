import os  # 导入Python操作系统交互库，用于文件/目录操作（如路径判断、创建等）
import sys  # 导入Python系统相关功能库，可访问命令行参数、退出机制等

from torch.utils.data import DataLoader  # 从PyTorch数据工具模块，导入数据加载器类，用于按批次加载数据集
from tqdm import tqdm  # 从进度条库tqdm，导入tqdm类，用于训练/迭代时显示进度（代码里原拼写tqdm可能是笔误，正确是tqdm）

import torch  # 导入PyTorch核心库，提供张量运算、深度学习基础功能
import torch.nn as nn  # 从PyTorch导入神经网络模块，简称为nn，用于构建模型层、损失函数等
import torch.optim as optim  # 从PyTorch导入优化器模块，简称为optim，用于模型训练优化（如Adam、SGD）

from data_loader import iris_dataloader  # 从自定义模块data_loader，导入iris_dataloader类，用于加载鸢尾花数据集

class NN(nn.Module):  # 定义继承自 nn.Module 的神经网络类，nn.Module 是 PyTorch 构建模型的基类
    def __init__(self, in_dim, hidden_dim1, hidden_dim2, out_dim):  # 构造函数，初始化网络层
        super().__init__()  # 调用父类构造函数，确保 nn.Module 的初始化逻辑被执行
        # 定义第一层线性层：输入维度 in_dim，输出维度 hidden_dim1
        self.layer1 = nn.Linear(in_dim, hidden_dim1)
        # 定义第二层线性层：输入维度 hidden_dim1，输出维度 hidden_dim2
        self.layer2 = nn.Linear(hidden_dim1, hidden_dim2)
        # 定义第三层线性层：输入维度 hidden_dim2，输出维度 out_dim
        self.layer3 = nn.Linear(hidden_dim2, out_dim)

    def forward(self, x):  # 前向传播函数，定义数据在网络中的流动逻辑
        x = self.layer1(x)  # 数据经过第一层线性层
        x = self.layer2(x)  # 数据经过第二层线性层
        x = self.layer3(x)  # 数据经过第三层线性层
        return x  # 返回最终输出

# 根据 GPU 是否可用，选择计算设备，可用则选 "cuda:0"（第一块 GPU ），否则选 "cpu"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 实例化自定义数据集，传入数据集文件路径（这里路径是 "./pytorch2.0-nn/iris_data.txt" ）
custom_dataset = iris_dataloader("./pytorch2.0-nn/iris_data.txt")

# 划分数据集：训练集占 70%、验证集占 20%、测试集占 10%（通过计算样本数量实现 ）
train_size = int(len(custom_dataset) * 0.7)
val_size = int(len(custom_dataset) * 0.2)
test_size = len(custom_dataset) - train_size - val_size

# 随机划分数据集为训练集、验证集、测试集，传入数据集和各集样本数量列表
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    custom_dataset, [train_size, val_size, test_size]
)

# 构建训练集数据加载器：批次大小 16，打乱数据（shuffle=True ）
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# 构建验证集数据加载器：批次大小 1，不打乱（方便按顺序评估 ）
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
# 构建测试集数据加载器：批次大小 1，不打乱
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

print(
    "训练集的大小", len(train_dataset),
    "验证集的大小", len(val_dataset),
    "测试集的大小", len(test_dataset)
)

def infer(model, dataset, device):
    # 将模型设置为评估模式（影响 dropout、batchnorm 等层行为）
    model.eval()
    acc_num = 0
    # 禁用梯度计算（节省内存，加速推理）
    with torch.no_grad():
        # 遍历数据集中的每个样本
        for data in dataset:
            # 拆分数据为特征(datas)和标签(label)
            datas, label = data
            # 将特征张量转移到指定计算设备（CPU/GPU）
            outputs = model(datas.to(device))
            # 取输出中维度 1 上的最大值索引，作为预测类别
            predict_y = torch.max(outputs, dim=1)[1]
            # 统计预测正确的样本数量：对比预测值与标签，求和后转为 Python 数值
            acc_num += torch.eq(predict_y, label.to(device)).sum().item()
    # 计算整体准确率：正确数 / 总样本数
    acc = acc_num / len(dataset)
    return acc

def main(lr=0.005, epochs=20):
    # 初始化模型：输入维度4，隐藏层维度12、6，输出维度3 → 转移到device（CPU/GPU）
    model = NN(4, 12, 6, 3).to(device)
    # 定义损失函数：交叉熵损失（适合分类任务）
    loss_f = nn.CrossEntropyLoss()

    # 构建优化器参数组：筛选需要计算梯度的参数
    pg = [p for p in model.parameters() if p.requires_grad]
    # 初始化Adam优化器：传入参数组、学习率
    optimizer = optim.Adam(pg, lr=lr)

    # 权重文件存储路径配置
    # 拼接当前工作目录与保存路径
    save_path = os.path.join(os.getcwd(), "results/weights")
    # 若路径不存在则创建
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    # 开始训练流程
    for epoch in range(epochs):
        # 切换模型为训练模式（启用 dropout、batchnorm 等训练行为）
        model.train()
        # 初始化正确样本计数（张量，转移到device）
        acc_num = torch.zeros(1).to(device)
        sample_num = 0  # 初始化总样本计数

        # 构建训练进度条：关联train_loader，输出到标准输出，设置进度条宽度
        train_bar = tqdm(train_loader, file=sys.stdout, ncols=100)
        for datas in train_bar:
            # 拆分数据为特征(data)和标签(label)
            data, label = datas
            # 压缩标签维度（处理可能的多余维度，如从 [batch,1] 转 [batch]）
            label = label.squeeze(-1)
            # 累加当前批次样本数（data.shape[0] 是批次大小）
            sample_num += data.shape[0]

            # 训练三步：反向传播前需清空梯度
            optimizer.zero_grad()
            # 特征转移到device，模型前向推理
            outputs = model(data.to(device))
            # 取输出维度1上的最大值索引（预测类别）
            pred_class = torch.max(outputs, dim=1)[1]
            # 统计当前批次正确样本数（对比预测与标签）
            acc_num += torch.eq(pred_class, label.to(device)).sum()

            # 计算损失：输出与标签（转device）送入损失函数
            loss = loss_f(outputs, label.to(device))
            # 反向传播计算梯度
            loss.backward()
            # 优化器更新参数
            optimizer.step()

            # 计算训练集准确率
            train_acc = acc_num / sample_num
            # 动态更新进度条描述：显示epoch、损失、准确率
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f} train_acc:{:.3f}".format(
                epoch + 1, epochs, loss, train_acc
            )

        # 调用 infer 函数，在验证集 val_loader 上推理，计算验证集准确率
        val_acc = infer(model, val_loader, device)

        # 打印训练过程信息：当前epoch、总epoch、损失、训练集准确率、验证集准确率
        print(
            "train epoch[{}/{}] loss:{:.3f} train_acc:{:.3f} val_acc:{:.3f}".format(
                epoch + 1, epochs, loss, train_acc, val_acc
            )
        )

        # 保存模型权重：将模型的状态字典（参数）保存到指定路径（save_path 下的 nn.pth 文件）
        torch.save(model.state_dict(), os.path.join(save_path, "nn.pth"))

        # 每次数据集迭代（epoch）后，重置训练集、验证集准确率指标（清零，为下一轮做准备）
        train_acc = 0.
        val_acc = 0.

        # 打印训练完成提示
        print("Fished Training")  # 注：拼写笔误，应为 "Finished Training"

        # 调用 infer 函数，在测试集 test_dataset 上推理，计算测试集准确率
        test_acc = infer(model, test_dataset, device)
        # 打印测试集准确率
        print("test_acc:", test_acc)


if __name__ == "__main__":
    main()