# 定义数据集

# 定义特征
x_data = [1,2,3]

# 定义数据标签
y_data = [2,4,6]

# 初始化参数W
w = 4

# 定义线性回归的模型
def forword(x):
    return x * w

# 定义损失函数
def cost(xs,ys):
    costValue = 0
    for x,y in zip(xs,ys):
        y_pred = forword(x)
        costValue = (y_pred - y) ** 2
    return costValue / len(xs)

# 梯度下降计算公式
def graient(xs,ys):
    grad = 0
    for x,y in zip(xs,ys):
        grad += 2 * x * (x * w - y)
    return grad / len(xs)

for epoch in range(100):
    cost_val = cost(x_data,y_data)

    grad_val = graient(x_data,y_data)

    w = w - 0.01 * grad_val

    print("训练次数",epoch,"w = ",w,"loss = ",cost_val)


