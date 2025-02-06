import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




# 1. 读取数据
data = pd.read_csv('logistic_data1.csv')
x0, x1, y = data['x0'].values/100, data['x1'].values/100, data['y'].values

# 将数据转换为矩阵形式
X = np.c_[np.ones(x0.shape[0]), x0, x1]  # 增加一列全为1，用于表示偏置项

y = y.reshape(-1, 1)  # 转换为列向量
m, n = X.shape  # m为样本数，n为特征数

# 2. 初始化模型参数
theta = np.zeros((n, 1))  # 初始化权重为零

# 3. 定义sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 4. 定义损失函数和梯度计算
def compute_cost(X, y, theta):
    h = sigmoid(X @ theta)  # 计算预测值
    cost = (y.T @ np.log(h) + (1 - y).T @ np.log(1 - h)) / m  # 对数损失函数
    return cost

def gradient_descent(X, y, theta, alpha, num_iters):
    cost_history = []

    for _ in range(num_iters):
        h = sigmoid(X @ theta)
        gradient = X.T @ (h - y) / m
        theta -= alpha * gradient
        cost_history.append(compute_cost(X, y, theta))

    return theta, cost_history

# 5. 梯度下降优化


alpha = 0.01  # 学习率
num_iters = 100000  # 迭代次数

theta, cost_history = gradient_descent(X, y, theta, alpha, num_iters)

# 6. 绘制决策边界
plt.figure(figsize=(8, 6))

# 绘制数据点
plt.scatter(x0[y.flatten() == 0], x1[y.flatten() == 0], color='red', label='Class 0')
plt.scatter(x0[y.flatten() == 1], x1[y.flatten() == 1], color='blue', label='Class 1')

# 绘制决策边界
x_values = np.array([min(x0), max(x0)])
y_values = -(theta[0] + theta[1] * x_values) / theta[2]
plt.plot(x_values, y_values, label='Decision Boundary', color='green')

plt.xlabel('x0')
plt.ylabel('x1')
plt.legend()
plt.title('Logistic Regression Decision Boundary')
# print(cost_history)  # todo: 显示损失下降过程
plt.show()

