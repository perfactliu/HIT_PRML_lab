import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# 创建高维特征
def polynomial_features(x0, x1, degree):
    # 计算 x0 和 x1 的不同幂次
    features = [np.ones_like(x0)]  # 添加偏置项列
    for d in range(1, degree + 1):
        features.append(x0 ** d)
        features.append(x1 ** d)

    # 通过按列拼接生成最终的特征矩阵
    return np.column_stack(features)  # eg:[1,x0,x1,x0^2,x1^2,x0^3,x1^3,x0^4,x1^4]


# 1. 读取数据
degree = 6  # todo:定义高维维数
L2 = True  # todo:正则化
data = pd.read_csv('logistic_data2.csv')
x0, x1, y = data['x0'].values, data['x1'].values, data['y'].values

# 生成高维特征矩阵
X = polynomial_features(x0, x1, degree)

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
    # todo:决定是否正则化

    # cost = (y.T @ np.log(h) + (1 - y).T @ np.log(1 - h)) / m  # 对数损失函数

    cost = (y.T @ np.log(h) + (1 - y).T @ np.log(1 - h)) / m + (Lambda / (2 * m)) * np.sum(theta[1:] ** 2)

    return cost


def gradient_descent(X, y, theta, alpha, num_iters):
    cost_history = []

    for _ in range(num_iters):
        h = sigmoid(X @ theta)

        if L2:
            gradient = np.zeros_like(theta)
            gradient[0] = 1 / m * (X[:, 0].T @ (h - y))  # 偏置项的梯度，不进行正则化
            gradient[1:] = 1 / m * (X[:, 1:].T @ (h - y)) + (Lambda / m) * theta[1:]  # 其他参数的梯度，加入正则化项
        else:
            gradient = X.T @ (h - y) / m
        theta -= alpha * gradient
        # cost_history.append(compute_cost(X, y, theta))

    return theta, cost_history


# 绘制决策边界的函数
def plot_decision_boundary(x0, x1, theta, degree):
    # 创建一个二维网格
    u = np.linspace(min(x0) - 1, max(x0) + 1, 100)
    v = np.linspace(min(x1) - 1, max(x1) + 1, 100)
    U, V = np.meshgrid(u, v)  # [[100]*100]

    # 使用多项式特征构造新的特征矩阵
    UV_poly = polynomial_features(U.ravel(), V.ravel(), degree)
    Z = UV_poly @ theta  # 10000*1
    Z = Z.reshape(U.shape)

    # 绘制决策边界
    plt.contour(U, V, Z, levels=[0], linewidths=2, colors='g')


# 5. 梯度下降优化
alpha = 0.01  # 学习率
Lambda = 3
num_iters = 100000  # 迭代次数

t = time.time()
theta, cost_history = gradient_descent(X, y, theta, alpha, num_iters)
print(f'cost:{time.time() - t:.4f}s')

# 6. 绘制决策边界
plt.figure(figsize=(8, 6))

# 绘制数据点
plt.scatter(x0[y.flatten() == 0], x1[y.flatten() == 0], color='red', label='Class 0')
plt.scatter(x0[y.flatten() == 1], x1[y.flatten() == 1], color='blue', label='Class 1')

# 绘制决策边界
plot_decision_boundary(x0, x1, theta, degree)

plt.xlabel('x0')
plt.ylabel('x1')
plt.legend()
plt.title(f'Logistic Regression Decision Boundary (degree = {degree} , L2 = {L2})')
# print(cost_history)  # todo:显示损失下降过程
plt.show()


