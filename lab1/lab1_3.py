import numpy as np
import pandas as pd

cost_history = []


def data_loader(path):
    data = pd.read_csv(path)
    sepal_length, sepal_width, petal_length, petal_width, species = (data['sepal length (cm) '].values,
                                                                     data['sepal width (cm) '].values,
                                                                     data['petal length (cm) '].values,
                                                                     data['petal width (cm) '].values,
                                                                     data['label'].values)
    X = np.c_[sepal_length, sepal_width, petal_length, petal_width]
    y = species.reshape(-1, 1)
    return X, y


def softmax(z):
    e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e_z / np.sum(e_z, axis=1, keepdims=True)


def compute_cost(Y, Y_hat):
    m = Y.shape[0]
    cost = -np.sum(Y * np.log(Y_hat + 1e-9))/m
    cost_history.append(cost)


def one_hot_encoding(y):
    m = y.shape[0]
    Y = np.zeros((m, 3))
    for i in range(m):
        Y[i, int(y[i])] = 1
    return Y


def train(X, y, alpha, iteration):
    features = 4
    species = 3
    Y = one_hot_encoding(y)
    W = np.random.randn(features, species) * 0.01
    b = np.zeros((1, species))
    m = X.shape[0]
    for i in range(iteration):
        Z = X@W + b
        Y_hat = softmax(Z)
        compute_cost(Y, Y_hat)
        dZ = Y_hat - Y
        dW = X.T@dZ / m
        db = np.sum(dZ, axis=0, keepdims=True) / m
        W -= alpha * dW
        b -= alpha * db
    return W, b


def test(X, y, W, b):
    Z = X@W + b
    Y_hat = softmax(Z)
    y_pred = np.argmax(Y_hat, axis=1)
    sum = len(y_pred)
    correct = 0
    for i in range(sum):
        if y_pred[i] == int(y[i]):
            correct += 1
    accuracy = correct/sum*100
    return accuracy


train_data_path = 'iris_train.csv'
test_data_path = 'iris_val.csv'
X_train, y_train = data_loader(train_data_path)
X_test, y_test = data_loader(test_data_path)
alpha = 0.01
iteration = 100000
W, b = train(X_train, y_train, alpha, iteration)
accuracy = test(X_test, y_test, W, b)
# accuracy = test(X_train, y_train, W, b)

# print(cost_history)
print(f'accuracy is {accuracy} %')

