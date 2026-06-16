import numpy as np
import pandas as pd

def train_nn(X_train, X_test, y_train):

    X_train_np = X_train.toarray()
    X_test_np = X_test.toarray()

    y_train_np = pd.get_dummies(y_train).values

    input_size = X_train_np.shape[1]
    hidden_size = 10
    output_size = y_train_np.shape[1]

    np.random.seed(42)

    W1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros((1, hidden_size))

    W2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros((1, output_size))

    lr = 0.01

    def softmax(x):
        exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

    for i in range(200):

        z1 = np.dot(X_train_np, W1) + b1
        a1 = np.tanh(z1)

        z2 = np.dot(a1, W2) + b2
        a2 = softmax(z2)

        error = a2 - y_train_np

        dW2 = np.dot(a1.T, error)
        db2 = np.sum(error, axis=0, keepdims=True)

        da1 = np.dot(error, W2.T) * (1 - np.tanh(z1) ** 2)

        dW1 = np.dot(X_train_np.T, da1)
        db1 = np.sum(da1, axis=0, keepdims=True)

        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

    z1_test = np.dot(X_test_np, W1) + b1
    a1_test = np.tanh(z1_test)

    z2_test = np.dot(a1_test, W2) + b2
    predictions = np.argmax(softmax(z2_test), axis=1)

    return predictions