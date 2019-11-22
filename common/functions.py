# coding: utf-8
import numpy as np


# 恒等関数
def identity_function(x):
    return x


# ステップ関数 (x<=0ならばx=0, x>0ならばx=1)
# 条件式(x>0)に対してのTrue,Falseをint型の1,0で返す
def step_function(x):
    return np.array(x > 0, dtype=np.int)


# シグモイド関数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))    


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)
    

# RELU関数 (x<=0ならばx=0, x>0ならばx=x)
def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    grad = np.zeros(x)
    grad[x>=0] = 1
    return grad
    

# softmax関数 ( exp(a_k)/{Σ(1→n)exp(a_i)} )
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)   # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x))


# 2乗和誤差 ( 1/2Σ_k(y_k-t_k)^2 )
def sum_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)


# 交差エントロピー誤差 ( -Σ_k(t_k*log(y_k)) )
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)
