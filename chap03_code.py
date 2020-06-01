import sys

import numpy as np
import pandas as pd

sys.modules['tensorflow'] = None

x_train, y_train, x_test = load_fashionmnist()

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# logの中身が0になるのを防ぐ
def np_log(x):
    return np.log(np.clip(a=x, a_min=1e-10, a_max=1e+10))


def softmax(x):
  x -= x.max(axis=1,keepdims=True)
  x_exp = np.exp(x)
  return x_exp / np.sum(x_exp, axis = 1,keepdims=True)

# weights
W = np.zeros(shape = (784,10)).astype('float32') 
b = np.zeros(shape=(10,)).astype('float32')

# 学習データと検証データに分割
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1)

def train(x, t, eps=0.08):
  global W,b

  batch_size = x.shape[0]

  y = softmax(np.matmul(x, W) + b)

  cost = (-t * np_log(y) - (1-t) * np_log(1-y)).mean()
  delta = y-t

  #パラメータ更新
  dW = np.matmul(x.T, delta) / batch_size
  db = np.matmul(np.ones(shape=(batch_size,)) , delta) / batch_size
  W -= eps * dW
  b -= eps * db

  return cost

def valid(x, t):
  y = softmax(np.matmul(x, W) + b)
  cost = (-t * np_log(y)).sum(axis=1).mean()

  return cost,y
max_epoch = 30

for epoch in range(max_epoch):
    # オンライン学習
  for x,y in zip(x_train, y_train):
    train_cost = train(x[None,:], y[None,:])
  valid_cost, y_pred = valid(x_valid, y_valid)
  
  print('EPOCH: {},  Valid Cost: {:.3f}, Valid Accuracy: {:.3f}'.format(
        epoch + 1,
        valid_cost,
        accuracy_score(y_valid.argmax(axis=1), y_pred.argmax(axis=1))
    ))