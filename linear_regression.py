import numpy as np
from sklearn.model_selection import train_test_split

def train_test_split_fun(x, y, test_size=0.1):
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
  return x_train, x_test, y_train, y_test

def initialization(feature_size):
  w = np.random.rand(feature_size, 1)
  bias = np.random.rand(1, 1)
  return w, bias

def forward_pass(x, w, bias):
  return np.dot(x, w) + bias

def compute_loss(y, y_pred):
  return np.sum((y - y_pred) ** 2) / len(y)

def grad_w(x, y_pred, y_train):
  return (2 * np.dot(x.T, (y_pred - y_train))) / y_pred.shape[0]

def grad_b(x, y_pred, y_train):
  return 2 * np.sum(y_pred - y_train) / y_pred.shape[0]

def backward_pass(w_old, b_old, dw, db, lr):
  w_new = w_old - lr * dw
  b_new = b_old - lr * db
  return w_new, b_new

def train(num_epochs = 10000):
  w, bias = initialization(feature_size=3)
  for epoch in range(num_epochs):
    y_pred = forward_pass(x_train, w, bias)
    loss = compute_loss(y_train, y_pred)
    dw = grad_w(x_train, y_pred, y_train)
    db = grad_b(x_train, y_pred, y_train)
    w, bias = backward_pass(w, bias, dw, db, lr=0.0001)
    if epoch % 10 == 0:
      print(f"Epoch: {epoch}, loss: {loss}")
  return w, bias

def evaluate_model(w, bias):
  y_test_pred = forward_pass(x_test, w, bias)
  loss = compute_loss(y_test, y_test_pred)
  print(f"Test loss: {loss}")


if __name__ == "__main__":
    feature_size = 3
    num_epochs = 1000
    x = np.random.rand(1000, feature_size)
    y = np.random.rand(1000, 1)
    x_train, x_test, y_train, y_test = train_test_split_fun(x, y, test_size=0.1)
    w, bias = train(num_epochs)
    evaluate_model(w, bias)
    
