import sys,os
sys.path.append(os.pardir)
from other.getMNIST import load_mnist
import numpy as np
import pickle

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c) #オーバーフロー対策 ゼロつく1 p.69参照
    sum_exp_a = np.sum(exp_a)
    y = exp_a/sum_exp_a
    return y

def get_data():
    (x_train,t_train),(x_test,t_test) = \
        load_mnist(normalize=True,flatten=True,one_hot_label=False)
    return x_test,t_test

def init_network():
    with open("sample_weight.pkl",'rb') as f:
        network = pickle.load(f)

        return network

def predict(newtwork,x):
    W1,W2,W3 = newtwork['W1'],newtwork['W2'],newtwork['W3']
    b1,b2,b3 = newtwork['b1'],newtwork['b2'],newtwork['b3']

    a1 = np.dot(x,W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3) + b3
    y = softmax(a3)

    return y

x, t = get_data()
print(x.shape)
print(t.shape)
batch_size = 100

network = init_network()
accuracy_cnt = 0

for i in range(0,len(x),batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network,x_batch)
    p = np.argmax(y_batch,axis=1)
    accuracy_cnt += np.sum(p==t[i:i+batch_size])

print("ACCURACY:" + str(float(accuracy_cnt) / len(x)))