import numpy as np
# シグモイド関数
def sigmoid(x):
    return 1/(1+np.exp(-x))

# ステップ関数
def step_function(x):
    return np.array(x>0,dtype=np.int)

# RELU関数
def relu(x):
    return np.maximum(0,x)
# ソフトマックス関数
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c) #オーバーフロー対策 ゼロつく1 p.69参照
    sum_exp_a = np.sum(exp_a)
    y = exp_a/sum_exp_a
    return y

# 2乗和誤差
def sum_squared_error(y,t):
    return 0.5*np.sum((y-t)**2)

# 交差エントロピー誤差
def cross_entropy_error(y,t):
    if y.dim == 1:
        t = t.reshape(1,t.size)
        y = y.reshape(1,t.size)
    
    batch_size = y.shape[0]
    return -np.sum(np.log[np.arrange(batch_size),t] + 1e-7) /batch_size

