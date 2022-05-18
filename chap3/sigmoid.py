import numpy as np
import matplotlib.pylab as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

def step_func(x):
    return np.array(x>0,dtype=np.int)

x = np.arange(-5.0,5.0,0.1)
y1 = sigmoid(x)
y2 = step_func(x)
plt.plot(x,y1,label="sigmoid")
plt.plot(x,y2,linestyle=":",label="step_func")
plt.ylim(-0.1,1.1)
plt.show()


""" a = np.array([-1.0,1.0,2.0])

print("sigmoid:{}".format(sigmoid(a))) """