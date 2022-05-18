import numpy as np
import matplotlib.pylab as plt

def RELU(x):
    return np.maximum(0,x)

x = np.arange(-6.0,6.0,0.1)
y = RELU(x)

plt.plot(x,y,label="relu")
plt.ylim(-1,5.5)
plt.show()