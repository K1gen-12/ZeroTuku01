import numpy as np

def step_func2(x):
    y = x > 0
    return y.astype(np.int)

x = np.array([-1.0,1.0,2.0])

print("step_func({}):{}".format(0,step_func2(x)))
