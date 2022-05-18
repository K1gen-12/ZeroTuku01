import numpy as np

def step_func(x):
    if x > 0:
        return 1
    else:
        return 0

print("step_func({}):{}".format(0,step_func(0)))
print("step_func({}):{}".format(1,step_func(1)))
print("step_func({}):{}".format(2,step_func(2)))
print("step_func({}):{}".format(3,step_func(3)))
