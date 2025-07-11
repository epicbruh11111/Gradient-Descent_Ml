from GD import gradient_descent


import numpy as np # I'll use numpy because its easier and mostly used

x:list = np.random.randn(10,1) # input data
# print(x) # development test
# print(f"Number of samples: {len(x)}") # test
y:list =  2*x + np.random.rand() 

# parameters for gradient descent
w:float = 0.0 # initial weight
b:float = 0.0 # initial bias\

gradient_descent(x, y, w, b, learning_rate=0.01) # run gradient descent