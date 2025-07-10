"""
Gradient Descent for Linear Reggression
y = wx + b
loss = (y-yhat)^2 / n

"""

# init some parameters
import numpy as np # I'll use numpy because its easier and mostly used

x:list = np.random.randn(10,1) # input data
# print(x) # development test
# print(f"Number of samples: {len(x)}") # test
y:list =  2*x + np.random.rand() 

# parameters for gradient descent
w:float = 0.0 # initial weight
b:float = 0.0 # initial bias

# Hyper parameters
learning_rate:float = 0.01 # how much to change the parameters

# Create gradient descent function
def descent(x:list, y:list, w:float, b:float, learning_rate:float):
    dldw:float = 0.0
    dldb:float = 0.0
    n:int = len(x)  # number of samples
    # loss = (y-(wx+b))^2
    for xi,yi in zip(x, y):
        dldw += -2*x*(yi-(w*xi+b)) #partial derivative of loss with respect to w
        dldb += -2*(yi-(w*xi+b)) #partial derivative of loss with respect to b

    w = w - learning_rate * (1/n) * dldw  # Update weight
    b = b - learning_rate * (1/n) * dldb  # Update bias

    return w, b

# iteratively make updates 
def gradient_descent(x,y,w,b,learning_rate):
    for epoch in range(400):
        #run gradient descent
        wf,bf = descent(x, y, w, b, learning_rate) # f = final
        yhat = wf*x + bf # predicted values
        loss = np.divide(np.sum((y-yhat)**2, axis=0), x.shape[0]) # calculate loss
        print(f"{epoch} loss is {loss}, parameters w: {wf}, b: {bf}")  # print loss and parameters