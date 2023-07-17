import numpy as np
from random import shuffle
import math

def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
  """
  # Initialize the loss and gradient to zero.
  
#   bias = np.zeros((1,len(y)))
  loss = 0.0
  N=X.shape[1]
  dW = np.zeros_like(W)
  z = np.dot(W,X).T
  z=z.astype(np.float128)
  exp_z = np.exp(z)
  p = exp_z/np.sum(exp_z,axis=1,keepdims=True) # softmax probababilities
    
  log_p = -np.log(p[range(N),y])
  loss = np.sum(log_p)/N
    
  regu_loss = 0.5*reg*np.sum(W * W)
  loss = loss+regu_loss
    
  dp = np.copy(p)

  dp[np.arange(N),y] -=1
  dW = np.dot(X,dp).T
  dW = dW/N +reg*W

  return loss, dW
