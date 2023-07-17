import numpy as np

def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) where x[i] is the ith input.
  We multiply this against a weight matrix of shape (D, M) where
  D = \prod_i d_i

  Inputs:
  x - Input data, of shape (N, d_1, ..., d_k)
  w - Weights, of shape (D, M)
  b - Biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  x_vec = np.reshape(x,(x.shape[0],np.prod(x.shape[1:len(x.shape)])))
  out = np.dot(x_vec,w)+b
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  x_vec = np.reshape(x,(x.shape[0],np.prod(x.shape[1:len(x.shape)])))
  dw = np.dot(x_vec.T,dout)
  db=np.sum(dout,axis=0)
  dx = np.reshape(np.dot(dout, w.T), x.shape)
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = np.maximum(0,x)
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  dx = dout*(x>0)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx
    
def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  (N, C, H, W) = x.shape
  (F, C, HH, WW) = w.shape

  stride = conv_param['stride']
  pad = conv_param['pad']

  h_bar = int(1 + (H + 2 * pad - HH) / stride)
  w_bar = int(1 + (W + 2 * pad - WW) / stride)
  out = np.zeros((N,F,h_bar,w_bar))

  x_pad = np.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)),'constant',constant_values=(0))

  for i in range(N):
      for f in range(F):
          for row in range(h_bar):
              for col in range(w_bar):
                  x_cut = x_pad[i,:,row*stride:row*stride+HH,col*stride:col*stride+WW]
                  out[i, f, row, col] = np.sum(np.multiply(x_cut, w[f,:,:,:]))
                  out[i, f, row, col] += b[f]
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """

  x,w,b,conv_param = cache
  dx = np.zeros_like(x)
  dw = np.zeros_like(w)
  db = np.zeros_like(b)
  stride = conv_param['stride']
  pad = conv_param['pad']
  (N, C, H, W) = x.shape
  (F, C, HH, WW) = w.shape
  h_bar = int(1 + (H + 2 * pad - HH) / stride)
  w_bar = int(1 + (W + 2 * pad - WW) / stride)
  x_pad = np.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)),'constant',constant_values=(0))
  dx_pad = np.pad(dx,((0,0),(0,0),(pad,pad),(pad,pad)),'constant',constant_values=(0))
  

    
  for i in range(N):
    for f in range(F):
        for row in range(h_bar):
          for col in range(w_bar):
            x_cut = x_pad[i,:,row*stride:row*stride+HH,col*stride:col*stride+WW]
            db[f] += dout[i,f,row,col]
            dw[f] += x_cut*dout[i,f,row,col]
            dx_pad[i,:,row*stride:row*stride+HH,col*stride:col*stride+WW] += w[f]*dout[i,f,row,col]
            
  dx=dx_pad[:,:,pad:pad+H,pad:pad+W]
            
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  (N, C, H, W) = x.shape
  stride = pool_param['stride']
  WW = pool_param['pool_width']
  HH = pool_param['pool_height']
  h_bar = int(1+(H-HH)/stride)
  w_bar = int(1+(W-WW)/stride)
  out = np.zeros((N,C,h_bar,w_bar))
    
  for i in range(N):
    for c in range(C):
        for row in range(h_bar):
            for col in range(w_bar):
                out[i,c,row,col] = np.max(x[i,c,row*stride:row*stride+HH, col*stride:col*stride+WW])
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  x=cache[0]
  pool_param=cache[1]
  (N, C, H, W) = x.shape
  stride = pool_param['stride']
  WW = pool_param['pool_width']
  HH = pool_param['pool_height']
  N,C,h_bar,w_bar = dout.shape
  dx = np.zeros((N,C,H,W))
    
  for i in range(N):
    for c in range(C):
        for row in range(h_bar):
            for col in range(w_bar):
                x_cut = x[i, c, row*stride:row*stride+HH, col*stride:col*stride+WW]
                [row_m, col_m] = np.unravel_index(np.argmax(x_cut), x_cut.shape)
                dx[i, c, row*stride+row_m, col*stride+col_m] += dout[i, c, row, col]
  return dx


def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx

