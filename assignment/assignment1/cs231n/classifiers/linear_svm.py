import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1] # 10
  num_train = X.shape[0]   # 500
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)    # x[i](500, 3073) w :(3073, 10)    
    correct_class_score = scores[y[i]] 
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        #dWyi=−xi(∑j≠yi1(xi⋅Wj−xi⋅Wyi+1>0))  （j=yi） 对Wyi求偏导
        dW[:,y[i]]  += -X[i,:] 
        #dWj=xi⋅1(xi⋅Wj−xi⋅Wyi+1>0)  (j≠yi) 对wj求偏导
        dW[:,j] += X[i,:]
        
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W    # W^2 求导 
  
  

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
 
    
# =============================================================================
#  Inputs:
#   - W: A numpy array of shape (D, C) containing weights.
#   - X: A numpy array of shape (N, D) containing a minibatch of data.
#   - y: A numpy array of shape (N,) containing training labels; y[i] = c means
#     that X[i] has label c, where 0 <= c < C.
#   - reg: (float) regularization strength
#   
# =============================================================================
  # compute the loss and the gradient
  
  num_train = X.shape[0]   # 500
  loss = 0.0
  scores =X.dot(W) #(n,c) 500,10
  correct_class_score =scores[np.arange(num_train),y].reshape((-1, 1))  # (500,1)reshapeoperands could not be broadcast together with shapes (500,10) (500,)
  mask= (scores -correct_class_score +1) >0  # (500,10 )(n,c)-(n,1)
  scores= (scores -correct_class_score +1)*mask  
  
  loss = (np.sum(scores)- num_train * 1) / num_train #- num_train * 1 不相减结果差1
  loss += reg * np.sum(W * W)

 
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  ds =np.ones_like(scores) #(500,10)
  ds *=mask
  ds[np.arange(num_train), y] = -1 * (np.sum(mask, axis=1) - 1)
  dW = np.dot(X.T, ds) / num_train  #(D,N) (N,C)-->(D,C) 3073,10
  dW += 2 * reg * W
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
