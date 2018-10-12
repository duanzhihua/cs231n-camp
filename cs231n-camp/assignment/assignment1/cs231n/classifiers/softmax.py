import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  #print( dW)
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0] 
  num_classes = W.shape[1] 
  for i in range(num_train):
      #foward
      score=np.dot(X[i],W) # (D,) (D,C) --->(C,)
      score -= max(score)
      score = np.exp(score) / np.sum(np.exp(score))  # e^si / sum(e^sk)
      #backward  
      for j in range(num_classes):
           if j == y[i]:   
               dW[:,j] += (score[j]-1) * X[i]
# =============================================================================
#                print( "X"+str(i))
#                print(X[i])
#                print("score"+str(j))
#                print (score[j])  
#                print("i = ",i,"\t j=",j,"\t y[i]",y[i],"\t dW :",)
#                print( dW)
# =============================================================================
           else:
               dW[:,j] += score[j] * X[i]
# =============================================================================
#                print( "X"+str(i))
#                print(X[i])
#                print("score"+str(j))
#                print (score[j])               
#                print("i = ",i,"\t j=",j,"\t y[i]",y[i])
#                print( dW)
#            print ("========================")
#            print()
# =============================================================================
      loss -= np.log(score[y[i]])
  loss /= num_train
  dW /=num_train
  loss += reg *np.sum(W*W)
  dW += 2 *reg *W
    
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = np.dot(X, W) #(N, D) (D, C) --->(N,C)
  scores -= np.max(scores, axis=1, keepdims=True)  # 数值稳定性
  scores = np.exp(scores)
  scores /= np.sum(scores, axis=1, keepdims=True)  # softmax
  ds = np.copy(scores) #(N,C)
  ds[np.arange(X.shape[0]), y] -= 1  # j == y[i]
  dW = np.dot(X.T, ds) #(D,N) (N,C)--->(D,C)
  loss = scores[np.arange(X.shape[0]), y]
  loss = -np.log(loss).sum()
  loss /= X.shape[0]
  dW /= X.shape[0]
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

