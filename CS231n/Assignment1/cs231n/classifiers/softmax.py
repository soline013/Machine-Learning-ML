import numpy as np
from random import shuffle
from past.builtins import xrange

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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  for i in xrange(num_train):
      sco = np.dot(X[i], W)
      exp_sco = np.exp(sco)
      sum_sco = np.sum(exp_sco)
      loss -= np.log(exp_sco[y[i]] / sum_sco)
      dW += np.dot(X[i].T[:, np.newaxis], exp_sco[np.newaxis, :]) / sum_sco
      dW[:, y[i]] -= X[i].T
  loss = loss/X.shape[0] + reg*np.sum(W**2)
  dW = dW/X.shape[0] + 2*reg*W
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
  '''
  sco = np.dot(X, W)
  sco[np.arange(sco.shape[0])] -= np.reshape(np.max(sco, axis=1),[-1, 1])
  exp_sco = np.exp(sco)
  sum_sco = np.sum(exp_sco) 
  softmax_sco = exp_sco / sum_sco
  cross_entropy = np.sum(-np.log(softmax_sco))
  loss = cross_entropy/X.shape[0] + reg*np.sum(W**2)

  dsco = exp_sco / (exp_sco.sum(axis=1)[:,np.newaxis])
  dsco[range(X.shape[0]), y] -= 1
  dsco /= X.shape[0]
  dW = np.dot(X.T, dsco) + 2*reg*W
  '''

  sco = np.dot(X, W)
  correct_sco = sco[range(X.shape[0]), y]
  exp_ = np.exp(sco - correct_sco[:, np.newaxis])
  log_ = np.log(exp_.sum(axis=1))
  loss = np.sum(log_) / X.shape[0]
  loss += reg * np.sum(W**2)

  dsco = exp_ / (exp_.sum(axis=1)[:, np.newaxis])
  dsco[range(X.shape[0]), y] -= 1
  dsco /= X.shape[0]
  dW = np.dot(X.T, dsco) + 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW