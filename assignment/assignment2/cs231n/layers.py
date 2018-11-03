# -*- coding: utf-8 -*-

from builtins import range
import numpy as np
import numbers

def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)  #(2,4,5,6)
    - w: A numpy array of weights, of shape (D, M)  #(120,3) 
    - b: A numpy array of biases, of shape (M,) #(3,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    D = w.shape[0]
    x1 =x.reshape(-1,D) 
    out = np.dot(x1,w) +b
     
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
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
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # z= wx +b ---->l   #▽Wl = ▽zl X^T   #▽bl = ▽zl * 1    #▽xl = ▽zl * w 
     
    dx =  np.dot (dout,w.T)      # (N,M)  (M,D)------> (N,D)
    dx =  dx.reshape(x.shape)      # (N,D)----> (N, d1, ..., d_k)
    dw = np.dot ( x.reshape(x.shape[0],-1).T ,dout   )   # (N,D)^T   (N,M) ---(D,M)
    db = dout.sum(axis=0)  # dout(N,M) --->(M,1)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
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
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = np.maximum(x, 0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
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
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
     
    dx=dout* (x>0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        # 
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        
        #公式： https://arxiv.org/abs/1502.03167
        mean_x = np.mean(x, axis = 0 )
        var_x = np.var(x, axis = 0)
        x_hat =( x - mean_x) / np.sqrt(var_x +  eps )
        out = gamma* x_hat + beta        
        running_mean = momentum * running_mean + (1 - momentum) * mean_x
        running_var = momentum * running_var + (1 - momentum) * var_x
        inv_var_x = 1 / np.sqrt(var_x +  eps)
        cache =(x,x_hat,gamma,mean_x,inv_var_x)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
       
        x_hat =( x - running_mean) / np.sqrt(running_var +  eps )
        out = gamma* x_hat + beta     
        
    
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
# =============================================================================
#     xi ----- uB----- o^2 B------------xi^--------------yi----------l 
#          xi----- 
#                           ub----            gamma--
#                           xi----            betla-- 
# =============================================================================
    
    x,x_hat,gamma,mean_x,inv_var_x = cache
    N = x.shape[0]
    # dx 求导合并：
    #1: l--->xi^--->xi
    dx= gamma * dout * inv_var_x
    #2: l----> o^2 B--->xi
    dx += (2 / N) * (x - mean_x) * np.sum(- (1/2) * inv_var_x ** 3 * (x - mean_x) * gamma * dout, axis=0)
    
    #3: l----> uB--->xi
    dx += (1 / N) * np.sum(-1 * inv_var_x * gamma * dout, axis=0)   
    
    # dgamma求导：l----> yi--->gamma 
    dgamma = np.sum(x_hat * dout, axis=0)
    
    # dbeta求导：l----> yi--->betla 
    dbeta = np.sum(dout, axis=0)




    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass. 
    See the jupyter notebook for more hints.
     
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    #https://blog.csdn.net/duan_zhihua/article/details/83107615
    x, x_hat, gamma, mean_x,inv_var_x = cache
    N = x.shape[0]
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(x_hat * dout, axis=0)
    dxhat = dout * gamma
    dx = (1. / N) * inv_var_x * (N * dxhat - np.sum(dxhat, axis=0) -
                                 x_hat * np.sum(dxhat * x_hat, axis=0))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.
    
    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get('eps', 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    #x: (N, D) ---->(D,N)
    x = x.T
    mean_x = np.mean(x,axis =0)
    var_x= np.var(x,axis = 0)
    inv_var_x = 1 / np.sqrt(var_x + eps)
    
    x_hat = (x - mean_x)/np.sqrt(var_x + eps) #(D,N)
    x_hat = x_hat.T #(D,N)---->(N,D)
    # gamma: (D,)  beta: (D,)
    out = gamma * x_hat + beta  
    cache =(x,x_hat,gamma,mean_x,inv_var_x)


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
     
    x, x_hat, gamma, mean_x,inv_var_x = cache
    d = x.shape[0]
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(x_hat * dout, axis=0)
    dxhat = dout * gamma
    dxhat = dxhat.T
    x_hat = x_hat.T
    dx = (1. / d) * inv_var_x * (d * dxhat - np.sum(dxhat, axis=0) -
                                 x_hat * np.sum(dxhat * x_hat, axis=0))    
    dx = dx.T 
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask
        
       # pass
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        
        out = x 
        
        
        #pass
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        
        dx = dout * mask 
        
        
        
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
     
    stride = conv_param['stride']
    padding = conv_param['pad']
    if isinstance(stride, numbers.Number):
        stride = (stride, stride)  #
    if isinstance(padding, numbers.Number):
        padding = [(padding, padding), (padding, padding)]
    else:
        padding = [(i,) * 2 for i in padding]
    pad = [(0, 0), (0, 0)]
    pad.extend(padding)
    x_pad = np.pad(x, pad_width=pad, mode='constant', constant_values=0)
    n, c, pad_h, pad_w = x_pad.shape
    f, w_c, hh, ww = w.shape
    assert c == w_c, 'input channels must equal to filter channels'
    out_h = (pad_h - hh) // stride[0] + 1
    out_w = (pad_w - ww) // stride[1] + 1
    out = np.zeros(shape=(n, f, out_h, out_w))
    for i in range(n):  # 每个样本点
        for j in range(f):  # 每个filter  汇总
            for _w in range(out_w):  # 水平方向
                for _h in range(out_h):  # 竖直方向
                    vert_start =  _h*stride[1]
                    vert_end   =  _h*stride[1] + hh 
                    horiz_start = _w*stride[0]
                    horiz_end   = _w*stride[0] + ww     
                    out[i, j, _h, _w] = np.sum(
                        x_pad[i, :, vert_start: vert_end, horiz_start:horiz_end] * w[j]) + b[j]
                    #print (i,"\t",j,"\t",_w,"\t", _h,"\t",stride[1],"\t",stride[0] ,"\t",w[j].shape)



    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
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
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    #pass
    x, w, b, conv_param = cache
    stride = conv_param['stride']
    padding = conv_param['pad']
    if isinstance(stride, numbers.Number):
        stride = (stride, stride)  #
    if isinstance(padding, numbers.Number):
        padding = [(padding, padding), (padding, padding)]
    else:
        padding = [(i,) * 2 for i in padding]
    pad = [(0, 0), (0, 0)]
    pad.extend(padding)
    x_pad = np.pad(x, pad_width=pad, mode='constant', constant_values=0)
    n, c, pad_h, pad_w = x_pad.shape
    f, w_c, hh, ww = w.shape
    assert c == w_c, 'input channels must equal to filter channels'
    out_h = (pad_h - hh) // stride[0] + 1
    out_w = (pad_w - ww) // stride[1] + 1
    
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    dx_pad = np.zeros_like(x_pad)    

    for i in range(n):  # 每个样本点
        for j in range(f):  # 每个filter
            for _w in range(out_w):  # 水平方向
                for _h in range(out_h):  # 竖直方向
                    dw_vert_start =  _h*stride[1]
                    dw_vert_end   =  _h*stride[1]+ ww
                    dw_horiz_start = _w*stride[0]
                    dw_horiz_end   = _w*stride[0]+ hh  
                    #dw[j] 跟filter相关，一个filter对应一个卷积核w ，对应的x_pad区域 链式求导，简化理解wx=out  dl/dw = dout*dx  ww、hh进行了交换。 
                    dw[j] += dout[i, j, _h, _w] * x_pad[i, :, dw_vert_start: dw_vert_end, dw_horiz_start:dw_horiz_end]
                    #db[j]跟filter相关，一个filter对应一个b，因此在一个filter上将各位维度（不同样本，不同的w、h）中b的变化量进行累加。
                    db[j] += dout[i, j, _h, _w]
                    
                    dx_pad_vert_start =  _h*stride[1]
                    dx_pad_vert_end   =  _h*stride[1]+ hh
                    dx_pad_horiz_start = _w*stride[0]
                    dx_pad_horiz_end   = _w*stride[0]+ ww  
                    
                    #dx_pad 跟各个维度相关 (不同样本，不同的过滤器、不同的w、h）  简化理解wx=out  dl/dx = dout*dw   每个过滤器对应一个dw。               
                    dx_pad[i, :, dx_pad_vert_start:dx_pad_vert_end, dx_pad_horiz_start: dx_pad_horiz_end] += \
                        dout[i, j, _h, _w] * w[j]
                    
    # 从dx_pad 中获取 dx的内容。pad: [(0, 0), (0, 0),(1, 1), (1, 1)] 去掉 h（竖直方向）、w（水平方向）的第一行及最后一行pad。               
    dx = dx_pad[:, :, pad[2][0]:-pad[2][1], pad[3][0]:-pad[3][1]] 
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    #pass
    pool_height=pool_param['pool_height']
    pool_width=pool_param['pool_width']
    stride=pool_param['stride']
    n, c, h, w = x.shape
    out_h = 1 + (h - pool_height) // stride
    out_w =1 + (w - pool_width) // stride
    out = np.zeros(shape=(n, c, out_h, out_w))
    for i in range(n):  # 每个样本点  
         for j in range(c):# 每个过滤器，分别计算max
            for _w in range(out_w):  # 水平方向
                for _h in range(out_h):  # 竖直方向
                    vert_start =  _h*stride
                    vert_end   =  _h*stride + pool_height 
                    horiz_start = _w*stride
                    horiz_end   = _w*stride + pool_width     
                    out[i, j, _h, _w] = np.max(
                        x[i,j, vert_start: vert_end, horiz_start:horiz_end] )                       


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    #pass
    x, pool_param = cache
    pool_height=pool_param['pool_height']
    pool_width=pool_param['pool_width']
    stride=pool_param['stride']
    n, c, h, w = x.shape
    out_h = 1 + (h - pool_height) // stride
    out_w =1 + (w - pool_width) // stride

    dx  = np.zeros_like(x)    
    
    for i in range(n):  # 每个样本点
        for j in range(c):  # 每个filter
            for _w in range(out_w):  # 水平方向
                for _h in range(out_h):  # 竖直方向
                    dx_vert_start =  _h*stride
                    dx_vert_end   =  _h*stride+ pool_height
                    dx_horiz_start = _w*stride
                    dx_horiz_end   = _w*stride+ pool_width  
                    
                    indices = np.unravel_index(np.argmax(x[i, j,dx_vert_start:dx_vert_end,
                                                         dx_horiz_start: dx_horiz_end]), dims=(pool_height, pool_width))

                    dx[i, j, _h*stride+indices[0], _w*stride+indices[1]] += dout[i, j, _h, _w]
 
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    #pass
    N, C, H, W = x.shape
    x_new = np.transpose(x, axes=(0, 2, 3, 1)).reshape((-1, C)) # N,H,W,C--->D,C
    out, cache = batchnorm_forward(x_new, gamma, beta, bn_param) #(D,C)
    out = out.reshape((N, H, W, C)).transpose((0, 3, 1, 2)) #(D,C)---> N,H,W,C--->N,C,H,W


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
   # pass
 
    N,C,H,W = dout.shape
    # out forward:D,C---> N,H,W,C--->N,C,H,W
    # dout backward:N,C,H,W--->N,H,W,C---->D,C
    dout = dout.transpose((0, 2, 3, 1)).reshape((-1, C))    
    dx_new, dgamma, dbeta = batchnorm_backward(dout, cache) 
    # x_new forward  N,H,W,C--->D,C          
    # dx backward: D,C---->  N,H,W,C ----->(N, C, H, W) 
    dx = dx_new.reshape((N, H, W, C)).transpose((0, 3, 1, 2))  


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry 
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get('eps',1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                # 
    ###########################################################################
    #pass
    N, C, H, W = x.shape
    # 按分组g将大的立方体积木拆成 C/G个小积木体。
    #N, C, H, W = 2, 6, 4, 5;G = 2 ;这里g为2个一组，拆成6/2=3组小立方体。
    x = x.reshape((N * G, C // G * H * W)) #(N, C, H, W)--->(N * G, C // G * H * W) 
    #接下来就可以将每1个小立方体作为一个Layer Norm的模块去处理。     
    x = x.T #(C // G * H * W,N * G)
    mean_x = np.mean(x,axis =0)
    var_x= np.var(x,axis = 0)
    inv_var_x = 1 / np.sqrt(var_x + eps)
    
    x_hat = (x - mean_x)/np.sqrt(var_x + eps) ##(C // G * H * W,N * G)
    x_hat = x_hat.T #(C // G * H * W,N * G)---->(N * G, C // G * H * W)
    
    
    x_hat = x_hat.reshape((N, C, H, W)) #(N * G, C // G * H * W)---->(N, C, H, W)
   

    # gamma: (1,C,1,1)  beta: (1,C,1,1)
    out = gamma * x_hat + beta  
    cache =( x_hat,gamma,mean_x,inv_var_x, G)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    pass

    x_hat,gamma,mean_x,inv_var_x, G = cache 
   
    #x_hat :(N, C, H, W)
    N, C, H, W = x_hat.shape
    # 在(N, H, W)维度上计算
    dgamma = np.sum(dout * x_hat, axis=(0, 2, 3), keepdims=True)
    dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)
    #forward时拆分成几个小立方体积来计算的，backward反向传播时仍需分组拆成几个小立方体计算。
   
    #dout :(N, C, H, W)--->(N * G, C // G * H * W) ---->(C // G * H * W, N * G)   
    dxhat = (dout * gamma).reshape((N * G, C // G * H * W)).T 
    
    #x_hat:(N, C, H, W)--->(N * G, C // G * H * W) ---->(C // G * H * W, N * G)
    x_hat = x_hat.reshape((N * G, C // G * H * W)).T    
    
    # d:   C // G * H * W 将每1个小立方体作为一个Layer Norm的反向backward模块去处理
    d = x_hat.shape[0]
    dx = (1. / d) * inv_var_x * (d * dxhat - np.sum(dxhat, axis=0) -
                                 x_hat * np.sum(dxhat * x_hat, axis=0))    
    
    
    dx = dx.T #(C // G * H * W, N * G) ----->(N * G, C // G * H * W) 
    # 将几个小立方体再重新拼接成一个大立方体
    dx = dx.reshape((N, C, H, W)) #(N * G, C // G * H * W) --->(N, C, H, W)
    
   

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
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
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True) # 数值稳定性
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)   #  ∑j e^ sj
    log_probs = shifted_logits - np.log(Z)  
    probs = np.exp(log_probs)   # e^(m-n) = e^m / e^n  ---->  softmax  e^ syi /  ∑j e^ sj
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N   # -log ai ---> loss = -np.log(np.sum (probs[np.arange(N), y]) ) / N  ?
    dx = probs.copy()
    dx[np.arange(N), y] -= 1   #i=j  ∂loss/∂zi = ∂loss/∂ai *   ∂ai/∂zi =  - 1/ai   *   ai(1−ai) =  ai - 1 
                               #i!=j  ∂loss/∂zi = ∂loss/∂ai *   ∂ai/∂zi =  - 1/ai  * -aiaj =  ai 
    dx /= N
    return loss, dx