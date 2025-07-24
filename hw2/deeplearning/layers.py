import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    #############################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You     #
    # will need to reshape the input into rows.                                 #
    #############################################################################
    
    # First reshape x to (N, D) matrix
    D = 1
    for dim in x.shape[1:]: D *= dim
    x_reshape = x.reshape(x.shape[0], D) # note: analogous to torch.Tensor.view (??)

    # Then calculate out
    out = x_reshape @ w + b

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
    #############################################################################
    # TODO: Implement the affine backward pass.                                 #
    #############################################################################
    
    # Remember: out (N, M) = x_reshape (N, D) * w (D, M) + b (M)
    # So: dx = w * dout, dw = x_reshape * dout, db = dout
    # Similar to what was learned in Karpathy's micrograd lecture.

    D = 1
    for dim in x.shape[1:]: D *= dim
    x_reshape = x.reshape(x.shape[0], D)

    # we transpose accordingly to match matmul operations
    dx = (dout @ w.T).reshape(x.shape)
    dw = (x_reshape.T @ dout).reshape(w.shape)
    db = dout.sum(0, keepdims=True).reshape(b.shape)

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
    out = None
    #############################################################################
    # TODO: Implement the ReLU forward pass.                                    #
    #############################################################################
    
    # store as a deep copy beforehand
    out = x.copy()
    out[out < 0] = 0

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
  
    # Remember: out = x (if x >= 0), 0 (if x < 0)
    # So: dx = dout (if x >= 0), 0 (if x < 0)

    # Initially store as a deep copy of dout
    dx = dout.copy()
    zeroed_out = np.where(x < 0) # you want to find index of where x < 0 first...
    dx[zeroed_out] = 0           # then set to 0.

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the mean
    and variance of each feature, and these averages are used to normalize data
    at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7 implementation
    of batch normalization also uses running averages.

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
        #############################################################################
        # TODO: Implement the training-time forward pass for batch normalization.   #
        # Use minibatch statistics to compute the mean and variance, use these      #
        # statistics to normalize the incoming data, and scale and shift the        #
        # normalized data using gamma and beta.                                     #
        #                                                                           #
        # You should store the output in the variable out. Any intermediates that   #
        # you need for the backward pass should be stored in the cache variable.    #
        #                                                                           #
        # You should also use your computed sample mean and variance together with  #
        # the momentum variable to update the running mean and running variance,    #
        # storing your result in the running_mean and running_var variables.        #
        #############################################################################
        
        sample_mean = x.mean(0)
        sample_var = ((x - sample_mean) ** 2).mean(0)
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        norm_x = (x - sample_mean) / np.sqrt(sample_var + eps)
        out = gamma * norm_x + beta

        cache = (x, sample_var, eps, norm_x, gamma)

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
    elif mode == 'test':
        #############################################################################
        # TODO: Implement the test-time forward pass for batch normalization. Use   #
        # the running mean and variance to normalize the incoming data, then scale  #
        # and shift the normalized data using gamma and beta. Store the result in   #
        # the out variable.                                                         #
        #############################################################################
        
        sample_mean = x.mean(0)
        sample_var = ((x - sample_mean) ** 2).mean(0)
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        norm_x = (x - sample_mean) / np.sqrt(sample_var + eps)
        out = gamma * norm_x + beta

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
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
    #############################################################################
    # TODO: Implement the backward pass for batch normalization. Store the      #
    # results in the dx, dgamma, and dbeta variables.                           #
    #############################################################################
    
    # Inspo: "Building makemore Part 4: Becoming a Backprop Ninja", Karpathy on YouTube
    x, var, eps, norm_x, gamma = cache
    
    # Remember: y = gamma * norm_x + beta
    dgamma = (norm_x * dout).sum(0)
    dbeta = dout.sum(0)

    # The tricky part is figuring out dx.
    # In the computation graph, there are three pathways. Calculate each and sum.
    # Karpathy uses Bessel's correction (1 / n-1), we ignore that here.
    n = len(x) # For simplicity
    dx = (gamma * ((var + eps) ** -0.5) / n) * (n * dout - dout.sum(0) - norm_x * (dout * norm_x).sum(0))


    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    #############################################################################
    # TODO: Implement the backward pass for batch normalization. Store the      #
    # results in the dx, dgamma, and dbeta variables.                           #
    #                                                                           #
    # After computing the gradient with respect to the centered inputs, you     #
    # should be able to compute gradients with respect to the inputs in a       #
    # single statement; our implementation fits on a single 80-character line.  #
    #############################################################################
    pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout
        and rescale the outputs to have the same mean as at test time.
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not in
        real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        ###########################################################################
        # TODO: Implement the training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                            #
        ###########################################################################

        mask = np.random.rand(*x.shape) < p
        out = x.copy()
        out[mask] = 0

        ###########################################################################
        #                            END OF YOUR CODE                             #
        ###########################################################################
    elif mode == 'test':
        out = x

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
        ###########################################################################
        # TODO: Implement the training phase backward pass for inverted dropout.  #
        ###########################################################################
        
        # Remember: out = 0 (p% chance) or x (1-p% chance)
        dx = dout.copy()
        dx[mask] = 0

        ###########################################################################
        #                            END OF YOUR CODE                             #
        ###########################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and width
    W. We convolve each input with F different filters, where each filter spans
    all C channels and has height HH and width WW.

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
    out = None
    #############################################################################
    # TODO: Implement the convolutional forward pass.                           #
    # Hint: you can use the function np.pad for padding.                        #
    #############################################################################
    
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    S, P = conv_param['stride'], conv_param['pad']
    out = []
    for n in range(N): # loop thru all N
      out.append([])
      for f in range(F):
        vals = []
        for i in range(0, H + 2 * P - HH + 1, S): # start from 0, up to padded height - filter height, with stride s
            vals.append([])
            for j in range(0, W + 2 * P - WW + 1, S): # start from 0, up to padded width - filter width, with stride s
                val = 0
                for c in range(C): # each channel
                    padded_img = np.pad(x[n, c], P, mode='constant')
                    cut = padded_img[i:i+HH, j:j+WW]
                    filter = w[f, c]
                    val += (cut * filter).sum()
                vals[-1].append(val + b[f])
        out[-1].append(vals)
    
    out = np.array(out)
            

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
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
    #############################################################################
    # TODO: Implement the convolutional backward pass.                          #
    #############################################################################
    
    # Inspo: https://pavisj.medium.com/convolutions-and-backpropagations-46026a8f5d2c
    # We treat this as a convolution in itself.
    # dw = conv(x, dout)
    # dx = conv(180 deg rotated w, dout)
    
    # dx: shape (N, C, H, W)
    # dw: shape (F, C, HH, WW)
    # db: shape (F,)

    x, w, _, conv_param = cache

    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    _, _, HO, WO = dout.shape # "H out" and "W out" needed for convolution
    S, P = conv_param['stride'], conv_param['pad']
    dx, dw, db = [], [], [] # dx: N/C/H/W, dw: F/C/HH/WW, db: F/

    for n in range(N): # loop thru each image
      dx.append([])
      for c in range(C): # each channel
        dx_vals = []
        for i in range(0, HO + 2 * P - HH + 1, S): # start from 0, up to out padded height - filter height, with stride s
            dx_vals.append([])
            for j in range(0, WO + 2 * P - WW + 1, S): # start from 0, up to out padded width - filter width, with stride s
                val = 0
                for f in range(F): # aggregate each filter
                    # For images, we slide a rotated filter over a padded dout.
                    filter = np.rot90(np.rot90(w[f, c])) # filter rotated 180˚
                    padded_out = np.pad(dout[n, f], P, mode='constant')
                    padded_out_cut = padded_out[i:i+HH, j:j+WW]
                    val += (filter * padded_out_cut).sum()
                dx_vals[-1].append(val)
        dx[-1].append(dx_vals)

    for f in range(F): # loop thru each filter
        dw.append([])
        for c in range(C): # each channel
            dw_vals = []
            for i in range(0, HO + 2 * P - H + 1, S): # start from 0, up to out padded height - x height, with stride s
                dw_vals.append([])
                for j in range(0, WO + 2 * P - W + 1, S): # start from 0, up to out padded width - x width, with stride s
                    val = 0
                    for n in range(N): # aggregate each image
                        # For filters, we slide dout over a padded image.
                        padded_img = np.pad(x[n, c], P, mode='constant')
                        cut = padded_img[i:i+HO, j:j+WO]
                        out_cut = dout[n, f]
                        val += (cut * out_cut).sum()
                    dw_vals[-1].append(val)
            dw[-1].append(dw_vals)

    for f in range(F): # loop thru each filter
        val = 0
        for n in range(N): # aggregate each image
            val += dout[n, f].sum()
        db.append(val)
            
    dx, dw, db = np.array(dx), np.array(dw), np.array(db)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
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
    out = None
    #############################################################################
    # TODO: Implement the max pooling forward pass                              #
    #############################################################################
    
    N, C, H, W = x.shape
    PH, PW, S = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    
    out = []
    for n in range(N):
        out.append([])
        for c in range(C):
            out_vals = []
            for i in range(0, H - PH + 1, S):
                vals = []
                for j in range(0, W - PW + 1, S):
                    chunk = x[n, c, i:i+PH, j:j+PW]
                    val = np.max(chunk)
                    vals.append(val)
                out_vals.append(vals)
            out[-1].append(out_vals)

    out = np.array(out)

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
    dx = None
    #############################################################################
    # TODO: Implement the max pooling backward pass                             #
    #############################################################################
    
    # out = pool(x), where pool just takes chunks of x and outputs maxes.
    # So dx = 0 in spots where it wasn't the max of its chunk, dout where it was.
    
    x, pool_param = cache
    N, C, H, W = x.shape
    PH, PW, S = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    dx = []
    for n in range(N):
        dx.append([])
        for c in range(C):
            dx_vals = []
            for i in range(0, H):
                vals = [] # should mostly contain zeros!
                for j in range(0, W):
                    cval = x[n, c, i, j]
                    c_i, c_j = i//PH, j//PW # define "chunk i" and "chunk j" pooling indices
                    chunk = x[n, c, c_i*PH:c_i*PH+PH, c_j*PW:c_j*PW+PW]
                    mval = np.max(chunk)
                    if cval == mval: vals.append(dout[n, c, c_i, c_j])
                    else: vals.append(0)
                dx_vals.append(vals)
            dx[-1].append(dx_vals)

    dx = np.array(dx)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
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

    #############################################################################
    # TODO: Implement the forward pass for spatial batch normalization.         #
    #                                                                           #
    # HINT: You can implement spatial batch normalization using the vanilla     #
    # version of batch normalization defined above. Your implementation should  #
    # be very short; ours is less than five lines.                              #
    #############################################################################
    
    mode = bn_param['mode']
    eps, momentum = bn_param.get('eps', 1e-5), bn_param.get('momentum', 0.9)

    N, C, H, W = x.shape # images, channels, height, width

    running_mean = bn_param.get('running_mean', np.zeros((C, H, W), dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros((C, H, W), dtype=x.dtype))

    out, cache = None, None

    sample_mean = x.mean(0) # mean over all images
    sample_var = ((x - sample_mean) ** 2).mean(0) # variance over all images
    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var
    norm_x = (x - sample_mean) / np.sqrt(sample_var + eps)
    # definitely a nicer way to do this, but I'm on a plane rn :)
    out = np.array([(gamma.reshape(C, 1) * norm_x[i].reshape(C, H*W) + beta.reshape(C, 1)).reshape(C, H, W) for i in range(norm_x.shape[0])])

    if mode == 'train':
      cache = (x, sample_var, eps, norm_x, gamma)
    
    if mode not in ['train', 'test']: raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

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

    #############################################################################
    # TODO: Implement the backward pass for spatial batch normalization.        #
    #                                                                           #
    # HINT: You can implement spatial batch normalization using the vanilla     #
    # version of batch normalization defined above. Your implementation should  #
    # be very short; ours is less than five lines.                              #
    #############################################################################
    
    # Same implementation as vanilla BN
    x, var, eps, norm_x, gamma = cache
    # x: N/C/H/W
    # var: C/H/W
    # eps: scalar
    # norm_x: N/C/H/W
    # gamma: C/

    # dout: N/C/H/W
    
    N, C, H, W = x.shape

    # Remember: y = gamma * norm_x + beta
    dgamma = (norm_x * dout).sum(0).reshape(C, H*W).sum(1)
    dbeta = dout.sum(0).reshape(C, H*W).sum(1)

    n = len(x) # For simplicity
    
    dx = (gamma.reshape(C, 1) * ((var + eps) ** -0.5).reshape(C, H*W) / n).reshape(1, C, H, W) * (n * dout - dout.sum(0) - norm_x * (dout * norm_x).sum(0))

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return dx, dgamma, dbeta


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
