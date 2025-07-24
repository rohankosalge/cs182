import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """
    next_h, cache = None, None
    ##############################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
    # hidden state and any values you need for the backward pass in the next_h   #
    # and cache variables respectively.                                          #
    ##############################################################################
    
    comb = np.hstack((prev_h, x)) # (N, H + D)
    comb_weights = np.vstack((Wh, Wx)) # (H + D, H)

    # Following the slide: a_l = sig(W * a_(l-1) + b)
    next_h = np.tanh((comb_weights.T @ comb.T).T + b) # (N, H)
    cache = (comb, comb_weights, next_h)

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.

    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state
    - cache: Cache object from the forward pass

    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
    #                                                                            #
    # HINT: For the tanh function, you can compute the local derivative in terms #
    # of the output value from tanh.                                             #
    ##############################################################################
    
    # Unpack cache
    comb, comb_weights, next_h = cache

    # Remember: a_l = sig(W * a_(l-1) + b)
    # Or: next_h = np.tanh(W * (prev_h + x) + b)

    # if z = W * (prev_h + x) + b, next_h = np.tanh(z), 
    # dz = 1 - next_h^2.

    H = next_h.shape[1] # Since next_h: (N, H)
    # z: (N, H), comb: (N, H + D), W: (H + D, H)
    dz = (1 - next_h ** 2) * dnext_h # Inspo: Karpathy's micrograd lecture
    db = dz.sum(0, keepdims=True)
    dW = comb.T @ dz # (H + D, H)
    dWx, dWh = dW[H:, :], dW[:H, :] # index properly here.
    dcomb = (comb_weights @ dz.T).T # (N, H + D)
    dx, dprev_h = dcomb[:, H:], dcomb[:, :H]

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    - cache: Values needed in the backward pass
    """
    h, cache = None, None
    ##############################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
    # input data. You should use the rnn_step_forward function that you defined  #
    # above. You can use a for loop to help compute the forward pass.            #
    ##############################################################################
    

    h_state = h0.copy() # we keep updating this value, to add to h.

    N, T, D = x.shape
    _, H = h_state.shape
    h = np.empty((N, 0, H))
    state_caches = []

    for i in range(T): # each timestep
        x_slice = x[:,i,:].reshape(N, D)
        h_state = h_state.reshape(N, H) # This is done to properly run in step func.
        h_state, h_state_cache = rnn_step_forward(x_slice, h_state, Wx, Wh, b)
        h_state = h_state.reshape(N, 1, H) # This is done to add to h.
        state_caches.append(h_state_cache)
        h = np.concatenate([h, h_state], axis=1)

    cache = (state_caches, Wx, Wh, b) 
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H)

    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire      #
    # sequence of data. You should use the rnn_step_backward function that you   #
    # defined above. You can use a for loop to help compute the backward pass.   #
    ##############################################################################
    
    N, T, H = dh.shape
    dx = None

    state_caches, Wx, Wh, b = cache
    dWx, dWh, db = np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)

    dnext_h = np.zeros((N, H))
    for i in range(T-1, -1, -1): # each timestep, in reverse

        dh_total = dh[:, i, :] + dnext_h # We need both h_t and h_{t+1}
        dx_slice, dnext_h, dWx_t, dWh_t, db_t = rnn_step_backward(dh_total, state_caches[i])
        # As referenced in lecture slides, add the gradients rather than update.
        dWx += dWx_t
        dWh += dWh_t
        db += db_t.squeeze()
        D = dx_slice.shape[1]
        dx_slice = dx_slice.reshape(N, 1, D)
        if dx is None: dx = dx_slice
        else: dx = np.concatenate([dx_slice, dx], axis=1) # add in reverse
    
    dh0 = dnext_h.copy()

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
    """
    Forward pass for word embeddings. We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    to a vector of dimension D.

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x muxt be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    ##############################################################################
    # TODO: Implement the forward pass for word embeddings.                      #
    #                                                                            #
    # HINT: This can be done in one line using NumPy's array indexing.           #
    ##############################################################################

    out = W[x]
    cache = (x, W)

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return out, cache


def word_embedding_backward(dout, cache):
    """
    Backward pass for word embeddings. We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.

    HINT: Look up the function np.add.at

    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass

    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D).
    """
    dW = None
    ##############################################################################
    # TODO: Implement the backward pass for word embeddings.                     #
    #                                                                            #
    # Note that Words can appear more than once in a sequence.                   #
    # HINT: Look up the function np.add.at                                       #
    ##############################################################################
    
    # Inspo: makemore part 4 (becoming a backprop ninja)
    x, W = cache
    N, T = x.shape
    dW = np.zeros_like(W)
    for n in range(N):
        for t in range(T):
            ix = x[n,t]
            dout_slice = dout[n,t]
            np.add.at(dW, ix, dout_slice)

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dW


def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    next_h, next_c, cache = None, None, None
    #############################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.        #
    # You may want to use the numerically stable sigmoid implementation above.  #
    #############################################################################
    
    _, H = prev_h.shape
    prev_state = np.hstack((prev_h, x)).T # (H + D, N)
    W = np.vstack((Wh, Wx)).T # (4H, H + D)
    gates = (W @ prev_state).T + b # (N, 4H)
    i, f, o, g = gates[:, :H], gates[:, H:2*H], gates[:, 2*H:3*H], gates[:, 3*H:] # (N, H) for all

    next_c = prev_c * sigmoid(f) + (sigmoid(i) * np.tanh(g))
    next_h = np.tanh(next_c) * sigmoid(o)
    cache = (prev_state, prev_c, next_c, W, gates, i, f, o, g)

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.

    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh, dc, dWx, dWh, db = None, None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for a single timestep of an LSTM.       #
    #                                                                           #
    # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
    # the output value from the nonlinearity.                                   #
    #############################################################################
    
    prev_state, prev_c, next_c, W, gates, i, f, o, g = cache


    # For tanh: 1 - tanhx^2
    # For sigmoid: sigmoid(x) * (1 - sigmoid(x))

    # next_h = tanh(next_c) * sigmoid(o)
    # next_c = prev_c * sigmoid(f) + (sigmoid(i) * tanh(g))

    # do = dnext_h * (sigmoid(o) * (1 - sigmoid(o)))

    _, H = dnext_h.shape

    dsigmoid = lambda var: sigmoid(var) * (1 - sigmoid(var))
    dtanh = lambda var: 1 - np.tanh(var)**2
    
    dnext_c_total = dnext_c + dnext_h * sigmoid(o) * dtanh(next_c)
    di = dnext_c_total * np.tanh(g) * dsigmoid(i) # (N, H)
    df = dnext_c_total * prev_c * dsigmoid(f) # (N, H)
    do = dnext_h * np.tanh(next_c) * dsigmoid(o) # (N, H)
    dg = dnext_c_total * sigmoid(i) * dtanh(g) # (N, H)

    dgates = np.hstack((di, df, do, dg)) # (N, 4H)
    # prev_state = (H + D, N)
    # dW = (H + D, 4H)
    dW = (dgates.T @ prev_state.T).T # (H + D, 4H)
    dprev_state = (dgates @ W) # (N, H + D)
    db = dgates.T.sum(1)
    
    dWh, dWx = dW[:H, :], dW[H:, :] # (H, 4H), (D, 4H)
    dprev_h, dx = dprev_state[:, :H], dprev_state[:, H:] # (N, H), (N, D)
    dprev_c = dnext_c_total * sigmoid(f) # (N, H)

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the LSTM forward, we return the hidden states for all timesteps.

    Note that the initial cell state is passed as input, but the initial cell
    state is set to zero. Also note that the cell state is not returned; it is
    an internal variable to the LSTM and is not accessed from outside.

    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    - cache: Values needed for the backward pass.
    """
    h, cache = None, None
    #############################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
    # You should use the lstm_step_forward function that you just defined.      #
    #############################################################################
    
    h_state = h0.copy() # we keep updating this value, to add to h
    c_state = np.zeros_like(h_state)
    N, T, D = x.shape
    _, H = h_state.shape
    h = np.empty((N, 0, H))
    state_caches = []

    for i in range(T): # each timestep
        x_slice = x[:,i,:].reshape(N, D)
        h_state = h_state.reshape(N, H) # This is done to properly run in step func.
        h_state, c_state, h_state_cache = lstm_step_forward(x_slice, h_state, c_state, Wx, Wh, b)
        h_state = h_state.reshape(N, 1, H) # This is done to add to h.
        state_caches.append(h_state_cache)
        h = np.concatenate([h, h_state], axis=1)

    cache = (state_caches, Wx, Wh, b)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return h, cache


def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.]

    Inputs:
    - dh: Upstream gradients of hidden states, of shape (N, T, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data of shape (N, T, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
    # You should use the lstm_step_backward function that you just defined.     #
    #############################################################################
    
    N, T, H = dh.shape
    dx = None 

    state_caches, Wx, Wh, b = cache
    dWx, dWh, db = np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)

    dnext_h, dnext_c = np.zeros((N, H)), np.zeros((N, H))
    for i in range(T-1, -1, -1): # each timestep, in reverse

        dh_total = dh[:, i, :] + dnext_h # We need both h_t and h_{t+1}
        dx_slice, dnext_h, dnext_c, dWx_t, dWh_t, db_t = lstm_step_backward(dh_total, dnext_c, state_caches[i])
        # As referenced in lecture slides, add the gradients rather than update.
        dWx += dWx_t
        dWh += dWh_t
        db += db_t.squeeze()
        D = dx_slice.shape[1]
        dx_slice = dx_slice.reshape(N, 1, D)
        if dx is None: dx = dx_slice
        else: dx = np.concatenate([dx_slice, dx], axis=1) # add in reverse
    
    dh0 = dnext_h.copy()

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
    """
    Forward pass for a temporal affine layer. The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.

    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out
    return out, cache


def temporal_affine_backward(dout, cache):
    """
    Backward pass for temporal affine layer.

    Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass

    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    """
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, summing the loss over all timesteps and averaging across the
    minibatch.

    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional mask argument tells us
    which elements should contribute to the loss.

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.

    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    """

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]

    if verbose: print('dx_flat: ', dx_flat.shape)

    dx = dx_flat.reshape(N, T, V)

    return loss, dx
