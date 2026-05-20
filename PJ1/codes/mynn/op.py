from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(size=(in_dim, out_dim))
        self.b = initialize_method(size=(1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        return X @ self.W + self.b

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        self.grads['W'] = self.input.T @ grad
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True)
        return grad @ self.W.T
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

def _im2col(X, kernel_size, stride, padding):
    """Unfold [B,C,H,W] to [B*out_h*out_w, C*k_h*k_w]."""
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)

    batch_size, channels, height, width = X.shape
    k_h, k_w = kernel_size
    stride_h, stride_w = stride
    pad_h, pad_w = padding

    if pad_h > 0 or pad_w > 0:
        X = np.pad(X, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant')

    height_padded, width_padded = X.shape[2], X.shape[3]
    out_h = (height_padded - k_h) // stride_h + 1
    out_w = (width_padded - k_w) // stride_w + 1

    col = np.zeros((batch_size * out_h * out_w, channels * k_h * k_w))
    col_idx = 0
    for i in range(out_h):
        h_start = i * stride_h
        h_end = h_start + k_h
        for j in range(out_w):
            w_start = j * stride_w
            w_end = w_start + k_w
            patch = X[:, :, h_start:h_end, w_start:w_end]
            col[col_idx:col_idx + batch_size] = patch.reshape(batch_size, -1)
            col_idx += batch_size
    return col, out_h, out_w, X


def _col2im(col, X_shape, kernel_size, stride, padding, out_h, out_w):
    """Map column gradient back to [B,C,H,W]."""
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)

    batch_size, channels, height, width = X_shape
    k_h, k_w = kernel_size
    stride_h, stride_w = stride
    pad_h, pad_w = padding

    height_padded = height + 2 * pad_h
    width_padded = width + 2 * pad_w
    dX_padded = np.zeros((batch_size, channels, height_padded, width_padded))

    col_idx = 0
    for i in range(out_h):
        h_start = i * stride_h
        h_end = h_start + k_h
        for j in range(out_w):
            w_start = j * stride_w
            w_end = w_start + k_w
            patch_grad = col[col_idx:col_idx + batch_size].reshape(batch_size, channels, k_h, k_w)
            dX_padded[:, :, h_start:h_end, w_start:w_end] += patch_grad
            col_idx += batch_size

    if pad_h > 0 or pad_w > 0:
        return dX_padded[:, :, pad_h:height_padded - pad_h, pad_w:width_padded - pad_w]
    return dX_padded


class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=None, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        k_h, k_w = kernel_size
        fan_in = in_channels * k_h * k_w

        def he_init(size):
            return np.random.randn(*size) * np.sqrt(2.0 / fan_in)

        if initialize_method is None:
            initialize_method = he_init

        self.W = initialize_method((out_channels, in_channels, k_h, k_w))
        self.b = np.zeros((1, out_channels, 1, 1))
        self.grads = {'W': None, 'b': None}
        self.params = {'W': self.W, 'b': self.b}
        self.input = None
        self.X_padded = None

        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [out_channels, in_channels, k_h, k_w]
        """
        self.input = X
        batch_size = X.shape[0]
        col, out_h, out_w, X_padded = _im2col(X, self.kernel_size, self.stride, self.padding)
        self.col = col
        self.out_h = out_h
        self.out_w = out_w
        self.X_padded = X_padded

        W_mat = self.W.reshape(self.out_channels, -1)
        out = col @ W_mat.T + self.b.reshape(1, -1)
        # im2col rows are ordered (out_h, out_w, batch), not (batch, out_h, out_w)
        return out.reshape(out_h, out_w, batch_size, self.out_channels).transpose(2, 3, 0, 1)

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        batch_size, _, out_h, out_w = grads.shape
        grad_col = np.zeros((batch_size * out_h * out_w, self.out_channels))
        col_idx = 0
        for i in range(out_h):
            for j in range(out_w):
                grad_col[col_idx:col_idx + batch_size] = grads[:, :, i, j]
                col_idx += batch_size
        W_mat = self.W.reshape(self.out_channels, -1)

        self.grads['W'] = (grad_col.T @ self.col).reshape(self.W.shape)
        self.grads['b'] = np.sum(grads, axis=(0, 2, 3), keepdims=True)

        dcol = grad_col @ W_mat
        dX = _col2im(dcol, self.input.shape, self.kernel_size, self.stride, self.padding, self.out_h, self.out_w)
        return dX

    def clear_grad(self):
        self.grads = {'W': None, 'b': None}


class MaxPool2D(Layer):
    """2D max pooling layer."""

    def __init__(self, kernel_size=2, stride=2) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        self.kernel_size = kernel_size
        self.stride = stride
        self.input = None
        self.argmax_idx = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        batch_size, channels, height, width = X.shape
        k_h, k_w = self.kernel_size
        stride_h, stride_w = self.stride
        out_h = (height - k_h) // stride_h + 1
        out_w = (width - k_w) // stride_w + 1
        output = np.zeros((batch_size, channels, out_h, out_w))
        argmax_idx = np.zeros((batch_size, channels, out_h, out_w), dtype=np.int32)

        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_h):
                    h_start = i * stride_h
                    for j in range(out_w):
                        w_start = j * stride_w
                        window = X[b, c, h_start:h_start + k_h, w_start:w_start + k_w]
                        flat = window.reshape(-1)
                        idx = int(np.argmax(flat))
                        output[b, c, i, j] = flat[idx]
                        argmax_idx[b, c, i, j] = idx

        self.argmax_idx = argmax_idx
        return output

    def backward(self, grads):
        dX = np.zeros_like(self.input)
        batch_size, channels, _, _ = grads.shape
        k_h, k_w = self.kernel_size
        stride_h, stride_w = self.stride

        for b in range(batch_size):
            for c in range(channels):
                for i in range(grads.shape[2]):
                    h_start = i * stride_h
                    for j in range(grads.shape[3]):
                        w_start = j * stride_w
                        idx = self.argmax_idx[b, c, i, j]
                        hi, wi = divmod(idx, k_w)
                        dX[b, c, h_start + hi, w_start + wi] += grads[b, c, i, j]
        return dX
        
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True
        self.predicts = None
        self.labels = None
        self.grads = None

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        self.predicts = predicts
        self.labels = labels
        if self.has_softmax:
            prob = softmax(predicts)
        else:
            prob = predicts
        batch_size = predicts.shape[0]
        correct_prob = prob[np.arange(batch_size), labels]
        loss = -np.mean(np.log(correct_prob + 1e-12))
        return loss
    
    def backward(self):
        # first compute the grads from the loss to the input
        batch_size = self.predicts.shape[0]
        if self.has_softmax:
            prob = softmax(self.predicts)
            self.grads = prob.copy()
            self.grads[np.arange(batch_size), self.labels] -= 1
            self.grads /= batch_size
        else:
            self.grads = -np.eye(self.max_classes)[self.labels] / batch_size
            self.grads /= (self.predicts + 1e-12)
        # Then send the grads to model for back propagation
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition