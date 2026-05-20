from .op import *
import numpy as np
import pickle

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        self.size_list = size_list
        self.act_func = act_func

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                fan_in = size_list[i]

                def he_init(size, fan_in=fan_in):
                    return np.random.randn(*size) * np.sqrt(2.0 / fan_in)

                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1], initialize_method=he_init)
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        for i in range(len(self.size_list) - 1):
            self.layers = []
            for i in range(len(self.size_list) - 1):
                layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
                layer.W = param_list[i + 2]['W']
                layer.b = param_list[i + 2]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = param_list[i + 2]['weight_decay']
                layer.weight_decay_lambda = param_list[i+2]['lambda']
                if self.act_func == 'Logistic':
                    raise NotImplemented
                elif self.act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(self.size_list) - 2:
                    self.layers.append(layer_f)
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
        

class Model_CNN(Layer):
    """
    A simple CNN for MNIST: Conv(1->8, 3x3, pad=1) -> ReLU -> MaxPool(2x2)
    -> Flatten -> Linear(1568->128) -> ReLU -> Linear(128->10)
    """
    def __init__(self, lambda_conv=1e-4, lambda_fc=1e-4, build=True):
        self.input_shape = (1, 28, 28)
        self.flat_dim = 8 * 14 * 14
        self.lambda_conv = lambda_conv
        self.lambda_fc = lambda_fc
        self.layers = []
        if build:
            self._build_layers()

    def _build_layers(self):
        fan_conv = 1 * 3 * 3

        def he_conv(size, fan_in=fan_conv):
            return np.random.randn(*size) * np.sqrt(2.0 / fan_in)

        conv = conv2D(1, 8, 3, stride=1, padding=1, initialize_method=he_conv)
        conv.weight_decay = True
        conv.weight_decay_lambda = self.lambda_conv

        fan_fc1 = self.flat_dim

        def he_fc1(size, fan_in=fan_fc1):
            return np.random.randn(*size) * np.sqrt(2.0 / fan_in)

        lin1 = Linear(self.flat_dim, 128, initialize_method=he_fc1)
        lin1.weight_decay = True
        lin1.weight_decay_lambda = self.lambda_fc

        fan_fc2 = 128

        def he_fc2(size, fan_in=fan_fc2):
            return np.random.randn(*size) * np.sqrt(2.0 / fan_in)

        lin2 = Linear(128, 10, initialize_method=he_fc2)

        self.layers = [conv, ReLU(), MaxPool2D(2, 2), lin1, ReLU(), lin2]
        self._flatten_shape = None

    def __call__(self, X):
        return self.forward(X)

    def _to_nchw(self, X):
        if X.ndim == 2:
            return X.reshape(X.shape[0], 1, 28, 28)
        return X

    def forward(self, X):
        out = self._to_nchw(X)
        for layer in self.layers[:3]:
            out = layer(out)
        self._flatten_shape = out.shape
        out = out.reshape(out.shape[0], -1)
        for layer in self.layers[3:]:
            out = layer(out)
        return out

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers[3:]):
            grads = layer.backward(grads)
        grads = grads.reshape(self._flatten_shape)
        for layer in reversed(self.layers[:3]):
            grads = layer.backward(grads)
        return grads

    def load_model(self, save_path):
        with open(save_path, 'rb') as f:
            param_list = pickle.load(f)
        self.lambda_conv = param_list[0]
        self.lambda_fc = param_list[1]
        self.flat_dim = param_list[2]
        self._build_layers()
        offset = 3
        for layer in self.layers:
            if layer.optimizable:
                layer.W = param_list[offset]['W']
                layer.b = param_list[offset]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = param_list[offset]['weight_decay']
                layer.weight_decay_lambda = param_list[offset]['lambda']
                offset += 1

    def save_model(self, save_path):
        param_list = [self.lambda_conv, self.lambda_fc, self.flat_dim]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({
                    'W': layer.params['W'],
                    'b': layer.params['b'],
                    'weight_decay': layer.weight_decay,
                    'lambda': layer.weight_decay_lambda,
                })
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)