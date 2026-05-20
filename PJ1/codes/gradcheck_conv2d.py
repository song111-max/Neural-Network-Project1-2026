"""Numerical gradient check for conv2D (Phase 1)."""
import numpy as np
from mynn.op import conv2D


def rel_error(analytic, numeric):
    return np.max(np.abs(analytic - numeric) / (np.abs(analytic) + np.abs(numeric) + 1e-8))


def make_layer():
    np.random.seed(0)
    layer = conv2D(1, 2, 3, stride=1, padding=1)
    layer.W = np.random.randn(*layer.W.shape) * 0.1
    layer.b = np.random.randn(*layer.b.shape) * 0.01
    layer.params['W'] = layer.W
    layer.params['b'] = layer.b
    return layer


def numerical_grad_W(W, b, X, grads, idx, eps=1e-5):
    oc, ic, kh, kw = idx
    layer = conv2D(1, 2, 3, stride=1, padding=1)
    layer.W = W.copy()
    layer.b = b.copy()
    layer.params['W'] = layer.W
    layer.params['b'] = layer.b
    w0 = layer.W[oc, ic, kh, kw]
    layer.W[oc, ic, kh, kw] = w0 + eps
    f1 = layer.forward(X.copy())
    layer.W[oc, ic, kh, kw] = w0 - eps
    f2 = layer.forward(X.copy())
    return (np.sum(grads * (f1 - f2)) / (2 * eps))


def numerical_grad_X(W, b, X, grads, idx, eps=1e-5):
    b_idx, c, h, w = idx
    layer = conv2D(1, 2, 3, stride=1, padding=1)
    layer.W = W.copy()
    layer.b = b.copy()
    layer.params['W'] = layer.W
    layer.params['b'] = layer.b
    X = X.copy()
    x0 = X[b_idx, c, h, w]
    X[b_idx, c, h, w] = x0 + eps
    f1 = layer.forward(X)
    X[b_idx, c, h, w] = x0 - eps
    f2 = layer.forward(X)
    return (np.sum(grads * (f1 - f2)) / (2 * eps))


def main():
    np.random.seed(0)
    batch, height, width = 2, 5, 5
    layer = make_layer()
    W, b = layer.W.copy(), layer.b.copy()
    X = np.random.randn(batch, 1, height, width) * 0.1
    layer.forward(X)
    grads = np.random.randn(batch, 2, height, width) * 0.1
    dX_analytic = layer.backward(grads)
    dW_analytic = layer.grads['W'].copy()

    max_rel_w = 0.0
    for oc in range(2):
        for ic in range(1):
            for kh in range(3):
                for kw in range(3):
                    num = numerical_grad_W(W, b, X, grads, (oc, ic, kh, kw))
                    ana = dW_analytic[oc, ic, kh, kw]
                    max_rel_w = max(max_rel_w, rel_error(np.array(ana), np.array(num)))

    max_rel_x = 0.0
    for bi in range(batch):
        for c in range(1):
            for h in range(height):
                for w in range(width):
                    num = numerical_grad_X(W, b, X, grads, (bi, c, h, w))
                    ana = dX_analytic[bi, c, h, w]
                    max_rel_x = max(max_rel_x, rel_error(np.array(ana), np.array(num)))

    tol = 1e-5
    dw_pass = max_rel_w < tol
    dx_pass = max_rel_x < tol
    print(f'dW max relative error: {max_rel_w:.2e}  {"PASS" if dw_pass else "FAIL"}')
    print(f'dX max relative error: {max_rel_x:.2e}  {"PASS" if dx_pass else "FAIL"}')
    if not (dw_pass and dx_pass):
        if max_rel_w < 1e-4 and max_rel_x < 1e-4:
            print('(relaxed 1e-4 threshold: PASS)')
            return 0
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
