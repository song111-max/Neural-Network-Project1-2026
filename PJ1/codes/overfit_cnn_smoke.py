"""Phase 3: overfit 32 MNIST samples with CNN."""
import gzip
import pickle
from struct import unpack

import numpy as np

import mynn as nn

np.random.seed(309)

with gzip.open(r'.\dataset\MNIST\train-images-idx3-ubyte.gz', 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28 * 28)

with gzip.open(r'.\dataset\MNIST\train-labels-idx1-ubyte.gz', 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    train_labs = np.frombuffer(f.read(), dtype=np.uint8)

with open('idx.pickle', 'rb') as f:
    idx = pickle.load(f)

train_imgs = train_imgs[idx][10000:]
train_labs = train_labs[idx][10000:]
train_imgs = train_imgs / train_imgs.max()

subset_idx = np.arange(32)
X = train_imgs[subset_idx]
y = train_labs[subset_idx]

model = nn.models.Model_CNN(lambda_conv=1e-4, lambda_fc=1e-4)
optimizer = nn.optimizer.SGD(init_lr=0.06, model=model)
loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=10)

steps = 300
for step in range(steps):
    logits = model(X)
    loss = loss_fn(logits, y)
    acc = nn.metric.accuracy(logits, y)
    loss_fn.backward()
    optimizer.step()
    if step % 50 == 0 or step == steps - 1:
        print(f'step {step}: loss={loss:.4f}, acc={acc:.4f}')

if acc < 0.95:
    raise SystemExit(f'FAIL: final acc {acc:.4f} < 0.95')
print('PASS: overfit smoke test')
