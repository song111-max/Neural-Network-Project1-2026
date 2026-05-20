# Part B: train CNN on MNIST (same data split as Part A for fair comparison)
import mynn as nn
from draw_tools.plot import plot

import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle
import os

np.random.seed(309)

train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28 * 28)

with gzip.open(train_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    train_labs = np.frombuffer(f.read(), dtype=np.uint8)

if os.path.exists('idx.pickle'):
    with open('idx.pickle', 'rb') as f:
        idx = pickle.load(f)
else:
    idx = np.random.permutation(np.arange(num))
    with open('idx.pickle', 'wb') as f:
        pickle.dump(idx, f)

train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

train_imgs = train_imgs / train_imgs.max()
valid_imgs = valid_imgs / valid_imgs.max()

cnn_model = nn.models.Model_CNN(lambda_conv=1e-4, lambda_fc=1e-4)
optimizer = nn.optimizer.SGD(init_lr=0.06, model=cnn_model)
scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[800, 2400, 4000], gamma=0.5)
loss_fn = nn.op.MultiCrossEntropyLoss(model=cnn_model, max_classes=train_labs.max() + 1)

runner = nn.runner.RunnerM(cnn_model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler)

runner.train(
    [train_imgs, train_labs],
    [valid_imgs, valid_labs],
    num_epochs=5,
    log_iters=100,
    save_dir=r'./best_models_cnn',
    eval_each_iter=False,  # CNN 在 NumPy 上较慢；每 epoch 末评估一次，与 MLP 公平对比指标
)

os.makedirs('./results', exist_ok=True)

summary_path = r'./results/part_b_summary.txt'
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write('Part B CNN — training summary\n')
    f.write('=' * 50 + '\n')
    f.write('Model: Conv(1->8, 3x3, pad=1) -> ReLU -> MaxPool(2x2)\n')
    f.write('       -> Flatten -> Linear(1568->128) -> ReLU -> Linear(128->10)\n')
    f.write('Epochs: 5, batch_size: 32, init_lr: 0.06\n')
    f.write(f'Best validation accuracy: {runner.best_score:.4f}\n')
    f.write(f'Final train loss: {runner.train_loss[-1]:.4f}\n')
    f.write(f'Final train accuracy: {runner.train_scores[-1]:.4f}\n')
    f.write(f'Final dev loss: {runner.dev_loss[-1]:.4f}\n')
    f.write(f'Final dev accuracy: {runner.dev_scores[-1]:.4f}\n')
print(f'Summary saved to: {summary_path}')
print(f'Best validation accuracy: {runner.best_score:.4f}')

_, axes = plt.subplots(1, 2)
axes.reshape(-1)
_.set_tight_layout(1)
plot(runner, axes)
curve_path = r'./results/part_b_learning_curve.png'
plt.savefig(curve_path, dpi=150, bbox_inches='tight')
print(f'Learning curve saved to: {curve_path}')

# plt.show()  # disabled for unattended training
