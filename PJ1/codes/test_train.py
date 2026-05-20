# An example of read in the data and train the model. The runner is implemented, while the model used for training need your implementation.
import mynn as nn
from draw_tools.plot import plot

import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle
import os

# fixed seed for experiment
np.random.seed(309)

train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        train_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(train_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)


# choose 10000 samples from train set as validation set.
idx = np.random.permutation(np.arange(num))
# save the index.
with open('idx.pickle', 'wb') as f:
        pickle.dump(idx, f)
train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

# normalize from [0, 255] to [0, 1]
train_imgs = train_imgs / train_imgs.max()
valid_imgs = valid_imgs / valid_imgs.max()

linear_model = nn.models.Model_MLP([train_imgs.shape[-1], 600, 10], 'ReLU', [1e-4, 1e-4])
optimizer = nn.optimizer.SGD(init_lr=0.06, model=linear_model)
scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[800, 2400, 4000], gamma=0.5)
loss_fn = nn.op.MultiCrossEntropyLoss(model=linear_model, max_classes=train_labs.max()+1)

runner = nn.runner.RunnerM(linear_model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler)

runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=5, log_iters=100, save_dir=r'./best_models')

os.makedirs('./results', exist_ok=True)

_, axes = plt.subplots(1, 2)
axes.reshape(-1)
_.set_tight_layout(1)
plot(runner, axes)
curve_path = r'./results/part_a_learning_curve.png'
plt.savefig(curve_path, dpi=150, bbox_inches='tight')
print(f'Learning curve saved to: {curve_path}')

summary_path = r'./results/part_a_summary.txt'
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write('Part A MLP Baseline — training summary\n')
    f.write('=' * 50 + '\n')
    f.write(f'Model: 784 -> 600 (ReLU) -> 10\n')
    f.write(f'Epochs: 5, batch_size: 32, init_lr: 0.06\n')
    f.write(f'Best validation accuracy: {runner.best_score:.4f}\n')
    f.write(f'Final train loss: {runner.train_loss[-1]:.4f}\n')
    f.write(f'Final train accuracy: {runner.train_scores[-1]:.4f}\n')
    f.write(f'Final dev loss: {runner.dev_loss[-1]:.4f}\n')
    f.write(f'Final dev accuracy: {runner.dev_scores[-1]:.4f}\n')
print(f'Summary saved to: {summary_path}')
print(f'Best validation accuracy: {runner.best_score:.4f}')

plt.show()