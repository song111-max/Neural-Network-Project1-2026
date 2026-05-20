# Part C: train CNN with MomentGD (same setup as Part B except optimizer)
import argparse
import os
import pickle
from struct import unpack
import gzip

import matplotlib.pyplot as plt
import numpy as np

import mynn as nn
from draw_tools.plot import plot

np.random.seed(309)

parser = argparse.ArgumentParser()
parser.add_argument('--smoke', action='store_true', help='1 epoch smoke test only')
args = parser.parse_args()

NUM_EPOCHS = 1 if args.smoke else 5
save_dir = r'./best_models_cnn_momentum_smoke' if args.smoke else r'./best_models_cnn_momentum'
os.makedirs(save_dir, exist_ok=True)

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
optimizer = nn.optimizer.MomentGD(init_lr=0.06, model=cnn_model, mu=0.9)
scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[800, 2400, 4000], gamma=0.5)
loss_fn = nn.op.MultiCrossEntropyLoss(model=cnn_model, max_classes=train_labs.max() + 1)

runner = nn.runner.RunnerM(cnn_model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler)

print(f'Training MomentGD: epochs={NUM_EPOCHS}, save_dir={save_dir}')
runner.train(
    [train_imgs, train_labs],
    [valid_imgs, valid_labs],
    num_epochs=NUM_EPOCHS,
    log_iters=100,
    save_dir=save_dir,
    eval_each_iter=False,
)

if args.smoke:
    smoke_path = os.path.join(save_dir, 'best_model.pickle')
    if os.path.exists(smoke_path):
        print(f'SMOKE PASS: checkpoint saved to {smoke_path}')
    else:
        raise SystemExit(f'SMOKE FAIL: no checkpoint at {smoke_path}')
    raise SystemExit(0)

os.makedirs('./results', exist_ok=True)

dev = np.array(runner.dev_scores)
best_val_epoch = int(np.argmax(dev))
best_val = float(dev[best_val_epoch])

def first_epoch_ge(threshold):
    for i, s in enumerate(dev):
        if s >= threshold:
            return i
    return 'none'

summary_path = r'./results/part_c_momentum_summary.txt'
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write('Part C CNN — MomentGD training summary\n')
    f.write('=' * 50 + '\n')
    f.write('Optimizer: MomentGD, mu=0.9, init_lr=0.06\n')
    f.write('Model: same as Part B CNN\n')
    f.write(f'epoch_0_dev: {dev[0]:.4f}\n')
    f.write(f'epoch_1_dev: {dev[1]:.4f}\n')
    f.write(f'epoch_2_dev: {dev[2]:.4f}\n')
    f.write(f'epoch_3_dev: {dev[3]:.4f}\n')
    f.write(f'epoch_4_dev: {dev[4]:.4f}\n')
    f.write(f'best_val: {best_val:.4f}\n')
    f.write(f'best_val_epoch: {best_val_epoch}\n')
    f.write(f'first_epoch_dev_ge_0.97: {first_epoch_ge(0.97)}\n')
    f.write(f'first_epoch_dev_ge_0.975: {first_epoch_ge(0.975)}\n')
    f.write(f'best_val_first_2_epochs: {float(dev[:2].max()):.4f}\n')
    f.write(f'Final train loss: {runner.train_loss[-1]:.4f}\n')
    f.write(f'Final train accuracy: {runner.train_scores[-1]:.4f}\n')
    f.write(f'Final dev loss: {runner.dev_loss[-1]:.4f}\n')
    f.write(f'Final dev accuracy: {runner.dev_scores[-1]:.4f}\n')
    f.write('\n')
    f.write('baseline_source: training_part_b.log epoch-end dev (SGD)\n')
    f.write('baseline_epoch_0_dev: 0.9655\n')
    f.write('baseline_epoch_1_dev: 0.9757\n')
    f.write('baseline_epoch_2_dev: 0.9783\n')
    f.write('baseline_epoch_3_dev: 0.9784\n')
    f.write('baseline_epoch_4_dev: 0.9790\n')
    f.write('baseline_best_val: 0.9790\n')

print(f'Summary saved to: {summary_path}')
print(f'Best validation accuracy: {best_val:.4f} (epoch {best_val_epoch})')

_, axes = plt.subplots(1, 2)
axes.reshape(-1)
_.set_tight_layout(1)
plot(runner, axes)
curve_path = r'./results/part_c_momentum_learning_curve.png'
plt.savefig(curve_path, dpi=150, bbox_inches='tight')
print(f'Learning curve saved to: {curve_path}')
