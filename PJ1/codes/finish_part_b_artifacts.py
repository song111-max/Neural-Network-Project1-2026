"""Generate Part B summary and learning curve from training_part_b.log (no retrain)."""
import os
import re

import matplotlib.pyplot as plt

LOG = 'training_part_b.log'
BEST_VAL = 0.9790
BATCHES_PER_EPOCH = 1563  # floor(50000/32)+1


def _open_log(path):
    for enc in ('utf-16', 'utf-16-le', 'utf-8'):
        try:
            with open(path, encoding=enc) as f:
                return f.readlines()
        except (UnicodeDecodeError, UnicodeError):
            continue
    with open(path, encoding='utf-8', errors='replace') as f:
        return f.readlines()


def parse_log(path):
    train_iters, train_loss, train_score = [], [], []
    dev_iters, dev_loss, dev_score = [], [], []
    iter_re = re.compile(r'epoch:\s*(\d+),\s*iteration:\s*(\d+)')
    train_re = re.compile(r'\[Train\]\s*loss:\s*([\d.eE+-]+),\s*score:\s*([\d.eE+-]+)')
    dev_re = re.compile(
        r'epoch:\s*(\d+)\s*finished.*?'
        r'\[Dev\]\s*loss:\s*([\d.eE+-]+),\s*score:\s*([\d.eE+-]+)'
    )

    global_iter = 0
    for line in _open_log(path):
        m = iter_re.search(line)
        if m:
            ep, it = int(m.group(1)), int(m.group(2))
            global_iter = ep * BATCHES_PER_EPOCH + it
        m = train_re.search(line)
        if m:
            train_iters.append(global_iter)
            train_loss.append(float(m.group(1)))
            train_score.append(float(m.group(2)))
        m = dev_re.search(line)
        if m:
            ep = int(m.group(1))
            dev_iters.append((ep + 1) * BATCHES_PER_EPOCH - 1)
            dev_loss.append(float(m.group(2)))
            dev_score.append(float(m.group(3)))

    return train_iters, train_loss, train_score, dev_iters, dev_loss, dev_score


def plot_curve(train_iters, train_loss, train_score, dev_iters, dev_loss, dev_score, out_path):
    colors = ('#E3E37D', '#968A62')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(train_iters, train_loss, color=colors[0], label='Train loss')
    axes[0].plot(dev_iters, dev_loss, color=colors[1], linestyle='--', label='Dev loss')
    axes[0].set_xlabel('iteration')
    axes[0].set_ylabel('loss')
    axes[0].legend(loc='upper right')

    axes[1].plot(train_iters, train_score, color=colors[0], label='Train accuracy')
    axes[1].plot(dev_iters, dev_score, color=colors[1], linestyle='--', label='Dev accuracy')
    axes[1].set_xlabel('iteration')
    axes[1].set_ylabel('score')
    axes[1].legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    train_iters, tr_loss, tr_score, dev_iters, dev_loss, dev_score = parse_log(LOG)
    if not tr_loss:
        raise SystemExit(f'No training metrics parsed from {LOG}')

    os.makedirs('results', exist_ok=True)
    curve_path = r'./results/part_b_learning_curve.png'
    plot_curve(train_iters, tr_loss, tr_score, dev_iters, dev_loss, dev_score, curve_path)
    print(f'Learning curve saved to: {curve_path} ({len(tr_loss)} train points, {len(dev_score)} dev points)')

    summary_path = r'./results/part_b_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('Part B CNN — training summary\n')
        f.write('=' * 50 + '\n')
        f.write('Model: Conv(1->8, 3x3, pad=1) -> ReLU -> MaxPool(2x2)\n')
        f.write('       -> Flatten -> Linear(1568->128) -> ReLU -> Linear(128->10)\n')
        f.write('Epochs: 5, batch_size: 32, init_lr: 0.06\n')
        f.write(f'Best validation accuracy: {BEST_VAL:.4f}\n')
        f.write(f'Final train loss: {tr_loss[-1]:.4f}\n')
        f.write(f'Final train accuracy: {tr_score[-1]:.4f}\n')
        f.write(f'Final dev loss: {dev_loss[-1]:.4f}\n')
        f.write(f'Final dev accuracy: {dev_score[-1]:.4f}\n')
    print(f'Summary saved to: {summary_path}')


if __name__ == '__main__':
    main()
