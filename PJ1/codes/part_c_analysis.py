"""Part C error analysis on Part B CNN (SGD) checkpoint."""
import os
from struct import unpack
import gzip

import matplotlib.pyplot as plt
import numpy as np

import mynn as nn

np.random.seed(309)

CHECKPOINT = r'.\best_models_cnn\best_model.pickle'
OUT_DIR = 'results/part_c'
NUM_CLASSES = 10
NUM_MISCLASS = 16

os.makedirs(OUT_DIR, exist_ok=True)

model = nn.models.Model_CNN(build=False)
model.load_model(CHECKPOINT)

test_images_path = r'.\dataset\MNIST\t10k-images-idx3-ubyte.gz'
test_labels_path = r'.\dataset\MNIST\t10k-labels-idx1-ubyte.gz'

with gzip.open(test_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28 * 28)

with gzip.open(test_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    test_labs = np.frombuffer(f.read(), dtype=np.uint8)

test_imgs = test_imgs / test_imgs.max()

logits = model(test_imgs)
preds = np.argmax(logits, axis=1)
test_acc = nn.metric.accuracy(logits, test_labs)

cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
for t, p in zip(test_labs, preds):
    cm[int(t), int(p)] += 1

row_sums = cm.sum(axis=1, keepdims=True)
cm_norm = np.divide(cm.astype(float), row_sums, where=row_sums != 0, out=np.zeros_like(cm, dtype=float))

def plot_confusion(mat, path, title, fmt='d', vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(mat, cmap='Blues', vmin=vmin, vmax=vmax)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            val = mat[i, j]
            if fmt == 'd':
                text = str(int(val))
            else:
                text = f'{val:.2f}' if row_sums[i, 0] > 0 else 'N/A'
            ax.text(j, i, text, ha='center', va='center', fontsize=7,
                    color='white' if mat[i, j] > mat.max() * 0.5 else 'black')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


plot_confusion(cm, os.path.join(OUT_DIR, 'part_c_confusion_matrix.png'),
               'Confusion Matrix (counts)', fmt='d')
plot_confusion(cm_norm, os.path.join(OUT_DIR, 'part_c_confusion_matrix_norm.png'),
               'Confusion Matrix (row-normalized)', fmt='.2f', vmin=0, vmax=1)

# Top confusion pairs (off-diagonal)
off_diag = []
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        if i != j and cm[i, j] > 0:
            off_diag.append((cm[i, j], i, j))
off_diag.sort(reverse=True)
top_pairs = [(i, j, c) for c, i, j in off_diag[:10]]

mis_idx = np.where(preds != test_labs)[0]
pair_buckets = { (i, j): [] for _, i, j in top_pairs }
other_mis = []
for idx in mis_idx:
    t, p = int(test_labs[idx]), int(preds[idx])
    if (t, p) in pair_buckets:
        pair_buckets[(t, p)].append(idx)
    else:
        other_mis.append(idx)

np.random.seed(309)
selected = []
for _, true_c, pred_c in top_pairs:
    bucket = pair_buckets[(true_c, pred_c)]
    if bucket:
        pick = np.random.choice(bucket)
        selected.append(pick)
    if len(selected) >= NUM_MISCLASS:
        break
if len(selected) < NUM_MISCLASS and other_mis:
    need = NUM_MISCLASS - len(selected)
    extra = np.random.choice(other_mis, size=min(need, len(other_mis)), replace=False)
    selected.extend(extra.tolist())
selected = selected[:NUM_MISCLASS]

fig, axes = plt.subplots(4, 4, figsize=(10, 10))
axes = axes.reshape(-1)
for k, idx in enumerate(selected):
    img = test_imgs[idx].reshape(28, 28)
    t, p = int(test_labs[idx]), int(preds[idx])
    axes[k].imshow(img, cmap='gray')
    axes[k].set_title(f'true={t} pred={p}', fontsize=9)
    axes[k].axis('off')
for k in range(len(selected), NUM_MISCLASS):
    axes[k].axis('off')
plt.suptitle('Misclassified (Top confusion pairs prioritized)')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'part_c_misclassified.png'), dpi=150, bbox_inches='tight')
plt.close()

# Conv kernels W[i, 0, :, :]
W = model.layers[0].params['W']
fig, axes = plt.subplots(2, 4, figsize=(8, 4))
axes = axes.reshape(-1)
for i in range(8):
    axes[i].imshow(W[i, 0, :, :], cmap='viridis')
    axes[i].set_title(f'filter {i}')
    axes[i].axis('off')
plt.suptitle('Conv kernels W[i, 0, :, :]')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'part_c_conv_kernels.png'), dpi=150, bbox_inches='tight')
plt.close()

# Per-class precision / recall
precisions = []
recalls = []
for c in range(NUM_CLASSES):
    col_sum = cm[:, c].sum()
    row_sum = cm[c, :].sum()
    tp = cm[c, c]
    if col_sum == 0:
        precisions.append('N/A')
    else:
        precisions.append(tp / col_sum)
    if row_sum == 0:
        recalls.append('N/A')
    else:
        recalls.append(tp / row_sum)

recall_numeric = []
for c in range(NUM_CLASSES):
    if isinstance(recalls[c], float):
        recall_numeric.append((recalls[c], c))
worst_recall_class = min(recall_numeric, key=lambda x: x[0])[1] if recall_numeric else -1

n_mis = int((preds != test_labs).sum())
n_total = len(test_labs)
mis_rate = n_mis / n_total

txt_path = os.path.join(OUT_DIR, 'part_c_error_analysis.txt')
with open(txt_path, 'w', encoding='utf-8') as f:
    f.write(f'Model: Part B CNN (SGD), checkpoint={CHECKPOINT}, test acc={test_acc:.4f}\n')
    f.write(f'Total test samples: {n_total}\n')
    f.write(f'Misclassified: {n_mis} ({mis_rate:.4%})\n\n')
    f.write('Per-class precision / recall:\n')
    for c in range(NUM_CLASSES):
        p = f'{precisions[c]:.4f}' if isinstance(precisions[c], float) else precisions[c]
        r = f'{recalls[c]:.4f}' if isinstance(recalls[c], float) else recalls[c]
        f.write(f'  class {c}: precision={p}, recall={r}\n')
    f.write('\nTop confusion pairs (true -> pred, count):\n')
    for true_c, pred_c, count in top_pairs[:5]:
        f.write(f'  {true_c} -> {pred_c}: {count}\n')
    f.write(f'\nWorst recall class: {worst_recall_class}\n')

print(f'Test accuracy: {test_acc:.4f}')
print(f'Error analysis saved to {OUT_DIR}/')
