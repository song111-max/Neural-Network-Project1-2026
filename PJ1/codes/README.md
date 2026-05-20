# PJ1/codes 

**完整说明：** [../README.md](../README.md) · 仓库根 [../../README.md](../../README.md)

---

## 权重目录

| 用途 | 路径 | 说明 |
|------|------|------|
| Part A MLP | `best_models/best_model.pickle` | `test_train.py` / `test_model.py` |
| Part B CNN-SGD | `best_models_cnn/best_model.pickle` | `test_train_cnn.py` / `test_model_cnn.py` |
| Part C CNN-MomentGD | `best_models_cnn_momentum/best_model.pickle` | `test_train_part_c_momentum.py` / `test_model_part_c_momentum.py` |

## 环境与依赖

```bash
# 在 PJ1/codes 下
python -m venv .venv
.\.venv\Scripts\activate          # Windows
pip install -r ../../requirements.txt
```

| 文件 | 说明 |
|------|------|
| `../../requirements.txt` | 宽松下限 `>=` |
| `requirements-freeze.txt` | `pip freeze` 锁定 `==`，复现见文件头注释 |
| `../../requirements-freeze.txt` | 与上相同（仓库根副本） |

锁定环境示例：

```bash
python -m venv .venv-freeze
.\.venv-freeze\Scripts\pip install -r requirements-freeze.txt
```

`.venv-freeze/` 已在 `.gitignore` 中忽略。

---

## 日志

- 指标：**`results/*.txt`**  
- Part B 长日志：**`training_part_b.log`**  
- 可选终端副本：**`log/`**（见 `log/README.md`）

---

## 清单

1. `mynn/op.py` — Linear、MultiCrossEntropyLoss、conv2D 等  
2. `mynn/models.py` — 模型结构  
3. `mynn/lr_scheduler.py` — 学习率调度  
4. `mynn/optimizer.py` — `MomentGD`  
5. `mynn/runner.py` — 按需调整  

### 训练 / 测试入口

| Part | 训练 | 测试 |
|------|------|------|
| A | `test_train.py` | `test_model.py` |
| B | `test_train_cnn.py` | `test_model_cnn.py` |
| C | `test_train_part_c_momentum.py` | `test_model_part_c_momentum.py` |
| C 分析 | `part_c_analysis.py` | — |
