# Project 1 — MNIST 分类（NumPy 自实现）

**结论（测试集）：** MLP **95.20%** → CNN-SGD **97.98%** → CNN-MomentGD **98.74%**（相同 `idx.pickle` 划分）。

**完整报告：** [PJ1/report/PJ1报告.md](PJ1/report/PJ1报告.md)  
**模型权重：** https://www.modelscope.cn/models/TOMOcaki/mnist-numpy-project1-25300740045

---

## 仓库结构

```text
Project1-2026/
├── .gitattributes          # * text=auto
├── README.md
├── requirements.txt        # 宽松 >=
├── requirements-freeze.txt # 锁定 ==（见文件头）
├── PJ1/
│   ├── README.md           # 权重路径、idx 警告、与课程框架差异
│   ├── report/
│   │   └── PJ1报告.md      # 课程报告（导出 PDF）
│   └── codes/
│       ├── best_models/              # Part A（勿提交）
│       ├── best_models_cnn/          # Part B
│       ├── best_models_cnn_momentum/ # Part C
│       ├── saved_models/             # 课程包遗留，可忽略
│       ├── idx.pickle                # 数据划分（勿提交）
│       └── results/                  # *.txt 指标；*.png 默认 gitignore
```

---

## 环境

```bash
cd PJ1/codes
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r ../../requirements.txt
```

**依赖文件：**

| 文件 | 用途 |
|------|------|
| `requirements.txt` | 推荐日常安装：`numpy`、`matplotlib`、`tqdm`、`Pillow`（`>=`） |
| `requirements-freeze.txt` | 与 `PJ1/codes/requirements-freeze.txt` 相同；`pip install -r requirements-freeze.txt` 复现锁定环境（`==`） |

---

## 数据

将 MNIST 四个 `.gz` 放入 `PJ1/codes/dataset/MNIST/`（**不随 repo 提供**）。

---

## 运行顺序

工作目录：`PJ1/codes`。

| 步骤 | 命令 | 说明 |
|------|------|------|
| Part A | `python test_train.py` | **会覆盖 `idx.pickle`**；仅首次或与 B/C 一并复现时运行 |
| Part A 测试 | `python test_model.py` | 读 `best_models/best_model.pickle` |
| Part B | `python test_train_cnn.py` | **复用**已有 `idx.pickle` |
| Part B 测试 | `python test_model_cnn.py` | 读 `best_models_cnn/` |
| Part C 分析 | `python part_c_analysis.py` | 仅用 Part B SGD 权重 |
| Part C 训练 | `python test_train_part_c_momentum.py` | 耗时较长 |
| Part C 测试 | `python test_model_part_c_momentum.py` | 读 `best_models_cnn_momentum/` |

### ⚠️ 重要：`idx.pickle`

**重跑 `test_train.py` 会覆盖 `idx.pickle`，破坏与 Part B/C 的可比性。**  
Part B/C 完成后请勿再跑 Part A 训练，除非已备份 idx 或准备重训 B/C。  
`test_train_cnn.py` 在存在 `idx.pickle` 时直接加载，不会主动覆盖。

---

## 权重 vs `saved_models/`

本项目 checkpoint 为 **`best_models*`**，不是课程框架中的 **`saved_models/`**。详见 [PJ1/README.md](PJ1/README.md)。

---

## 日志

权威数字：`PJ1/codes/results/*.txt`；Part B 长日志：`PJ1/codes/training_part_b.log`。  
**可选归档副本：** `PJ1/codes/log/test_train_part_*.log`（见 `log/README.md`）。

---

## 可视化

卷积核与混淆矩阵：运行 `python part_c_analysis.py`。勿依赖 `weight_visualization.py`（已弃用，路径过时）。

---

## 报告 PDF

1. 打开 [PJ1/report/PJ1报告.md](PJ1/report/PJ1报告.md)  
2. 确认文中引用的 `codes/results/...png` 在本地已生成且预览可见  
3. 导出 PDF 后提交 eLearning  

更多说明见 [PJ1/README.md](PJ1/README.md)。
