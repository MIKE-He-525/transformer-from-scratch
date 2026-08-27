# Transformer from Scratch

纯 NumPy 手写 Transformer（完整前向/反向），用于印度国内航班票价回归预测。

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## 这是什么

从零实现 Transformer 的全部组件（Embedding、Multi-Head Attention、Feed-Forward、LayerNorm、Encoder/Decoder），不依赖 PyTorch 或 TensorFlow。核心库约 2,300 行代码，配套 11 项单元测试和完整的梯度反向传播。

将 Encoder-only 架构用于回归任务：输入 22 维航班特征（经停次数、舱位、航空公司、起飞时段等），预测机票价格。

---

## 特性

- **纯 NumPy 实现**：Transformer 全栈（Embedding、Attention、FFN、LayerNorm、残差连接）
- **Encoder-only 回归**：将航班特征编码后通过线性层回归为价格
- **多航线联合训练**：Bangalore→Delhi、Delhi→Mumbai、Mumbai→Delhi 三条航线联合建模
- **11 项单元测试**：覆盖导入、形状、反向传播、梯度更新、mask 机制、收敛性和数值稳定性
- **7 类可视化图表**：训练曲线、特征重要性、预测散点图、购票时机分析等

---

## 快速开始

```bash
# 克隆仓库
git clone https://github.com/MIKE-He-525/transformer-from-scratch.git
cd transformer-from-scratch

# 安装依赖
pip install -r requirements.txt

# 运行测试（验证 Transformer 实现）
python test_transformer.py
```

### 训练模型

数据集需自行准备（见下方[数据集](#数据集)章节）。将 CSV 文件放入 `dataset/` 目录后：

```bash
# 训练模型并生成可视化（需数分钟）
python flight_price_predictor.py
```

训练完成后，图表将保存至 `results_economy/` 和 `results_business/`。

---

## 实验结果

使用三条航线的训练数据联合训练，在测试集上的表现（详见[研究报告.md](研究报告.md)）：

| 指标 | 经济舱 | 商务舱 |
|------|--------|--------|
| R² | ~0.72 | ~0.68 |
| RMSE | ~1,570 元 | ~2,900 元 |
| MAE | ~1,210 元 | ~2,200 元 |

**主要发现**：
- 提前 **34–39 天** 购票价格最低
- 临飞前购票比最优时机贵约 **73%**
- `days_left`（距起飞天数）是最显著的价格驱动因素

训练过程生成 7 类分析图表：数据分布、特征重要性、训练曲线、预订时间线、特征敏感性、预测散点图等。

---

## 数据集

本项目使用 Kaggle 印度航班价格数据集。数据集**未包含在仓库中**，需自行下载后放入 `dataset/` 目录：

```
dataset/
├── train_bangalore_delhi_economy.csv
├── test_bangalore_delhi_economy.csv
├── train_delhi_mumbai_economy.csv
├── test_delhi_mumbai_economy.csv
├── train_mumbai_delhi_economy.csv
├── test_mumbai_delhi_economy.csv
├── train_bangalore_delhi_business.csv
├── test_bangalore_delhi_business.csv
└── ...（其他商务舱文件同理）
```

**特征说明（22 维）**：

| 特征列 | 说明 |
|--------|------|
| `stops` | 经停次数（0/1/2） |
| `class` | 舱位（0=经济舱，1=商务舱） |
| `duration` | 飞行时长（标准化） |
| `days_left` | 距起飞天数（标准化） |
| `airline_*` | 航空公司独热编码（5 家航司） |
| `departure_time_*` | 出发时段独热编码（5 个时段） |
| `arrival_time_*` | 到达时段独热编码（5 个时段） |
| `route_*` | 航线独热编码（3 条航线） |

---

## 模型架构

```
航班特征 (22 维)
    │
    ▼
Continuous Embedding  →  Positional Encoding
    │
    ▼
Encoder Layer × 3  (Self-Attention + FFN)
    │
    ▼
LayerNorm  →  Linear(d_model → 1)  →  预测价格
```

完整的 Seq2Seq Transformer（Encoder + Decoder）实现在 `transformer/model.py`，可用于序列到序列任务。航班价格预测使用 Encoder-only 变体，见 `flight_price_predictor.py`。

---

## 项目结构

```
transformer-from-scratch/
├── transformer/              # 核心库（纯 NumPy Transformer 实现）
│   ├── embedding.py          # 词嵌入 + 正弦位置编码
│   ├── attention.py          # 缩放点积注意力 + 多头注意力
│   ├── feed_forward.py       # 前馈网络
│   ├── normalization.py      # LayerNorm + Pre-LN 残差连接
│   ├── encoder.py            # 编码器
│   ├── decoder.py            # 解码器
│   ├── model.py              # 完整 Seq2Seq Transformer
│   └── utils.py              # Mask 工具函数
├── flight_price_predictor.py # 航班价格预测主程序
├── test_transformer.py       # 单元测试
├── 研究报告.md               # 完整研究报告（原理、实验、分析）
├── requirements.txt
└── LICENSE
```

---

## 依赖

- Python 3.10+
- NumPy 1.24+
- Pandas 2.0+
- Matplotlib 3.7+

---

## 许可证

[MIT](LICENSE) © MIKE-HE-525
