# Transformer from Scratch — 航班价格预测

纯 **NumPy** 从零实现的 Transformer（含完整前向/反向传播），并应用于印度国内航班机票价格回归预测。

> 不依赖 PyTorch、TensorFlow 等深度学习框架，核心库约 1500 行，配套测试与研究报告。

## 特性

- **手写 Transformer 全栈**：Embedding、Multi-Head Attention、Feed-Forward、LayerNorm、Encoder/Decoder
- **Encoder-only 回归**：将 22 维航班特征映射为标量价格
- **多航线联合训练**：Bangalore→Delhi、Delhi→Mumbai、Mumbai→Delhi 三条航线
- **完整测试套件**：11 项单元测试，覆盖导入、形状、反向传播与收敛
- **可视化分析**：训练过程自动生成 7 类分析图表

## 项目结构

```
transformer/
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
├── dataset/                  # 训练/测试 CSV 数据
├── 研究报告.md               # 完整研究报告（原理、实验、分析）
└── requirements.txt
```

## 环境要求

- Python 3.10+
- NumPy、Pandas、Matplotlib

## 快速开始

```bash
# 克隆仓库
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# 安装依赖
pip install -r requirements.txt

# 运行测试（验证 Transformer 实现）
python test_transformer.py

# 训练模型并生成可视化（约需数分钟）
python flight_price_predictor.py
```

训练完成后，图表将保存至 `results_economy/` 和 `results_business/`。

## 实验结果（摘要）

| 指标 | 经济舱 | 商务舱 |
|------|--------|--------|
| R² | ~0.72 | — |
| MAE | ~1,210 元 | — |

主要发现：**提前 34–39 天购票** 价格最低，临飞前购票可贵约 **73%**。

详细推导、实验设置与特征分析见 [研究报告.md](研究报告.md)。

## 数据集

`dataset/` 包含 3 条航线 × 2 种舱位（经济/商务）的 train/test CSV，特征包括：

- 经停次数、舱位、飞行时长、距起飞天数
- 航空公司、出发/到达时段、航线（独热编码）

## 引用

若本项目对你有帮助，欢迎 Star。研究背景基于 Vaswani 等 (2017) 的 Transformer 架构：

> Vaswani, A., et al. *Attention Is All You Need.* NeurIPS 2017.

## License

MIT
