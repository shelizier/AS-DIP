# AS-DIP

[English](./README.md) | 中文版

AS-DIP（Accelerated Seismic Deep Image Prior）是一个面向地震数据的自监督去噪框架，核心思想是将以下三部分结合起来：

- Deep Image Prior（DIP）：利用未训练网络的结构先验进行重建
- Deep Random Projector（DRP）：冻结大部分随机初始化网络权重，降低优化成本
- Total Variation（TV）正则：增强地震事件连续性，抑制伪影和噪声

该框架不依赖成对的干净/含噪标签进行监督训练，而是直接对单幅含噪地震剖面进行自监督恢复。

## 主要特性

- 同时支持 `standard_dip`、`drp_dip` 和 `as_dip`
- 采用模块化工程结构，便于复用和维护
- 支持合成地震数据生成，包括 Ricker 子波、随机噪声和相干噪声
- 支持真实数据加载，当前使用 `.npy`，并预留 SEG-Y 扩展接口
- 提供 benchmark 脚本、结果汇总表和自动对比图
- 提供符合论文展示习惯的地震剖面与残差图绘制

## 项目结构

```text
AS-DIP/
├── configs/        # YAML 配置文件
├── core/           # 训练器、损失函数、设备工具
├── data/           # 合成数据生成与真实数据加载
│   └── field/      # 本项目当前使用的真实地震数据
├── models/         # UNet、轻量生成器、DRP 封装、激活函数
├── outputs/        # 实验输出结果
├── scripts/        # Benchmark 与汇总脚本
├── utils/          # 指标、绘图、汇总、f-k 工具
└── main.py         # 统一实验入口
```

## 核心思路

AS-DIP 的目标是：输入一幅含噪地震剖面，在没有成对监督标签的情况下，恢复更干净的地震反射结构。

它目前支持三种方法变体：

1. `standard_dip`
   直接优化生成网络参数，让网络拟合输入的含噪地震数据，是传统 DIP 基线。

2. `drp_dip`
   保留原始 DRP-DIP 思想，冻结大部分随机初始化网络参数，只优化 latent seed 和少量轻量参数，作为对比基线。

3. `as_dip`
   表示最终提出的方法 AS-DIP。它保留 DRP 风格的加速优化机制，并加入 TV 正则作为面向地震数据的增强。

可以简单理解为：

- `standard_dip` = 传统 DIP 基线
- `drp_dip` = 原始 DRP-DIP 对比方法
- `as_dip` = 最终提出的 AS-DIP 方法

## 数据说明

当前仓库中的真实地震数据保存在：

- `data/field/noisy.npy`
- `data/field/clean.npy`

这两份数据原本来自旧基线目录，现已迁移到新的统一数据目录中，并作为本项目最重要的真实数据源。

## 快速开始

使用 YAML 配置运行默认实验：

```bash
python main.py --config configs/default.yaml
```

默认配置现在运行的是 `as_dip`。

运行真实数据 benchmark：

```bash
python main.py \
  --dataset-type field \
  --benchmark \
  --iterations 10 \
  --experiment-name field_benchmark_full10 \
  --noisy-path data/field/noisy.npy \
  --clean-path data/field/clean.npy \
  --save-inputs
```

通过 YAML 批量运行 benchmark：

```bash
python scripts/run_benchmark.py --config configs/benchmark_field.yaml
```

汇总所有实验结果：

```bash
python scripts/aggregate_results.py --outputs-dir outputs --save-dir outputs/aggregate
```

## 当前对比方法

目前 benchmark 中对比的方法有三个：

- `standard_dip`
- `drp_dip`
- `as_dip`

当前自动统计的指标包括：

- 运行时间
- PSNR
- SNR
- SNR gain
- SSIM
- residual energy

在图表和结果表中，对外显示名称分别是：

- Standard DIP
- DRP-DIP
- AS-DIP

## 当前真实数据实验结果

当前已经保存的全尺寸真实地震数据 benchmark 结果保存在：

- `outputs/field_benchmark_full10/`

其中关键文件包括：

- `benchmark_summary.csv`
- `benchmark_summary.json`
- `method_overview.png`
- `benchmark_curves.png`
- `standard_dip/seismic_panels.png`
- `drp_dip/seismic_panels.png`
- `as_dip/seismic_panels.png`

当前这组真实数据 benchmark 的结果摘要为：

- Standard DIP: `306.67 s`, `PSNR 13.20 dB`
- DRP-DIP: `199.63 s`, `PSNR 16.04 dB`
- AS-DIP: `207.94 s`, `PSNR 22.41 dB`

在当前这组 10 次迭代的真实数据 benchmark 中，AS-DIP 在三种方法里给出了最好的重建质量。

## 说明

- 当前真实数据 benchmark 已经可以作为有效基线，但还不是最终调优后的最优结果。
- 如果要用于论文级实验，还需要进一步增加迭代次数并进行超参数搜索。
- 仓库中已经不再保留旧的 `DIP/` 和 `DRP_DIP/` 目录，其有效内容已整合进当前标准化结构。

## 引用建议

如果你将本仓库用于论文或学术研究，建议在正式写作中同时引用 DIP、DRP 以及相关地震去噪工作。
