# Jura 多输出高斯过程 (MOGP)

用镍和锌辅助预测瑞士 Jura 地区土壤中的镉浓度。对比单输出 GP 基线、ICM 与 LMC 三种模型。

## 问题设定

Jura 数据集有 359 个采样位置，每个位置测了多种重金属。本项目采用 MOGP 文献里的标准
**heterotopic（异位）** 设定：

| 输出 | 观测数 | 角色 |
|---|---|---|
| Cd（镉） | 259 | 主变量，另外 100 个位置藏起来做测试 |
| Ni（镍） | 359 | 辅助变量，全部位置可见 |
| Zn（锌） | 359 | 辅助变量，全部位置可见 |

任务：在那 100 个被藏起来的位置上预测 Cd。现实动机是 Cd 的测定成本高于 Ni/Zn。

## 模型

MOGP 是类别名，ICM 和 LMC 都属于它；独立 GP 是类别之外的基线。

```
独立 GP        只用 259 个 Cd 观测
                                       
MOGP
 ├── ICM       K = k(x,x') · B[i,j]           单一空间尺度，所有输出共享
 └── LMC       K = Σ_q k_q(x,x') · B_q[i,j]   每个潜在函数各有一个尺度
               （ICM 是 LMC 在 Q=1 时的特例）
```

其中 `B` 由 `IndexKernel` 参数化为 `B = W Wᵀ + diag(κ)`。采用 Hadamard 长表形式
（每行 = 坐标 + 任务编号 + 观测值）以支持缺失观测——代价是失去 Kronecker 加速。

## 主要结果

**MOGP 显著优于单输出基线**，10 次随机划分全部验证：

| 模型 | 测试 RMSE | 相对独立 GP |
|---|---|---|
| 独立 GP | 0.8694 ± 0.0620 | — |
| ICM | 0.6399 ± 0.0619 | +26.3% ± 5.4% |
| LMC (Q=2) | 0.6320 ± 0.0544 | +27.3% ± 4.5% |

配对 t 检验 p = 6×10⁻⁸，10/10 划分获胜。

**LMC 恢复出双尺度结构**（BIC 与 AIC 一致选定 Q=2）：

| 潜在函数 | ℓ_X (km) | ℓ_Y (km) | Cd 方差占比 | 跨输出相关 | 解释 |
|---|---|---|---|---|---|
| q=1 短程 | 0.052 | 0.042 | 77.7% | 0.67 | 局部人为输入 |
| q=2 长程 | 0.511 | 0.347 | 22.3% | **0.88–0.94** | 区域地质背景 |

长程分量的跨输出相关性接近 1，意味着三种金属在该尺度上是同一潜在场的线性缩放。

**但 LMC 并不比 ICM 预测得更好**：10 次划分下 5 胜 5 负，p = 0.19。LMC 的价值在
可解释性——ICM 只有一个 `B`，它报告的相关性 0.743 是 0.673 与 0.941 的加权混合，
不对应任何真实过程。

## 流水线

```bash
python src/01_load_jura         # 下载 + EDA
python src/02_prepare_data      # log 变换、防泄漏标准化、heterotopic 划分
# src/03_train_icm.ipynb        # ICM vs 独立 GP
# src/04_sensitivity.ipynb      # lengthscale 初值敏感性 + rank 可辨识性
# src/05_true_lmc.ipynb         # 真 LMC + BIC 选 Q      (脚本版: 05_true_lmc.py)
python src/06_spatial_maps.py   # 空间后验图 + 潜在函数分解
python src/07_robustness.py     # 10 次随机划分 + 配对检验
python src/08_sampling_design.py
python src/09_icm_params.py
python src/10_report_figures.py
```

`src/models.py` 存放共享的模型定义与 `exact()` 数值上下文。

## 两个必须注意的实现细节

**1. 强制精确 Cholesky。** GPyTorch 的 `max_cholesky_size` 默认为 800，而本项目训练集
有 977 个观测。不显式提高阈值，边际似然就会走 CG + 随机 Lanczos 估计：同一组参数
重复求值极差达 0.0095，且系统性高估对数似然约 0.03。所有训练与预测都必须包在
`exact()` 上下文里。

**2. 各潜在函数的 lengthscale 初值必须拉开。** 全设成相同值会导致梯度对称，所有 q
收敛到同一个解，Q>1 会静默退化成 ICM。

## 数据来源

Jura 数据集来自 Goovaerts (1997) *Geostatistics for Natural Resources Evaluation*，
经 OpenML（`data_id=41554`）分发。`src/01_load_jura` 首次运行时自动下载并缓存到
`data/jura.csv`。

## 环境

见 `requirements.txt`。Python 3.14，PyTorch CPU 版即可（完整流水线约 30 分钟）。
