"""
把 03_train_icm.ipynb 从旧版 (被误称为 LMC 的 ICM, 且受 CG 数值噪声影响)
修正为: 正确命名的 ICM + 精确 Cholesky + 订正后的叙述。

改动清单:
  代码  RANK 2->1, N_ITER 300->500, 加 exact() 上下文, 去 fast_pred_var,
        类名 LMCModel->ICMModel, 最终 loss 改为更新后重算, B 矩阵加经验相关性对照
  叙述  顶部改写为 "这是 ICM 不是 LMC" + rank!=Q 的证明
        末尾两段重写 (原 "lengthscale 崩塌 => 空间自相关弱 => 参数不可辨识" 三处均错)

用法:  python src/patch_03_to_icm.py [--execute]
       --execute 会用 nbclient 实际执行 notebook, 把输出写回文件
"""

import json
import sys
from pathlib import Path

NB = Path(r"E:\Research\src\03_train_icm.ipynb")

CELLS = {}

# ============================================================
CELLS["3db1bb4e"] = r"""# Jura: ICM (多输出 GP) vs 独立 GP 基线

核心对比:

  - 基线: 只用 259 个 Cd 观测建单输出 GP
  - ICM : 用 259 Cd + 359 Ni + 359 Zn 建多输出 GP

  在同样的 100 个测试位置上预测 Cd, 比 RMSE / MAE / NLL

---

## 这里建的是 ICM, 不是 LMC

kernel 结构:

    K[(x,i), (x',j)] = K_input(x, x') * B[i, j]

    K_input : 单一 RBF, ARD (两个空间方向各一个 lengthscale)
    B       : IndexKernel, B = W @ W.T + diag(kappa), W 形状 (P, rank)

**所有任务共享同一个 `K_input`** —— 这正是 ICM (Intrinsic Coregionalization
Model) 的定义。真正的 LMC 是多个核**求和**, 每个潜在函数有自己的空间尺度:

    K_LMC = sum_q  K_q(x, x') * B_q[i, j]      (每个 K_q 的 lengthscale 独立)

ICM 是 LMC 在 Q=1 时的特例。真 LMC 的实现见 `src/05_true_lmc.py`。

## rank 不是 LMC 的 Q

`rank` 只控制 `W` 的列数, 即 `B` 的秩, **完全不影响空间尺度的数量**。

而且 P=3 时它连 `B` 都影响不了: `B = W W^T + diag(kappa)` 在 rank=1 时已有
3+3 = 6 个自由参数, 而 3x3 正定矩阵的自由度也正好是 6 —— **rank=1 已张满
整个可行空间**, rank=2/3 加的全是冗余参数。

阶段 8 在精确算术下实测:

| 模型 | loss |
|---|---|
| ICM rank=1 | 1.004174 |
| ICM rank=2 | 1.004174 |
| ICM rank=3 | 1.004174 |
| LMC Q=1 | 1.004174 |

极差 1.34e-09, 机器精度级别的零。"""

# ============================================================
CELLS["dbcc5a53"] = r'''import contextlib

import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(r"E:\Research")
DATA = np.load(ROOT / "data" / "jura_prepared.npz", allow_pickle=True)

torch.manual_seed(0)

RANK = 1          # B 的秩。P=3 时 rank=1 已张满 PSD 空间, 更大的值纯冗余 (见顶部)
N_ITER = 500
LR = 0.05

# lengthscale 初值 (km)。
# 实测采样尺度: 最近邻 mean=0.125, 规则网格 0.25, 加密簇内低至 0.005
#              (见 src/08_sampling_design.py)
# 早期版本此处注释写的 "采样间距 0.27km" 是 sqrt(包围盒面积 / N) 的密度估计,
# 不是实测值 —— Jura 是网格 + 嵌套簇的设计, 密度估计会高估典型间距。
LS_INIT = 0.5


# ---------- 精确计算上下文 ----------
# GPyTorch 的 max_cholesky_size 默认 = 800。本项目训练集 N=977 > 800,
# 不加这个上下文, 边际似然就走 CG + Lanczos 随机估计:
#   - 同一组参数重复求值, 极差达 0.0095
#   - 且系统性高估对数似然约 0.03 (不是零均值噪声)
# 独立 GP 只有 259 点 (<800), 本来就是精确的, 一并套上以保持一致。
@contextlib.contextmanager
def exact():
    with gpytorch.settings.max_cholesky_size(10_000), \
         gpytorch.settings.fast_computations(
             covar_root_decomposition=False, log_prob=False, solves=False
         ):
        yield'''

# ============================================================
CELLS["64c169f5"] = r"""# 模型 1: ICM (Hadamard 长表形式, 支持缺失观测)

长表: 每行 = (空间坐标 x, 任务编号 i, 观测值 y)。Cd 只有 259 行, Ni/Zn 各 359 行。

宽表 (359x3 矩阵) 放不下 Cd 的 100 个缺口, 长表可以 —— 这就是能做 heterotopic
实验的前提。

代价: `covar_x.mul(covar_i)` 是逐元素相乘, 即 Kronecker 积 `B ⊗ K_input` 在长表下
的写法。数学等价, 但**失去了 Kronecker 的计算优势** (特征分解需要完整块结构),
所以 O((Dn)^3) → O(D^3 + n^3) 的加速用不上。这里 N=977, 直接 Cholesky 即可。

**允许缺失 与 Kronecker 加速, 二者不可兼得** —— 这正是真实场景需要 SVGP 的原因,
值得写进报告。"""

# ============================================================
CELLS["4b6024df"] = r'''class ICMModel(gpytorch.models.ExactGP):
    """Intrinsic Coregionalization Model: 单一空间核 x 任务协方差 B。"""

    def __init__(self, x, i, y, likelihood, num_tasks, rank):
        super().__init__((x, i), y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # 注意: 这里不加 ScaleKernel。B 的对角元已经承担了 output scale,
        #       再加一个会造成参数不可辨识 (两个参数只能确定一个乘积)。
        #       独立 GP 基线要加, 因为它没有 B 来提供尺度 —— 这个不对称是有意的。
        self.covar_module = gpytorch.kernels.RBFKernel(ard_num_dims=2)
        self.task_covar_module = gpytorch.kernels.IndexKernel(
            num_tasks=num_tasks, rank=rank
        )

    def forward(self, x, i):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)          # K_input(x, x')
        covar_i = self.task_covar_module(i)     # B[i, j]
        return gpytorch.distributions.MultivariateNormal(
            mean_x, covar_x.mul(covar_i)        # 逐元素相乘 = Kronecker 的长表版本
        )


lik_icm = gpytorch.likelihoods.GaussianLikelihood().double()
icm = ICMModel(Xtr, itr, ytr, lik_icm, num_tasks=P, rank=RANK).double()
icm.covar_module.lengthscale = torch.tensor([[LS_INIT, LS_INIT]], dtype=torch.float64)

icm.train(); lik_icm.train()
opt = torch.optim.Adam(icm.parameters(), lr=LR)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik_icm, icm)

print("\n--- 训练 ICM (精确 Cholesky) ---")
with exact():                                   # <-- 关键: 不加这句就是 CG 随机估计
    for k in range(N_ITER):
        opt.zero_grad()
        loss = -mll(icm(Xtr, itr), ytr)
        loss.backward()
        if (k + 1) % 100 == 0:
            print(f"  iter {k+1:4d}  loss={loss.item():.6f}")
        opt.step()
    # 最后一步更新后重算, 而不是沿用更新前的值
    with torch.no_grad():
        loss_icm = -mll(icm(Xtr, itr), ytr).item()
print(f"  最终 loss = {loss_icm:.6f}")'''

# ============================================================
CELLS["08dcd33b"] = r'''mask = (itr == 0)
Xcd, ycd = Xtr[mask], ytr[mask]


class SingleGP(gpytorch.models.ExactGP):
    """单输出 GP 基线: 只看 259 个 Cd 观测。"""

    def __init__(self, x, y, likelihood):
        super().__init__(x, y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # 这里要 ScaleKernel: 没有 B 来提供 output scale, 必须单独有一个
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=2)
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


lik_gp = gpytorch.likelihoods.GaussianLikelihood().double()
gp = SingleGP(Xcd, ycd, lik_gp).double()
gp.covar_module.base_kernel.lengthscale = torch.tensor(
    [[LS_INIT, LS_INIT]], dtype=torch.float64
)

gp.train(); lik_gp.train()
opt2 = torch.optim.Adam(gp.parameters(), lr=LR)
mll2 = gpytorch.mlls.ExactMarginalLogLikelihood(lik_gp, gp)

# 259 < 800, 本来就走精确 Cholesky; 套上 exact() 只是让两个模型的数值路径一致
print("\n--- 训练独立 GP 基线 ---")
with exact():
    for k in range(N_ITER):
        opt2.zero_grad()
        loss = -mll2(gp(Xcd), ycd)
        loss.backward()
        if (k + 1) % 100 == 0:
            print(f"  iter {k+1:4d}  loss={loss.item():.6f}")
        opt2.step()
    with torch.no_grad():
        loss_gp = -mll2(gp(Xcd), ycd).item()
print(f"  最终 loss = {loss_gp:.6f}")'''

# ============================================================
CELLS["28bc60d0"] = r'''def unstd_log(y):
    """标准化尺度 -> log 浓度尺度 (task 0 = Cd)"""
    return y * sd[0] + mu[0]


def metrics(pred_dist, y_true, name):
    m = pred_dist.mean
    rmse_s = torch.sqrt(((m - y_true) ** 2).mean()).item()
    mae_s = (m - y_true).abs().mean().item()
    nll = -pred_dist.log_prob(y_true).item() / len(y_true)

    # 还原到原始 ppm 尺度
    #   naive : exp(后验均值)   -> 对数正态的中位数
    #   corr  : exp(mu + s^2/2) -> 对数正态的期望
    # 文献里两种都有人用, 报告里说清楚用了哪个即可。
    log_m = unstd_log(m.detach().numpy())
    log_v = (pred_dist.variance.detach().numpy()) * sd[0] ** 2
    true_ppm = np.exp(unstd_log(y_true.numpy()))
    rmse_naive = np.sqrt(((np.exp(log_m) - true_ppm) ** 2).mean())
    rmse_corr = np.sqrt(((np.exp(log_m + log_v / 2) - true_ppm) ** 2).mean())

    print(f"\n[{name}]")
    print(f"  标准化尺度   RMSE={rmse_s:.4f}  MAE={mae_s:.4f}  NLL={nll:.4f}")
    print(f"  原始 ppm     RMSE={rmse_naive:.4f} (naive) / {rmse_corr:.4f} (lognormal 修正)")
    return dict(name=name, rmse_std=rmse_s, mae_std=mae_s, nll=nll,
                rmse_ppm=rmse_naive, mean=m.detach().numpy())


icm.eval(); lik_icm.eval(); gp.eval(); lik_gp.eval()
# 注意: 这里不能用 gpytorch.settings.fast_pred_var() —— 它用 Lanczos 近似后验方差,
#       RMSE/MAE 只取均值不受影响, 但 NLL 会偏 (旧版本因此报 1.2060, 精确值 1.2530)。
with torch.no_grad(), exact():
    pred_icm = lik_icm(icm(Xte, ite))
    pred_gp = lik_gp(gp(Xte))

print("\n" + "=" * 58)
print("测试集结果 (100 个被藏起来的 Cd 位置)")
print("=" * 58)
r_gp = metrics(pred_gp, yte, "独立 GP (只看 Cd)")
r_icm = metrics(pred_icm, yte, "ICM (借用 Ni/Zn)")

imp = (r_gp["rmse_std"] - r_icm["rmse_std"]) / r_gp["rmse_std"] * 100
print("\n" + "-" * 58)
print(f">> ICM 相对独立 GP 的 RMSE 改善: {imp:+.2f}%")
print("-" * 58)
print("注: 这是单次划分 (seed=0) 的结果。10 次随机划分的稳健版本见")
print("    src/07_robustness.py —— ICM +26.3% ± 5.4%, 10/10 胜。")'''

# ============================================================
CELLS["daa41f0c"] = r'''B = icm.task_covar_module.covar_matrix.to_dense().detach().numpy()
D = np.sqrt(np.diag(B))
B_corr = B / np.outer(D, D)

print("\n学到的任务协方差矩阵 B:")
print(np.round(B, 4))
print("\n转成相关矩阵 (可与 EDA 的经验相关性对比):")
for r, n in enumerate(NAMES):
    print(f"  {n}: " + "  ".join(f"{B_corr[r, c]:+.3f}" for c in range(P)))

# EDA 实测的 log 尺度 Pearson 相关 (见 src/01_load_jura)
EMP = {("Cd", "Ni"): 0.654, ("Cd", "Zn"): 0.686, ("Ni", "Zn"): 0.699}
print("\n模型学到的 vs EDA 经验相关性:")
print(f"  {'配对':<10}{'经验':<10}{'学到':<10}{'差':<10}")
for (a, b), emp in EMP.items():
    ia, ib = NAMES.index(a), NAMES.index(b)
    print(f"  {a+'-'+b:<10}{emp:<10.3f}{B_corr[ia, ib]:<10.3f}{B_corr[ia, ib]-emp:<+10.3f}")
print("  >> 模型在完全不知道经验相关性的情况下自己学出了几乎相同的结构。")
print("     系统性略高是正常的: B 刻画去噪后的潜在相关性, 而 Pearson 是含噪观测的")
print("     相关性, 噪声会稀释相关系数。")

ls = icm.covar_module.lengthscale.detach().numpy().ravel()
print(f"\n学到的空间 lengthscale (km): X={ls[0]:.4f}, Y={ls[1]:.4f}")
print(f"噪声方差: ICM={lik_icm.noise.item():.4f}, 独立GP={lik_gp.noise.item():.4f}")'''

# ============================================================
CELLS["b94ff5bb"] = r"""# 观察: ICM 学到的 lengthscale 非常短

学到约 `(0.079, 0.058) km`, 比规则网格间距 (0.25 km) 小得多。这个数意味着什么,
值得追问 —— 但要先把参照系搞对。

## 早期版本的参照系是错的

原来这里用 "平均采样间距 0.27 km" 作参照。**那个数不是实测值**, 而是
`sqrt(包围盒面积 / N) = 0.2525` 的密度估计。

Jura 的实际设计是**规则网格 + 嵌套加密簇**, 最近邻距离呈双峰
(见 `src/08_sampling_design.py`):

| 统计量 | 值 | 含义 |
|---|---|---|
| NN p25 | 0.0143 km | 加密簇内 |
| NN 中位数 | 0.100 km | |
| NN p75–p100 | 0.2496–0.2505 km | 规则网格 |
| 有邻居 < 50 m 的点 | 151 / 359 (42%) | |

**没有单一的"采样间距"**, 有两个尺度: 簇内约 10 m, 网格 250 m。密度估计恰好落在
网格那个峰上, 所以它测的是网格间距, 不是平均间距。

## 用正确参照重算

lengthscale 几何均值 0.0674 km, RBF 在各距离上的相关性:

| 距离 | 相关性 | |
|---|---|---|
| 0.005 km | **0.997** | 簇内最近点对 |
| 0.010 km | **0.989** | 簇内典型 |
| 0.050 km | 0.759 | |
| 0.125 km | 0.179 | 平均最近邻 |
| 0.250 km | 0.001 | 网格间距 |

**结论要改**: 模型不是"完全不用空间信息", 而是**只用簇内的空间信息**。在网格尺度上
相关性确实几乎为零, 但 42% 的采样点处在簇里, 那部分被充分利用了。

## "退化成同点回归"的担忧成立吗

部分成立。实测一个同点线性回归 `Cd ~ Ni + Zn` 作对照:

| 模型 | 测试 RMSE |
|---|---|
| 独立 GP | 0.9715 |
| **同点线性回归** | **0.7629** |
| ICM | 0.7428 |
| 真 LMC (Q=2) | 0.7313 |

三参数的线性回归 (系数 Ni=0.42, Zn=0.41) 就能到 0.7629 —— **"改善"里的大部分确实
来自"用了同点的 Ni/Zn"这件事本身**。但 GP 仍额外贡献 4.15%, 且线性回归:

- 给不出校准的不确定性 (本研究的核心产出之一)
- 在没有同点 Ni/Zn 的位置完全失效, GP 还能退回空间插值

**报告里应主动给出这个对照**, 否则一定会被问。"""

# ============================================================
CELLS["cebb928a"] = r"""## 两个模型的 lengthscale 差很多 —— 但这不能推出"不可辨识"

| 模型 | ℓ_X | ℓ_Y | 训练数据 |
|---|---|---|---|
| 独立 GP | 0.096 | 0.266 | 259 个 Cd |
| ICM | 0.079 | 0.058 | 977 个 (含 718 个 Ni/Zn) |

早期版本据此推断"似然曲面平坦、参数不可辨识"。**这个推断两处都错:**

**1. 这两个数本来就不该直接比。** 它们拟合的是不同数据、不同似然函数。独立 GP 的
lengthscale 只受 259 个 Cd 观测约束; ICM 的被观测数 2.8 倍的 Ni/Zn 主导。差异来自
"谁主导优化", 不是"曲面平坦"。

**2. 曲面根本不平。** 阶段 8 在精确算术下做了 5 个初值的扫描, ICM 的 loss
**极差 = 1.3e-09** —— 所有初值收敛到同一个解, 到机器精度。

## 真正的问题在别处: 数值精度

原始训练日志里有一处异常:

```
iter 200  loss=0.9970
iter 250  loss=1.0013   <-- 上升
iter 300  loss=0.9992
```

而独立 GP 的日志是 `1.2163` 连续四次纹丝不动。**两者形态的差异就是线索**:

训练集 977 > GPyTorch 的 `max_cholesky_size` 默认值 800, 所以 ICM 走了
CG + Lanczos 随机估计, 独立 GP (259 点) 走的是精确 Cholesky。

实测同一组参数重复求值:

| 计算方式 | 重复求值 | 极差 |
|---|---|---|
| CG + Lanczos | 1.490759, 1.488572, 1.482091, 1.481232, 1.481291 | **0.0095** |
| 精确 Cholesky | 1.485629, 1.485629, 1.485629 | **0** |

而且是**系统性偏差**, 不是零均值噪声: 精确重跑后所有 loss 一致升高 0.023–0.030。

本 notebook 已加 `exact()` 上下文修正, 上面的训练日志现在是单调收敛的。

## 结论订正

| 早期结论 | 现状 |
|---|---|
| "Cd 空间自相关弱" | ❌ 被推翻 |
| "任意两点相关性 ≈ 0.0005" | ❌ 参照系用错 |
| "参数不可辨识" | ❌ 极差 1.3e-9 |
| "rank 是 LMC 的 Q" | ❌ rank 完全无效 |

**"空间自相关弱"是 ICM 单一尺度约束造成的假象。** 真 LMC (`src/05_true_lmc.py`)
用两个独立核重跑, 找到了并存的两个尺度:

| | ℓ_X | ℓ_Y | Cd 方差占比 | Cd–Ni 相关性 |
|---|---|---|---|---|
| q=1 短程 | 0.052 | 0.042 | 77.7% | 0.673 |
| q=2 长程 | **0.511** | **0.347** | 22.3% | **0.941** |

长程结构一直存在, ICM 被迫二选一时选了短程。在预测面上, 长程分量贡献了 **62.9%**
的幅度。

而 ICM 报告的单一相关性 0.743, 是 0.673 和 0.941 的加权混合 —— **一个不对应任何
真实过程的平均数**。这才是应该用真 LMC 的理由 (预测精度上 LMC 与 ICM 无显著差异,
10 次划分 5:5, p=0.19)。

---

**后续**: `src/05_true_lmc.py` (真 LMC + BIC 选 Q) → `src/06_spatial_maps.py`
(空间图) → `src/07_robustness.py` (10 次划分)"""


# ============================================================
def main():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    ids = {c.get("id") for c in nb["cells"]}
    missing = set(CELLS) - ids
    if missing:
        sys.exit(f"notebook 里找不到这些 cell id: {missing}")

    n = 0
    for c in nb["cells"]:
        cid = c.get("id")
        if cid in CELLS:
            c["source"] = CELLS[cid].splitlines(keepends=True)
            n += 1
        # 所有代码单元格清空旧输出 —— 它们对应的是旧代码
        if c["cell_type"] == "code":
            c["outputs"] = []
            c["execution_count"] = None

    NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] 已修改 {n} 个单元格, 并清空所有代码单元格的旧输出")
    print(f"     {NB}")

    if "--execute" in sys.argv:
        import nbformat
        from nbclient import NotebookClient

        print("\n执行 notebook (约 60 秒) ...")
        book = nbformat.read(NB, as_version=4)
        client = NotebookClient(
            book, timeout=1800, kernel_name="python3",
            resources={"metadata": {"path": str(NB.parent)}},
        )
        client.execute()
        nbformat.write(book, NB)
        print(f"[OK] 执行完成, 输出已写回 {NB}")


if __name__ == "__main__":
    main()
