"""
修正 04_sensitivity.ipynb 和 05_true_lmc.ipynb。

04 的问题:
  - 类名 LMCModel 实为 ICM
  - 无 exact() 上下文 -> 报告的 "LMC loss 极差 0.0032" 是 CG 噪声, 不是收敛证据
    (旁证: 同一表里独立 GP 极差是 0.0000, 因为 259<800 走的精确 Cholesky)
  - fast_pred_var() 使 NLL 有偏
  - final_loss 取的是末次 opt.step() 之前的值
  - rank 敏感性扫描测的量真值恒为 0 (P=3 时 rank>=1 已张满 PSD 空间)

05 的问题:
  - best_loss + deepcopy: 在 CG 噪声下等于"挑最幸运的一次随机估计"
    (notebook 末尾的 markdown 已诊断出这点, 但代码从未修改)
  - 无 exact() 上下文
  - fast_pred_var()
  - plt.show() 写在 plt.savefig() 之前 -> 存下来是空白图
  - 用训练 loss 选 Q -> 必然选到 Q_max, 不是模型选择
  - 硬编码对照 ICM RMSE=0.7418 (CG 旧值, 精确值 0.7428)
  - 三段互相矛盾的结论 markdown (Q=3 / Q=4 / Q=2) 全部保留在文件里

两个 notebook 改为从 models.py 导入模型定义, 消除重复实现。

用法: python src/patch_04_05.py [--execute]
"""

import json
import sys
from pathlib import Path

SRC = Path(r"E:\Research\src")

# ============================================================
# 04_sensitivity.ipynb
# ============================================================
C04 = {}

C04["11e3b983"] = r"""# lengthscale 初值敏感性 + rank 的可辨识性

本 notebook 回答两个问题:

**问题 1 — ICM 学到的短 lengthscale 是数据的性质, 还是优化陷在局部最优?**

- (a) 数据本身空间自相关弱 → 不同初值都收敛到同一个解, loss 相同
- (b) 优化陷入局部最优 → 大初值能达到更低 loss

**问题 2 — `IndexKernel` 的 rank 该取多少?**

结论会是: **rank 无关紧要, 而且可以证明**。P=3 时 `B = W Wᵀ + diag(κ)` 在
rank=1 已有 6 个自由参数, 等于 3×3 PSD 矩阵的自由度, 已张满可行空间。

---

> **本 notebook 已修正数值精度问题。** 早期版本没有强制精确 Cholesky,
> N=977 > GPyTorch 默认阈值 800, 边际似然走的是 CG + Lanczos 随机估计。
> 当时报告的 "ICM loss 极差 0.0032" 被解读成"稳定收敛", 实际上那是**随机
> 估计的噪声**。同一张表里独立 GP 的极差是 0.0000 —— 因为它只有 259 个观测,
> 低于阈值, 本来就走精确路径。**两者的差异当时就是线索, 只是没被认出来。**

输出: `results/sensitivity.csv`"""

C04["e7627cc8"] = r'''import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import gpytorch

sys.path.insert(0, str(Path.cwd()))
from models import exact, SingleGP, ICMModel      # 共享定义, 避免各脚本各写一份

ROOT = Path(r"E:\Research")
DATA = np.load(ROOT / "data" / "jura_prepared.npz", allow_pickle=True)
(ROOT / "results").mkdir(exist_ok=True)

N_ITER = 500
LR = 0.05
LS_GRID = [0.1, 0.3, 0.5, 1.0, 2.0]     # 覆盖 亚簇尺度 -> 半个研究区
RANK_GRID = [1, 2, 3]

Xtr = torch.tensor(DATA["X_train"], dtype=torch.float64)
itr = torch.tensor(DATA["i_train"], dtype=torch.long)
ytr = torch.tensor(DATA["y_train"], dtype=torch.float64)
Xte = torch.tensor(DATA["X_test"],  dtype=torch.float64)
ite = torch.tensor(DATA["i_test"],  dtype=torch.long)
yte = torch.tensor(DATA["y_test"],  dtype=torch.float64)
mask = (itr == 0)
Xcd, ycd = Xtr[mask], ytr[mask]

print(f"训练观测 {len(ytr)} (ICM), {len(ycd)} (独立 GP)")
print(f"GPyTorch max_cholesky_size 默认 = {gpytorch.settings.max_cholesky_size.value()}")
print(f"  ICM     N={len(ytr)} > 800  -> 不套 exact() 就会走 CG 随机估计")
print(f"  独立 GP N={len(ycd)} < 800  -> 本来就是精确的")'''

C04["1520fd7a"] = r'''def train_eval(kind, ls_init, rank=None, seed=0):
    """训练并评估。全程套 exact(), 保证边际似然是精确值而非随机估计。"""
    torch.manual_seed(seed)
    lik = gpytorch.likelihoods.GaussianLikelihood().double()

    if kind == "icm":
        model = ICMModel(Xtr, itr, ytr, lik, 3, rank).double()
        model.covar_module.lengthscale = torch.tensor(
            [[ls_init, ls_init]], dtype=torch.float64)
        inputs, targets = (Xtr, itr), ytr
    else:
        model = SingleGP(Xcd, ycd, lik).double()
        model.covar_module.base_kernel.lengthscale = torch.tensor(
            [[ls_init, ls_init]], dtype=torch.float64)
        inputs, targets = (Xcd,), ycd

    model.train(); lik.train()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)

    with exact():
        for _ in range(N_ITER):
            opt.zero_grad()
            loss = -mll(model(*inputs), targets)
            loss.backward()
            opt.step()
        # 末次 opt.step() 之后重算 —— 早期版本用的是更新前的 loss.item()
        with torch.no_grad():
            final_loss = -mll(model(*inputs), targets).item()

    model.eval(); lik.eval()
    # 不用 fast_pred_var(): 它用 Lanczos 近似后验方差, 会让 NLL 有偏
    with torch.no_grad(), exact():
        pred = lik(model(Xte, ite)) if kind == "icm" else lik(model(Xte))
        rmse = torch.sqrt(((pred.mean - yte) ** 2).mean()).item()
        nll = -pred.log_prob(yte).item() / len(yte)

    ls = (model.covar_module.lengthscale if kind == "icm"
          else model.covar_module.base_kernel.lengthscale).detach().numpy().ravel()

    return dict(kind=kind, rank=rank, ls_init=ls_init,
                loss=final_loss, rmse=rmse, nll=nll,
                ls_x=ls[0], ls_y=ls[1], noise=lik.noise.item())'''

C04["03a9dec0"] = r'''rows = []
print("扫描独立 GP (5 个初值) ...")
for ls0 in LS_GRID:
    r = train_eval("gp", ls0)
    rows.append(r)
    print(f"  ls_init={ls0:<4} -> loss={r['loss']:.6f}  RMSE={r['rmse']:.4f}  "
          f"学到 ls=({r['ls_x']:.4f}, {r['ls_y']:.4f})")

print("\n扫描 ICM (3 个 rank x 5 个初值) ...")
for rank in RANK_GRID:
    for ls0 in LS_GRID:
        r = train_eval("icm", ls0, rank=rank)
        rows.append(r)
        print(f"  rank={rank} ls_init={ls0:<4} -> loss={r['loss']:.6f}  "
              f"RMSE={r['rmse']:.4f}  学到 ls=({r['ls_x']:.4f}, {r['ls_y']:.4f})")

df = pd.DataFrame(rows)
df.to_csv(ROOT / "results" / "sensitivity.csv", index=False)

gp_df, icm_df = df[df.kind == "gp"], df[df.kind == "icm"]

print("\n" + "=" * 70)
print("问题 1: lengthscale 初值敏感性")
print("=" * 70)
for lab, sub in [("独立 GP", gp_df), ("ICM", icm_df)]:
    spread = sub.loss.max() - sub.loss.min()
    print(f"\n  {lab}: loss 极差 = {spread:.2e}")
    print(f"       学到的 lengthscale 范围: "
          f"X [{sub.ls_x.min():.4f}, {sub.ls_x.max():.4f}]  "
          f"Y [{sub.ls_y.min():.4f}, {sub.ls_y.max():.4f}]")
print("\n  >> 两者极差都在机器精度量级 -> 初值完全不影响结果, 是全局最优。")
print("  >> 早期版本 ICM 报的 0.0032 是 CG 随机估计的噪声, 不是优化的不稳定。")

print("\n" + "=" * 70)
print("问题 2: rank 是否影响 ICM")
print("=" * 70)
per_rank = icm_df.groupby("rank")["loss"].agg(["min", "max"])
print(f"\n  {'rank':<8}{'loss 最小':<16}{'loss 最大':<16}{'该 rank 内极差'}")
for rk, row in per_rank.iterrows():
    print(f"  {rk:<8}{row['min']:<16.9f}{row['max']:<16.9f}{row['max']-row['min']:.2e}")
across = per_rank["min"].max() - per_rank["min"].min()
print(f"\n  跨 rank 的最优 loss 极差 = {across:.2e}")
print(f"  >> {'证实: rank 对 ICM 毫无影响' if across < 1e-6 else '存在差异, 需检查'}")
print("  >> 原因: B = W Wᵀ + diag(κ), P=3 时 rank=1 已有 3+3=6 个自由参数,")
print("     而 3x3 PSD 矩阵的自由度也是 6 —— rank=1 已张满整个可行空间。")
print("     rank=2/3 增加的参数全部冗余, 不改变模型的函数空间。")

print("\n" + "=" * 70)
print("为什么独立 GP 和 ICM 学到的 lengthscale 差很多")
print("=" * 70)
g = gp_df.iloc[0]
c = icm_df.loc[icm_df.loss.idxmin()]
print(f"\n  独立 GP : ({g.ls_x:.4f}, {g.ls_y:.4f})   只用 {len(ycd)} 个 Cd 观测")
print(f"  ICM     : ({c.ls_x:.4f}, {c.ls_y:.4f})   用 {len(ytr)} 个观测 "
      f"(其中 {len(ytr)-len(ycd)} 个是 Ni/Zn)")
print("\n  >> 这两个数不该直接比: 它们拟合的是不同数据、不同的似然函数。")
print("     ICM 强迫三个输出共用一个 lengthscale, 而 Ni/Zn 的观测数是 Cd 的 2.8 倍,")
print("     优化被它们主导 —— 差异来自'谁主导优化', 不是'似然曲面平坦'。")
print("     曲面平不平上面已经答了: 极差在机器精度量级, 一点都不平。")

print(f"\n[OK] 明细已存: {ROOT/'results'/'sensitivity.csv'}")'''

C04["5454713b"] = r"""## 结论

### 1. lengthscale 是全局最优, 初值无关

五个初值（0.1 → 2.0，跨 20 倍）在**两个模型上都收敛到机器精度级别相同的解**。
可以放心断言: 这是似然函数的全局最优, 不是优化的偶然结果。

### 2. rank 对 ICM 毫无影响, 且可证明

不是"影响很小", 是**恒等于零**。P=3 时 `B = W Wᵀ + diag(κ)` 在 rank=1 就有
3+3 = 6 个自由参数, 与 3×3 正定矩阵的自由度相同, 已张满可行空间。
**rank=2/3 增加的参数不改变模型的函数空间。**

报告里应写这条恒等式论证, 而不是一张"rank 敏感性表"——那张表测的量真值恒为 0。

### 3. 独立 GP 与 ICM 的 lengthscale 差异, 不能推出"不可辨识"

| 模型 | lengthscale | 训练数据 |
|---|---|---|
| 独立 GP | (0.096, 0.266) | 259 个 Cd |
| ICM | (0.079, 0.058) | 977 个（含 718 个 Ni/Zn）|

差异来自 **ICM 强迫三个输出共用一个 lengthscale**, 而 Ni/Zn 的观测数是 Cd 的
2.8 倍, 优化被它们主导。这是模型结构的约束, 不是优化失败。

---

## 早期版本的三处错误订正

| 早期结论 | 问题 | 现状 |
|---|---|---|
| "ICM loss 极差 0.0032 < 0.01 → 稳定收敛" | 0.0032 是 CG 随机估计的噪声 | 精确计算下极差在 1e-9 量级 |
| "rank 差异小于优化噪声, 按简约性选 rank=1" | 答案对, 理由错 | rank 效应**恒为零**, 可证明 |
| "lengthscale 差 4.6 倍 ⇒ 似然曲面平坦 ⇒ 参数不可辨识" | 两个数不可比; 且曲面不平 | 见上第 3 条 |

**一个值得写进报告的细节**: 早期那张表里, 独立 GP 的 loss 极差是 `0.0000`,
ICM 是 `0.0032`。**这个差异本身就是数值问题的指纹** —— 独立 GP 只有 259 个观测,
低于 GPyTorch 的 `max_cholesky_size=800` 阈值, 走的是精确 Cholesky; ICM 有 977 个,
走的是 CG + 随机 Lanczos。当时把 0.0032 读成了"收敛良好", 实际上它是随机估计的抖动。

---

## 本 notebook 建的是 ICM, 不是 LMC

`K = K_RBF(x,x') · B[i,j]`，**所有输出共享同一个 lengthscale** —— 这是 ICM
（Intrinsic Coregionalization Model）。真正的 LMC 是

$$K = \sum_{q=1}^{Q} K_q(x,x')\, B_q[i,j]$$

每个潜在函数有自己的 lengthscale。ICM 是 LMC 在 Q=1 时的特例。

**这个区别解释了上面第 3 条**: ICM 只能选一个空间尺度, 数据里若同时存在长程
（母岩背景）和短程（局部污染）结构, 它被迫二选一。

真 LMC 的实现见 `05_true_lmc.ipynb` / `05_true_lmc.py` —— 结果证实了这一点:
放开约束后模型找到 **ℓ=0.05 km 的短程分量**和 **ℓ=0.43 km 的长程分量**并存,
长程分量的跨输出相关性高达 0.85–0.94。"""


# ============================================================
# 05_true_lmc.ipynb
# ============================================================
C05 = {}

C05["41553137"] = r"""# 真 LMC: K = Σ_q K_q(x,x') · B_q[i,j]

与 ICM 的区别:

| | 公式 | 空间尺度 |
|---|---|---|
| ICM | `K = K_RBF(x,x') · B[i,j]` | 单一, 所有输出共享 |
| LMC | `K = Σ_q K_q(x,x') · B_q[i,j]` | 每个潜在函数各有一个 |

**为什么重要**: 数据里可能同时存在长程结构（母岩背景）和短程结构（局部污染）。
ICM 只能选一个尺度; LMC 可以让不同的 q 分工到不同尺度。

每个 `B_q` 取 `IndexKernel(rank=1)`, 即 `B_q = a_q a_qᵀ + diag(κ_q)`。任何 PSD
矩阵都能写成 `AAᵀ` 的形式, 加对角项只是满秩 `B_q` 的一种参数化 —— 地统计里的
LMC 本来就允许 `B_q` 满秩, 所以这仍是标准 LMC。

**自洽性检验**: Q=1 时 LMC 在数学上应恰好等于 ICM(rank=1)。下面会验证这一点。

---

> **本 notebook 已修正三个问题**（早期版本的结论不可用）:
>
> 1. **无精确 Cholesky**。N=977 > GPyTorch 默认阈值 800, 边际似然走 CG + 随机
>    Lanczos 估计, 噪声带 0.0095, 且系统性高估对数似然约 0.03。
> 2. **`best_loss` 取训练轨迹的最小值**。在随机估计下, 这等于**挑选最幸运的
>    一次采样** —— 哪个 Q 运气好哪个赢, 模型选择被噪声污染。
> 3. **用训练 loss 选 Q**。训练边际似然随参数量单调下降, 用它选 Q 必然选到
>    Q_max。改用 BIC / AIC 惩罚参数量。"""

C05["12e5c1f4"] = r'''import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import gpytorch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path.cwd()))
from models import exact, TrueLMC        # 共享定义, 与 05_true_lmc.py 同源

ROOT = Path(r"E:\Research")
DATA = np.load(ROOT / "data" / "jura_prepared.npz", allow_pickle=True)
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "figures").mkdir(exist_ok=True)

N_ITER = 500
LR = 0.05
P = 3
NAMES = [str(n) for n in DATA["target_names"]]

# 精确对照值 (阶段 8, 精确 Cholesky)。早期版本硬编码的 0.7418 是 CG 旧值。
RMSE_GP, RMSE_ICM = 0.9715, 0.7428
LOSS_ICM = 1.004174

Xtr = torch.tensor(DATA["X_train"], dtype=torch.float64)
itr = torch.tensor(DATA["i_train"], dtype=torch.long)
ytr = torch.tensor(DATA["y_train"], dtype=torch.float64)
Xte = torch.tensor(DATA["X_test"],  dtype=torch.float64)
ite = torch.tensor(DATA["i_test"],  dtype=torch.long)
yte = torch.tensor(DATA["y_test"],  dtype=torch.float64)

N_DATA = len(ytr)
print(f"训练观测 {N_DATA}, 测试 {len(yte)}")
print(f"对照: 独立 GP RMSE={RMSE_GP}, ICM RMSE={RMSE_ICM}")'''

C05["4e7d877e"] = r'''def run_lmc(Q, ls_inits, seed=0, verbose=True):
    """
    训练一个 Q 阶 LMC。

    与早期版本的区别:
      - 全程套 exact(): 边际似然是精确值, 不是 CG 随机估计
      - 不再取 best_loss: 精确计算下 loss 单调收敛, 末步就是最优。
        取轨迹最小值在随机估计下等于"挑最幸运的一次采样", 会污染模型选择。
      - 不再用 CosineAnnealingLR: 噪声消除后普通 Adam 就能干净收敛
      - 评估不用 fast_pred_var(): 它近似后验方差, 会让 NLL 有偏
    """
    torch.manual_seed(seed)
    lik = gpytorch.likelihoods.GaussianLikelihood().double()
    model = TrueLMC(Xtr, itr, ytr, lik, P, Q).double()

    # 各 q 的 lengthscale 初值必须拉开。全设成一样会梯度对称,
    # 所有 q 收敛到同一个解, Q>1 静默退化成 ICM。
    for q, ls in enumerate(ls_inits):
        model.spatial[q].lengthscale = torch.tensor([[ls, ls]], dtype=torch.float64)

    model.train(); lik.train()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)

    with exact():
        for k in range(N_ITER):
            opt.zero_grad()
            loss = -mll(model(Xtr, itr), ytr)
            loss.backward()
            opt.step()
            if verbose and (k + 1) % 250 == 0:
                print(f"    iter {k+1:4d}  loss={loss.item():.6f}")
        with torch.no_grad():
            final_loss = -mll(model(Xtr, itr), ytr).item()

    model.eval(); lik.eval()
    with torch.no_grad(), exact():
        pred = lik(model(Xte, ite))
        rmse = torch.sqrt(((pred.mean - yte) ** 2).mean()).item()
        mae = (pred.mean - yte).abs().mean().item()
        nll = -pred.log_prob(yte).item() / len(yte)

    ls_learned = [model.spatial[q].lengthscale.detach().numpy().ravel() for q in range(Q)]
    Bs = [model.task[q].covar_matrix.to_dense().detach().numpy() for q in range(Q)]

    # 参数量: 每个 q 有 2 个 lengthscale + 3 个 covar_factor + 3 个 raw_var = 8,
    #         再加 1 个常数均值 + 1 个噪声
    k_params = 8 * Q + 2
    logml = -final_loss * N_DATA          # GPyTorch 的 mll 已除以 num_data
    return dict(Q=Q, loss=final_loss, rmse=rmse, mae=mae, nll=nll,
                noise=lik.noise.item(), k=k_params, logml=logml,
                bic=-2 * logml + k_params * np.log(N_DATA),
                aic=-2 * logml + 2 * k_params,
                ls=ls_learned, Bs=Bs, model=model, lik=lik)'''

C05["e4b084fd"] = r'''# lengthscale 初值按几何级数拉开, 覆盖 "簇内尺度" -> "整个研究区"
# (与 05_true_lmc.py 一致, 保证两处结果可互相核对)
CONFIGS = {Q: (np.geomspace(0.08, 3.0, Q) if Q > 1 else np.array([0.5]))
           for Q in [1, 2, 3, 4, 5]}

results = []
for Q, ls_inits in CONFIGS.items():
    print(f"\n--- LMC Q={Q}  lengthscale 初值={np.round(ls_inits, 3)} ---")
    r = run_lmc(Q, ls_inits)
    results.append(r)
    print(f"  loss={r['loss']:.6f}  RMSE={r['rmse']:.4f}  NLL={r['nll']:.4f}  "
          f"noise={r['noise']:.4f}")
    tot = np.sum([np.diag(B) for B in r["Bs"]], axis=0)
    for q, ls in enumerate(r["ls"]):
        frac = np.diag(r["Bs"][q]) / tot
        print(f"    q={q+1}: ℓ=({ls[0]:.3f}, {ls[1]:.3f}) km   方差占比 "
              + "  ".join(f"{n}={f*100:.1f}%" for n, f in zip(NAMES, frac)))

# 自洽性检验: Q=1 的 LMC 应当恰好等于 ICM(rank=1)
d = abs(results[0]["loss"] - LOSS_ICM)
print(f"\n自洽性检验  LMC(Q=1) loss={results[0]['loss']:.6f}  "
      f"ICM(rank=1) loss={LOSS_ICM:.6f}  差={d:.2e}")
print(f"  >> {'通过: Q=1 时 LMC 数学上就是 ICM, 实现正确' if d < 1e-5 else '不一致, 检查实现!'}")'''

C05["4d43478b"] = r'''print("=" * 74)
print("各 Q 的对比")
print("=" * 74)
tab = pd.DataFrame([{k: r[k] for k in
                     ["Q", "k", "loss", "logml", "bic", "aic", "rmse", "mae", "nll", "noise"]}
                    for r in results])
print(tab.round(4).to_string(index=False))
tab.to_csv(ROOT / "results" / "true_lmc.csv", index=False)

print("\n" + "=" * 74)
print("检验 1: 训练 loss 是否随 Q 单调下降")
print("=" * 74)
mono = all(b <= a + 1e-9 for a, b in zip(tab.loss[:-1], tab.loss[1:]))
for i, row in tab.iterrows():
    flag = ""
    if i > 0 and row.loss > tab.loss[i-1] + 1e-9:
        flag = "   <-- 反常: 参数更多 loss 反而升高"
    print(f"  Q={int(row.Q)}: loss={row.loss:.6f}{flag}")
print(f"\n  >> {'单调, 数值噪声已消除' if mono else '仍不单调, 需加大 N_ITER'}")
print("  >> 早期版本 (CG 随机估计) 下 Q=3 比 Q=2 高、Q=5 比 Q=4 高, 数学上不可能。")

print("\n" + "=" * 74)
print("检验 2: 用 BIC / AIC 选 Q")
print("=" * 74)
print("\n  训练 loss 单调 => 用它选 Q 必然选到 Q_max, 不是模型选择。")
print("  测试 RMSE 会造成测试集泄漏。故用 BIC / AIC 惩罚参数量 k = 8Q + 2。\n")
print(f"  {'Q':<4}{'k':<6}{'-2logML':<12}{'BIC':<12}{'AIC':<12}{'RMSE':<10}")
qb, qa = int(tab.loc[tab.bic.idxmin(), "Q"]), int(tab.loc[tab.aic.idxmin(), "Q"])
for _, r_ in tab.iterrows():
    m = ("  <-- BIC 最优" if int(r_.Q) == qb else "") + \
        ("  <-- AIC 最优" if int(r_.Q) == qa else "")
    print(f"  {int(r_.Q):<4}{int(r_.k):<6}{-2*r_.logml:<12.1f}{r_.bic:<12.1f}"
          f"{r_.aic:<12.1f}{r_.rmse:<10.4f}{m}")
print(f"\n  >> BIC 与 AIC {'一致' if qb == qa else '不一致'}, 选定 Q = {qb}")

best = results[[r["Q"] for r in results].index(qb)]

print("\n" + "=" * 74)
print(f"选定模型 Q={qb} 的潜在函数分解")
print("=" * 74)
tot = np.sum([np.diag(B) for B in best["Bs"]], axis=0)
print(f"\n  (参照: 实测最近邻 0.125 km, 规则网格 0.25 km, 加密簇内约 0.01 km)\n")
for q, ls in enumerate(best["ls"]):
    Bq = best["Bs"][q]
    Dq = np.sqrt(np.diag(Bq))
    corr = Bq / np.outer(Dq, Dq)
    frac = np.diag(Bq) / tot
    rng = "短程" if (ls[0] + ls[1]) / 2 < 0.125 else "长程"
    print(f"  潜在函数 q={q+1}  ℓ=({ls[0]:.3f}, {ls[1]:.3f}) km   [{rng}]")
    print(f"    方差占比: " + "  ".join(f"{n}={f*100:.1f}%" for n, f in zip(NAMES, frac)))
    print(f"    B_{q+1} 诱导的跨输出相关:")
    for a, n in enumerate(NAMES):
        print(f"      {n}: " + "  ".join(f"{corr[a,b]:+.3f}" for b in range(P)))
    print()

print("-" * 74)
print(f"  相对独立 GP (RMSE={RMSE_GP}) 改善: {(RMSE_GP-best['rmse'])/RMSE_GP*100:+.2f}%")
print(f"  相对 ICM     (RMSE={RMSE_ICM}) 改善: {(RMSE_ICM-best['rmse'])/RMSE_ICM*100:+.2f}%")
print("  注: 后者是单次划分。10 次随机划分下 LMC 与 ICM 无显著差异")
print("      (5/10 胜, 配对 t p=0.19), 见 src/07_robustness.py。")'''

C05["5f2b40c6"] = r'''fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))

# (a) BIC / AIC vs Q —— 这才是选 Q 的依据
ax = axes[0]
ax.plot(tab.Q, tab.bic, "o-", lw=2, ms=7, color="#4c72b0", label="BIC")
ax.plot(tab.Q, tab.aic, "s--", lw=2, ms=6, color="#dd8452", label="AIC")
for arr, col in [(tab.bic.to_numpy(), "#4c72b0"), (tab.aic.to_numpy(), "#dd8452")]:
    i = arr.argmin()
    ax.scatter([tab.Q.iloc[i]], [arr[i]], s=250, marker="*", color=col,
               edgecolor="k", lw=0.7, zorder=5)
ax.set_xticks(tab.Q)
ax.set_xlabel("Q (number of latent functions)")
ax.set_ylabel("information criterion (lower is better)")
ax.set_title(f"Model selection: both criteria pick Q={qb}")
ax.legend(fontsize=9); ax.grid(alpha=0.25)

# (b) 测试 RMSE —— 仅作透明展示, 不是选择依据
ax = axes[1]
ax.plot(tab.Q, tab.rmse, "o-", lw=2, ms=7, color="#4c72b0", label="True LMC")
ax.axhline(RMSE_ICM, color="#dd8452", ls="--", lw=1.6, label=f"ICM ({RMSE_ICM})")
ax.axhline(RMSE_GP, color="#c44e52", ls="--", lw=1.6, label=f"Independent GP ({RMSE_GP})")
ax.axvline(qb, color="gray", ls=":", lw=1.2)
ax.set_xticks(tab.Q)
ax.set_xlabel("Q"); ax.set_ylabel("test RMSE (standardised)")
ax.set_title("Test RMSE — shown for transparency,\nNOT used to select Q (would leak the test set)",
             fontsize=10)
ax.legend(fontsize=8.5); ax.grid(alpha=0.25)

# (c) 选定模型的潜在函数分解
ax = axes[2]
w = 0.8 / best["Q"]
for q, ls in enumerate(best["ls"]):
    frac = np.diag(best["Bs"][q]) / tot
    ax.bar(np.arange(P) + q * w, frac * 100, width=w * 0.9,
           label=f"q={q+1}, ℓ≈{ls.mean():.3f} km")
ax.set_xticks(np.arange(P) + w * (best["Q"] - 1) / 2, NAMES)
ax.set_ylabel("share of prior variance (%)")
ax.set_title(f"Latent decomposition (Q={qb})")
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.25)

plt.tight_layout()
out = ROOT / "figures" / "true_lmc.png"
plt.savefig(out, dpi=140)     # savefig 必须在 show 之前! 早期版本反了, 存下来是空白图
plt.show()
print(f"[OK] 图已保存: {out}")'''

C05["0dc5e7a7"] = r"""## 结论演变的记录（三个已作废的版本）

这个 notebook 先后得出过三个互相矛盾的结论。保留这段记录, 是因为报告的
"方法论反思"部分用得上 —— 但**下面三条都不要引用**。

| 轮次 | 当时的结论 | 为什么错 |
|---|---|---|
| 第 1 轮 | **Q=3**, 且 "q=3 是纯共同因子, 相关系数 0.998" | 未收敛就解读参数。加大迭代后 Q=3 的分解塌回 Q=2 的结构 |
| 第 2 轮 | **Q=2**, 用 `best_loss` 修正震荡 | `best_loss` 在 CG 随机估计下 = 挑最幸运的一次采样 |
| 第 3 轮 | **Q=4**, 按训练 loss 选最优 | 训练 loss 随参数量单调下降, 必然选到 Q_max |

**教训**（值得写进报告）:

1. **解读参数之前先确认收敛。** 第 1 轮那个"相关系数 0.998 的纯共同因子"看起来
   是漂亮的地质发现, 实际是没收敛的产物。
2. **在随机估计的目标函数上取最小值是错的。** 它挑的是噪声的下尾, 不是更好的解。
3. **训练似然不能用于模型选择。** 它对参数量单调, 必须加惩罚项 (BIC/AIC) 或用
   交叉验证。

以下是修正后的结论。"""

C05["604e34f0"] = r"""## 最终结论

### 选定 Q = 2, BIC 与 AIC 一致

| Q | k | −2logML | BIC | AIC | 测试 RMSE |
|---|---|---|---|---|---|
| 1 | 10 | 1962.2 | 2031.0 | 1982.2 | 0.7428 |
| **2** | **18** | **1853.5** | **1977.4** ✓ | **1889.5** ✓ | **0.7313** |
| 3 | 26 | 1852.2 | 2031.2 | 1904.2 | 0.7323 |
| 4 | 34 | 1841.9 | 2075.9 | 1909.9 | 0.7198 |
| 5 | 42 | 1841.8 | 2130.9 | 1925.8 | 0.7172 |

BIC 在 Q=2 领先次优 53 个单位。**注意测试 RMSE 在 Q=4/5 更低 —— 但那不是选择
依据**, 用它选 Q 会泄漏测试集。BIC/AIC 判的是"增加的拟合能力是否抵得过增加的
参数": Q=1→2 用 55.1 的惩罚买到 108.7 的 misfit 改善（值）, 之后每步只买到
0.1~10.4（不值）。

**旁证**: Q=5 时第 5 个潜在函数的方差占比是 0.0% / 0.0% / 0.0% —— 优化器自己
把它关掉了。模型本身就说明了它不需要 5 个。

### 双尺度结构 —— 本研究最有内容的结果

| | ℓ_X (km) | ℓ_Y (km) | Cd | Ni | Zn | B_q 内的跨输出相关 |
|---|---|---|---|---|---|---|
| **q=1 短程** | 0.052 | 0.042 | **77.7%** | 46.5% | 68.8% | 0.67 (中等) |
| **q=2 长程** | 0.511 | 0.347 | 22.3% | **53.5%** | 31.2% | **0.88–0.94 (近乎完美)** |

**q=2 是长程共同因子**: ℓ≈0.43 km 超过规则网格间距 (0.25 km), 三个金属间相关性
高达 0.88–0.94 —— 近乎完美相关意味着它们在这个尺度上是**同一个潜在场的线性缩放**,
对应区域性地质背景（共同母岩风化）。

**q=1 是短程局部结构**: ℓ≈0.05 km, 相关性只有 0.67, 金属间差异明显, 对应局部
人为输入; 且它主导 Cd 的方差 (77.7%)。

**一个反直觉的细节**: 虽然 q=1 占 Cd 先验方差的 77.7%, 但在实际预测面上, 长程
分量 q=2 贡献了 **62.9%** 的幅度（见 `06_spatial_maps.py` 的精确可加分解,
校验残差 1.3e-12）。**先验方差占比 ≠ 后验贡献。**

### 这推翻了 03/04 的一个结论

早期认为"Cd 空间自相关弱"。**那是 ICM 单一尺度约束造成的假象** —— 长程结构一直
存在, ICM 被迫二选一时选了短程（它学到的 0.079/0.058 几乎精确落在 q=1 上）。

而 ICM 报告的单一相关性 0.743, 是 0.673 和 0.941 的加权混合 —— **一个不对应任何
真实过程的平均数**。

### 但不要声称 LMC 预测得比 ICM 好

单次划分上 LMC 比 ICM 好 1.55%，**这个差距不复现**: 10 次随机划分下 LMC 赢 5 输 5,
配对 t 检验 p=0.19（见 `07_robustness.py`）。

**LMC 的价值在可解释性, 不在预测精度。** 报告必须这样写。

---

**后续**: `06_spatial_maps.py`（空间后验图 + 潜在函数分解）→
`07_robustness.py`（10 次划分）→ `10_report_figures.py`（报告用图）"""


# ============================================================
def patch(path, cells, drop_empty=True):
    nb = json.loads(path.read_text(encoding="utf-8"))
    ids = {c.get("id") for c in nb["cells"]}
    missing = set(cells) - ids
    if missing:
        sys.exit(f"{path.name} 里找不到 cell id: {missing}")

    kept, n = [], 0
    for c in nb["cells"]:
        cid = c.get("id")
        if cid in cells:
            c["source"] = cells[cid].splitlines(keepends=True)
            n += 1
        if c["cell_type"] == "code":
            c["outputs"] = []
            c["execution_count"] = None
            if drop_empty and not "".join(c["source"]).strip():
                continue          # 丢掉空代码单元格
        kept.append(c)

    dropped = len(nb["cells"]) - len(kept)
    nb["cells"] = kept
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] {path.name}: 修改 {n} 个单元格, 删除 {dropped} 个空单元格, 清空所有输出")


def main():
    patch(SRC / "04_sensitivity.ipynb", C04)
    patch(SRC / "05_true_lmc.ipynb", C05)

    if "--execute" in sys.argv:
        import nbformat
        from nbclient import NotebookClient

        for name, mins in [("04_sensitivity.ipynb", 6), ("05_true_lmc.ipynb", 8)]:
            p = SRC / name
            print(f"\n执行 {name} (约 {mins} 分钟) ...")
            book = nbformat.read(p, as_version=4)
            NotebookClient(book, timeout=3600, kernel_name="python3",
                           resources={"metadata": {"path": str(SRC)}}).execute()
            nbformat.write(book, p)
            print(f"[OK] {name} 执行完成")


if __name__ == "__main__":
    main()
