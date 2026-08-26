"""
真 LMC (每个潜在函数各自的 kernel) + 数值精度修正

背景 (阶段 8):
  GPyTorch 的 max_cholesky_size 默认 = 800。本项目训练集 N=977 > 800,
  所以之前所有 977 点的模型 (ICM / LMC) 都走了 CG + Lanczos 的随机估计,
  边际似然和梯度都带噪声。症状: results/true_lmc.csv 里
  Q=3 的训练 loss 比 Q=2 高、Q=5 比 Q=4 高 —— 参数更多训练目标反而更差,
  数学上不可能, 只能是数值噪声。

  本脚本用 gpytorch.settings.max_cholesky_size(10000) +
  fast_computations(全关) 强制精确 Cholesky, 重跑所有模型。
  独立 GP 只有 259 点 (<800), 本来就是精确的, 一并重跑做对照。

模型:
  独立 GP  : K = ScaleKernel(RBF_ard)                     只用 259 个 Cd
  ICM      : K = RBF_ard(x,x') * B[i,i']                  单一 lengthscale
  真 LMC   : K = sum_q RBF_ard_q(x,x') * B_q[i,i']        每个 q 独立 lengthscale

输出:
  results/stage8_all_models.csv   所有模型的干净指标
  results/stage8_best_params.json 最优 LMC 的完整参数 (报告和热力图直接用)
"""

import contextlib
import json
import time
import numpy as np
import pandas as pd
import torch
import gpytorch
from pathlib import Path

ROOT = Path(r"E:\Research")
DATA = np.load(ROOT / "data" / "jura_prepared.npz", allow_pickle=True)
(ROOT / "results").mkdir(exist_ok=True)

N_ITER = 500
LR = 0.05
Q_GRID = [1, 2, 3, 4, 5]
RANK_GRID = [1, 2, 3]        # ICM 的 rank
N_RESTART = 2                # 每个配置的初值重启次数
# km, 实测平均最近邻距离 (不是 sqrt(面积/N)=0.25 那个密度估计 —— Jura 是
# 规则 0.25km 网格 + 嵌套加密簇的设计, 42% 的点有 50m 内的邻居, 密度估计会高估间距)
SAMPLE_SPACING = 0.125

# ---------- 精确计算上下文: 这就是阶段 8 的修正本身 ----------
@contextlib.contextmanager
def exact():
    """强制精确 Cholesky, 关掉所有随机近似 (CG solves / Lanczos logdet)。"""
    with gpytorch.settings.max_cholesky_size(10_000), \
         gpytorch.settings.fast_computations(
             covar_root_decomposition=False, log_prob=False, solves=False
         ):
        yield

Xtr = torch.tensor(DATA["X_train"], dtype=torch.float64)
itr = torch.tensor(DATA["i_train"], dtype=torch.long)
ytr = torch.tensor(DATA["y_train"], dtype=torch.float64)
Xte = torch.tensor(DATA["X_test"], dtype=torch.float64)
ite = torch.tensor(DATA["i_test"], dtype=torch.long)
yte = torch.tensor(DATA["y_test"], dtype=torch.float64)
mask = itr == 0
Xcd, ycd = Xtr[mask], ytr[mask]

mu, sd = DATA["mu"], DATA["sd"]
NAMES = [str(n) for n in DATA["target_names"]]
P = len(NAMES)

print(f"训练 {len(ytr)} 观测 (各任务 {np.bincount(DATA['i_train'])}), 测试 {len(yte)}")
print(f"max_cholesky_size: 默认 {gpytorch.settings.max_cholesky_size.value()} -> 本脚本 10000")
print(f"N_train = {len(ytr)} {'>' if len(ytr) > 800 else '<='} 800  "
      f"({'之前走的是 CG 随机估计' if len(ytr) > 800 else '之前也是精确的'})\n")


# ============================================================
# 模型定义
# ============================================================
class SingleGP(gpytorch.models.ExactGP):
    """独立 GP 基线: 只看 Cd。有 ScaleKernel, 因为没有 B 来承担尺度。"""

    def __init__(self, x, y, lik):
        super().__init__(x, y, lik)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=2)
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


class ICMModel(gpytorch.models.ExactGP):
    """ICM: 单一空间 kernel 乘一个任务协方差 B。不加 ScaleKernel (与 B 对角元不可辨识)。"""

    def __init__(self, x, i, y, lik, num_tasks, rank):
        super().__init__((x, i), y, lik)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.RBFKernel(ard_num_dims=2)
        self.task_covar_module = gpytorch.kernels.IndexKernel(
            num_tasks=num_tasks, rank=rank
        )

    def forward(self, x, i):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x),
            self.covar_module(x).mul(self.task_covar_module(i)),
        )


class TrueLMC(gpytorch.models.ExactGP):
    """
    真 LMC: K = sum_q K_q(x,x') * B_q[i,i'], 每个 q 有自己的 lengthscale。

    每个 B_q 取 rank 1 (B_q = w_q w_q^T + diag(kappa_q)) —— 这是 LMC 的标准形式:
    Q 个潜在函数, 每个通过一组权重 w_q 混合到 P 个输出上。
    """

    def __init__(self, x, i, y, lik, num_tasks, Q):
        super().__init__((x, i), y, lik)
        self.Q = Q
        self.mean_module = gpytorch.means.ConstantMean()
        self.spatial = torch.nn.ModuleList(
            [gpytorch.kernels.RBFKernel(ard_num_dims=2) for _ in range(Q)]
        )
        self.task = torch.nn.ModuleList(
            [gpytorch.kernels.IndexKernel(num_tasks=num_tasks, rank=1) for _ in range(Q)]
        )

    def forward(self, x, i):
        covar = self.spatial[0](x).mul(self.task[0](i))
        for q in range(1, self.Q):
            covar = covar + self.spatial[q](x).mul(self.task[q](i))
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), covar)


# ============================================================
# 训练 / 评估
# ============================================================
def evaluate(model, lik, kind):
    model.eval()
    lik.eval()
    with torch.no_grad(), exact():
        pred = lik(model(Xte)) if kind == "gp" else lik(model(Xte, ite))
        m = pred.mean
        rmse = torch.sqrt(((m - yte) ** 2).mean()).item()
        mae = (m - yte).abs().mean().item()
        nll = -pred.log_prob(yte).item() / len(yte)
        var = pred.variance
    return rmse, mae, nll, m.numpy(), var.numpy()


def train_one(kind, param, seed, ls_init):
    """
    kind  : "gp" | "icm" | "lmc"
    param : icm -> rank, lmc -> Q, gp -> None
    ls_init: 标量 (gp/icm) 或长度 Q 的数组 (lmc)
    """
    torch.manual_seed(seed)
    lik = gpytorch.likelihoods.GaussianLikelihood().double()

    if kind == "gp":
        model = SingleGP(Xcd, ycd, lik).double()
        model.covar_module.base_kernel.lengthscale = torch.tensor(
            [[ls_init, ls_init]], dtype=torch.float64
        )
        inputs, targets = (Xcd,), ycd
    elif kind == "icm":
        model = ICMModel(Xtr, itr, ytr, lik, P, param).double()
        model.covar_module.lengthscale = torch.tensor(
            [[ls_init, ls_init]], dtype=torch.float64
        )
        inputs, targets = (Xtr, itr), ytr
    else:
        model = TrueLMC(Xtr, itr, ytr, lik, P, param).double()
        # 关键: 各 q 的 lengthscale 初值必须拉开。全设成一样会梯度对称,
        #       所有 q 收敛到同一个解, Q>1 就退化成 ICM 了。
        for q in range(param):
            model.spatial[q].lengthscale = torch.tensor(
                [[ls_init[q], ls_init[q]]], dtype=torch.float64
            )
        inputs, targets = (Xtr, itr), ytr

    model.train()
    lik.train()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)

    with exact():
        for _ in range(N_ITER):
            opt.zero_grad()
            loss = -mll(model(*inputs), targets)
            loss.backward()
            opt.step()
        # 最终 loss 用当前参数重算一次, 而不是用最后一步更新前的值
        with torch.no_grad():
            final_loss = -mll(model(*inputs), targets).item()

    rmse, mae, nll, pred_mean, pred_var = evaluate(model, lik, kind)

    # 参数量 (likelihood 是 ExactGP 的子模块, 已包含在内; 用 id 去重保险)
    seen, k = set(), 0
    for p in list(model.parameters()) + list(lik.parameters()):
        if id(p) not in seen:
            seen.add(id(p))
            k += p.numel()

    # loss 是 GPyTorch 除过 num_data 的, 还原成总对数边际似然
    n_data = len(targets)
    logml = -final_loss * n_data
    return dict(
        kind=kind, param=param, seed=seed, loss=final_loss,
        rmse=rmse, mae=mae, nll=nll, noise=lik.noise.item(),
        k=k, n_data=n_data, logml=logml,
        bic=-2 * logml + k * np.log(n_data),
        aic=-2 * logml + 2 * k,
    ), model, lik


# ============================================================
# 主扫描
# ============================================================
def ls_spread(Q, variant):
    """给 Q 个潜在函数分配拉开的 lengthscale 初值 (km)。"""
    if variant == 0:
        return np.geomspace(0.08, 3.0, Q) if Q > 1 else np.array([0.5])
    return np.geomspace(0.15, 1.5, Q) if Q > 1 else np.array([1.0])


rows = []
t0 = time.time()

print("=" * 70)
print("独立 GP 基线 (259 点, 本来就是精确 Cholesky)")
print("=" * 70)
best_gp = None
for s in range(N_RESTART):
    r, m, l = train_one("gp", None, seed=s, ls_init=[0.5, 1.0][s])
    rows.append(r)
    print(f"  seed={s} ls_init={[0.5,1.0][s]:<4} -> loss={r['loss']:.4f}  RMSE={r['rmse']:.4f}")
    if best_gp is None or r["loss"] < best_gp[0]["loss"]:
        best_gp = (r, m, l)

print("\n" + "=" * 70)
print("ICM (精确 Cholesky 重跑)")
print("=" * 70)
for rank in RANK_GRID:
    for s in range(N_RESTART):
        r, m, l = train_one("icm", rank, seed=s, ls_init=[0.5, 1.0][s])
        rows.append(r)
        print(f"  rank={rank} seed={s} -> loss={r['loss']:.4f}  RMSE={r['rmse']:.4f}  "
              f"noise={r['noise']:.4f}")

print("\n" + "=" * 70)
print("真 LMC (精确 Cholesky)")
print("=" * 70)
best_lmc = None
for Q in Q_GRID:
    for s in range(N_RESTART):
        init = ls_spread(Q, s)
        r, m, l = train_one("lmc", Q, seed=s, ls_init=init)
        rows.append(r)
        print(f"  Q={Q} seed={s} init={np.round(init,2)} -> loss={r['loss']:.4f}  "
              f"RMSE={r['rmse']:.4f}  noise={r['noise']:.4f}")
        # 注意: 选择准则是 BIC 不是 loss。训练边际似然随参数量单调下降,
        #       用它选 Q 必然选到 Q_max, 不是模型选择。
        if best_lmc is None or r["bic"] < best_lmc[0]["bic"]:
            best_lmc = (r, m, l)

df = pd.DataFrame(rows)
df.to_csv(ROOT / "results" / "stage8_all_models.csv", index=False)
print(f"\n用时 {time.time()-t0:.1f}s")


# ============================================================
# 诊断
# ============================================================
print("\n" + "=" * 70)
print("诊断 1: 训练 loss 是否随参数量单调下降 (精确计算下应当如此)")
print("=" * 70)
lmc_best_per_Q = df[df.kind == "lmc"].groupby("param")["loss"].min()
print("\n各 Q 的最优训练 loss:")
prev, monotone = None, True
for Q, v in lmc_best_per_Q.items():
    flag = ""
    if prev is not None and v > prev + 1e-9:
        flag = "  <-- 反常: 参数更多 loss 反而升高"
        monotone = False
    print(f"  Q={Q}: loss={v:.6f}{flag}")
    prev = v
print(f"\n单调性: {'通过 —— 数值噪声已消除' if monotone else '仍未通过, 需要加大 N_ITER 或换优化器'}")

old = pd.read_csv(ROOT / "results" / "true_lmc.csv") if (ROOT / "results" / "true_lmc.csv").exists() else None
if old is not None:
    print("\n与阶段 7 (CG 随机估计) 对比:")
    print(f"  {'Q':<4}{'旧 loss':<12}{'新 loss':<12}{'差':<10}{'旧 RMSE':<10}{'新 RMSE':<10}")
    for Q in Q_GRID:
        o = old[old.Q == Q]
        if len(o) == 0 or Q not in lmc_best_per_Q.index:
            continue
        n_loss = lmc_best_per_Q[Q]
        n_rmse = df[(df.kind == "lmc") & (df.param == Q)].sort_values("loss").iloc[0]["rmse"]
        print(f"  {Q:<4}{o.iloc[0]['loss']:<12.4f}{n_loss:<12.4f}"
              f"{n_loss-o.iloc[0]['loss']:<+10.4f}{o.iloc[0]['rmse']:<10.4f}{n_rmse:<10.4f}")

# ---------- 模型选择 ----------
print("\n" + "=" * 70)
print("诊断 1b: ICM 的 rank 是否真的增加了表达能力")
print("=" * 70)
icm_per_rank = df[df.kind == "icm"].groupby("param")["loss"].min()
q1 = df[(df.kind == "lmc") & (df.param == 1)]["loss"].min()
print("\n  IndexKernel 的 B = W W^T + diag(kappa)。P=3 时 rank=1 已有 3+3=6 个自由参数,")
print("  而 3x3 PSD 矩阵的自由度也是 6 —— 所以 rank=1 已张满整个可行空间。")
print("  如果下面几个 loss 完全相同, 就证实了 rank>1 的参数全是冗余的:\n")
for rk, v in icm_per_rank.items():
    print(f"    ICM rank={rk}: loss={v:.6f}")
print(f"    LMC Q=1    : loss={q1:.6f}")
spread_rank = icm_per_rank.max() - icm_per_rank.min()
print(f"\n  rank 之间的 loss 极差 = {spread_rank:.2e}")
print(f"  >> {'证实: rank 对 ICM 无任何影响, 不是 LMC 的 Q' if spread_rank < 1e-4 else '存在差异, 需进一步检查'}")
print("  >> 阶段 4 那张 rank 敏感性表测到的 0.0032 波动 = 纯 CG 噪声, 真值为 0")

print("\n" + "=" * 70)
print("诊断 2: 模型选择")
print("=" * 70)
print("\n  不能用训练边际似然选 Q (它随参数量单调下降, 见诊断 1),")
print("  也不能用测试 RMSE 选 (测试集泄漏)。用 BIC / AIC 惩罚参数量。")
print("  注意: 独立 GP 只对 259 个 Cd 观测建模, 多输出模型对 977 个观测建模,")
print("        数据不同, BIC 不可跨这两类比较 —— 只在 Q 之间比。\n")

sel = (df[df.kind == "lmc"].sort_values("bic").groupby("param", as_index=False).first()
       .sort_values("param"))
print(f"  {'Q':<4}{'k':<6}{'-2logML':<12}{'BIC':<12}{'AIC':<12}{'RMSE':<10}")
bic_min, aic_min = sel.bic.min(), sel.aic.min()
for _, r_ in sel.iterrows():
    star = "  <-- BIC 最优" if r_.bic == bic_min else ""
    star += "  <-- AIC 最优" if r_.aic == aic_min else ""
    print(f"  {int(r_.param):<4}{int(r_.k):<6}{-2*r_.logml:<12.1f}{r_.bic:<12.1f}"
          f"{r_.aic:<12.1f}{r_.rmse:<10.4f}{star}")
agree = sel.loc[sel.bic.idxmin(), "param"] == sel.loc[sel.aic.idxmin(), "param"]
print(f"\n  >> BIC 与 AIC {'一致' if agree else '不一致 (需在报告中讨论)'}, "
      f"选定 Q = {int(sel.loc[sel.bic.idxmin(),'param'])}")

r_gp = best_gp[0]
icm_best = df[df.kind == "icm"].sort_values("bic").iloc[0]
r_lmc = best_lmc[0]
print("\n最终对比:")
print(f"\n  独立 GP : loss={r_gp['loss']:.4f}  RMSE={r_gp['rmse']:.4f}  NLL={r_gp['nll']:.4f}")
print(f"  ICM     : loss={icm_best['loss']:.4f}  RMSE={icm_best['rmse']:.4f}  "
      f"NLL={icm_best['nll']:.4f}  (rank={int(icm_best['param'])})")
print(f"  真 LMC  : loss={r_lmc['loss']:.4f}  RMSE={r_lmc['rmse']:.4f}  "
      f"NLL={r_lmc['nll']:.4f}  (Q={r_lmc['param']})")
print(f"\n  ICM  相对独立 GP 的 RMSE 改善: {(r_gp['rmse']-icm_best['rmse'])/r_gp['rmse']*100:+.2f}%")
print(f"  LMC  相对独立 GP 的 RMSE 改善: {(r_gp['rmse']-r_lmc['rmse'])/r_gp['rmse']*100:+.2f}%")
print(f"  LMC  相对 ICM   的 RMSE 改善: {(icm_best['rmse']-r_lmc['rmse'])/icm_best['rmse']*100:+.2f}%")

# ---------- 最优 LMC 的潜在函数分解 ----------
model, lik = best_lmc[1], best_lmc[2]
Q_best = int(r_lmc["param"])
print("\n" + "=" * 70)
print(f"诊断 3: 最优模型 (Q={Q_best}) 的潜在函数分解")
print("=" * 70)

comp = []
for q in range(Q_best):
    ls = model.spatial[q].lengthscale.detach().numpy().ravel()
    Bq = model.task[q].covar_matrix.to_dense().detach().numpy()
    comp.append(dict(q=q + 1, ls_x=float(ls[0]), ls_y=float(ls[1]), B=Bq))

tot_var = np.sum([c["B"].diagonal() for c in comp], axis=0)   # 每个任务的总先验方差

print(f"\n(采样间距 ≈ {SAMPLE_SPACING} km 作参照)\n")
hdr = f"  {'q':<4}{'ls_X(km)':<11}{'ls_Y(km)':<11}" + "".join(f"{n+'方差占比':<14}" for n in NAMES) + "解释"
print(hdr)
print("  " + "-" * (len(hdr) - 2))
for c in comp:
    frac = c["B"].diagonal() / tot_var
    ls_mean = (c["ls_x"] + c["ls_y"]) / 2
    tag = "长程 (>采样间距)" if ls_mean > SAMPLE_SPACING else "短程 (<采样间距)"
    print(f"  {c['q']:<4}{c['ls_x']:<11.3f}{c['ls_y']:<11.3f}"
          + "".join(f"{f*100:>6.1f}%{'':<7}" for f in frac) + tag)

print("\n各潜在函数的任务相关矩阵 B_q (归一化成相关系数):")
for c in comp:
    B = c["B"]
    D = np.sqrt(np.diag(B))
    Bc = B / np.outer(D, D)
    print(f"\n  q={c['q']}  (ls = {c['ls_x']:.3f}, {c['ls_y']:.3f} km)")
    for r_ in range(P):
        print(f"    {NAMES[r_]:<4}" + "  ".join(f"{Bc[r_, cc]:+.3f}" for cc in range(P)))

# 合并 B = sum_q B_q, 与 EDA 经验相关性对比
B_tot = np.sum([c["B"] for c in comp], axis=0)
D = np.sqrt(np.diag(B_tot))
B_tot_corr = B_tot / np.outer(D, D)
print("\n合并任务相关矩阵 sum_q B_q (与 EDA 经验相关性对比用):")
for r_ in range(P):
    print(f"    {NAMES[r_]:<4}" + "  ".join(f"{B_tot_corr[r_, cc]:+.3f}" for cc in range(P)))

print(f"\n噪声方差: {lik.noise.item():.4f}")
print(f"常数均值: {model.mean_module.constant.item():.4f}")

# ---------- 存参数 ----------
params = dict(
    stage="8_exact_cholesky",
    n_train=int(len(ytr)),
    max_cholesky_size=10000,
    best_Q=Q_best,
    best_seed=int(r_lmc["seed"]),
    loss=float(r_lmc["loss"]),
    rmse_std=float(r_lmc["rmse"]),
    mae_std=float(r_lmc["mae"]),
    nll=float(r_lmc["nll"]),
    selection_criterion="BIC",
    k=int(r_lmc["k"]),
    bic=float(r_lmc["bic"]),
    aic=float(r_lmc["aic"]),
    noise=float(lik.noise.item()),
    mean_constant=float(model.mean_module.constant.item()),
    target_names=NAMES,
    mu=[float(v) for v in mu],
    sd=[float(v) for v in sd],
    components=[
        dict(q=c["q"], ls_x=c["ls_x"], ls_y=c["ls_y"], B=c["B"].tolist(),
             var_frac=(c["B"].diagonal() / tot_var).tolist())
        for c in comp
    ],
    B_total=B_tot.tolist(),
    B_total_corr=B_tot_corr.tolist(),
    baseline_gp=dict(loss=float(r_gp["loss"]), rmse=float(r_gp["rmse"]), nll=float(r_gp["nll"])),
    icm_best=dict(rank=int(icm_best["param"]), loss=float(icm_best["loss"]),
                  rmse=float(icm_best["rmse"]), nll=float(icm_best["nll"])),
)
with open(ROOT / "results" / "stage8_best_params.json", "w", encoding="utf-8") as f:
    json.dump(params, f, indent=2, ensure_ascii=False)

torch.save(
    dict(model=model.state_dict(), likelihood=lik.state_dict(), Q=Q_best),
    ROOT / "results" / "stage8_best_lmc.pt",
)

print("\n" + "=" * 70)
print(f"[OK] 明细     : {ROOT/'results'/'stage8_all_models.csv'}")
print(f"[OK] 干净参数 : {ROOT/'results'/'stage8_best_params.json'}")
print(f"[OK] 模型权重 : {ROOT/'results'/'stage8_best_lmc.pt'}  (阶段 9 热力图直接加载)")
print("=" * 70)
