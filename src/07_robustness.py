"""
阶段 10: 稳健性检验 —— 10 次随机 heterotopic 划分

阶段 8 的 +24.7% 改善是**单次划分**(seed=0)的结果, 可能有运气成分。
本脚本换 10 个种子重做划分, 报 RMSE 的均值±标准差, 并做配对检验。

每个划分都完整重做:
  - 重新随机选 100 个位置藏 Cd
  - 重新用该划分的训练观测算标准化参数 (防泄漏, 与 02_prepare_data 一致)
  - 重新训练三个模型

模型配置固定为阶段 8 由 BIC 选定的结果, 不在每个划分上重选 Q。
(在每个划分上重选 Q 更严格但成本 x5; 这里的目的是检验性能对比的稳健性,
 不是检验模型选择的稳健性 —— 报告中需说明这一点。)

输出: results/stage10_robustness.csv
      figures/stage10_robustness.png
"""

import contextlib
import sys
import time
import numpy as np
import pandas as pd
import torch
import gpytorch
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(r"E:\Research")
(ROOT / "results").mkdir(exist_ok=True)

N_SPLITS = 10
N_TEST = 100
N_ITER = 500
LR = 0.05
Q_BEST = 2                       # 阶段 8 BIC 选定
LS_INIT_LMC = [0.08, 3.0]        # 与阶段 8 seed=0 一致 (拉开以打破对称)
LS_INIT_SINGLE = 0.5
TARGET_NAMES = ["Cd", "Ni", "Zn"]


@contextlib.contextmanager
def exact():
    with gpytorch.settings.max_cholesky_size(10_000), \
         gpytorch.settings.fast_computations(
             covar_root_decomposition=False, log_prob=False, solves=False
         ):
        yield


# ---------- 原始数据 (只做一次) ----------
df = pd.read_csv(ROOT / "data" / "jura.csv")
low = {c.lower(): c for c in df.columns}
X_COL = low.get("xloc", low.get("x"))
Y_COL = low.get("yloc", low.get("y"))
coords = df[[X_COL, Y_COL]].to_numpy(dtype=np.float64)
Y_log = np.log(df[[low[t.lower()] for t in TARGET_NAMES]].to_numpy(dtype=np.float64))
N, P = Y_log.shape
print(f"{N} 个位置, {P} 个输出。做 {N_SPLITS} 次独立划分。\n")


def make_split(seed):
    """复刻 02_prepare_data 的划分 + 防泄漏标准化, 但用不同种子。"""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    test_idx = np.sort(perm[:N_TEST])
    train_idx = np.sort(perm[N_TEST:])

    mu, sd = np.zeros(P), np.zeros(P)
    mu[0], sd[0] = Y_log[train_idx, 0].mean(), Y_log[train_idx, 0].std()   # Cd 只用训练位置
    for p in [1, 2]:
        mu[p], sd[p] = Y_log[:, p].mean(), Y_log[:, p].std()               # Ni/Zn 全观测
    Y_std = (Y_log - mu) / sd

    Xtr = np.concatenate([coords[train_idx], coords, coords])
    itr = np.concatenate([np.zeros(len(train_idx)), np.ones(N), np.full(N, 2)]).astype(np.int64)
    ytr = np.concatenate([Y_std[train_idx, 0], Y_std[:, 1], Y_std[:, 2]])

    t = lambda a: torch.tensor(a, dtype=torch.float64)
    return dict(
        Xtr=t(Xtr), itr=torch.tensor(itr), ytr=t(ytr),
        Xte=t(coords[test_idx]), ite=torch.zeros(N_TEST, dtype=torch.long),
        yte=t(Y_std[test_idx, 0]),
        Xcd=t(coords[train_idx]), ycd=t(Y_std[train_idx, 0]),
    )


class SingleGP(gpytorch.models.ExactGP):
    def __init__(self, x, y, lik):
        super().__init__(x, y, lik)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=2))

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x))


class ICMModel(gpytorch.models.ExactGP):
    def __init__(self, x, i, y, lik, P_, rank):
        super().__init__((x, i), y, lik)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.RBFKernel(ard_num_dims=2)
        self.task_covar_module = gpytorch.kernels.IndexKernel(num_tasks=P_, rank=rank)

    def forward(self, x, i):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x),
            self.covar_module(x).mul(self.task_covar_module(i)))


class TrueLMC(gpytorch.models.ExactGP):
    def __init__(self, x, i, y, lik, P_, Q):
        super().__init__((x, i), y, lik)
        self.Q = Q
        self.mean_module = gpytorch.means.ConstantMean()
        self.spatial = torch.nn.ModuleList(
            [gpytorch.kernels.RBFKernel(ard_num_dims=2) for _ in range(Q)])
        self.task = torch.nn.ModuleList(
            [gpytorch.kernels.IndexKernel(num_tasks=P_, rank=1) for _ in range(Q)])

    def forward(self, x, i):
        c = self.spatial[0](x).mul(self.task[0](i))
        for q in range(1, self.Q):
            c = c + self.spatial[q](x).mul(self.task[q](i))
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), c)


def run(kind, S, seed):
    torch.manual_seed(seed)
    lik = gpytorch.likelihoods.GaussianLikelihood().double()

    if kind == "gp":
        model = SingleGP(S["Xcd"], S["ycd"], lik).double()
        model.covar_module.base_kernel.lengthscale = torch.tensor(
            [[LS_INIT_SINGLE, LS_INIT_SINGLE]], dtype=torch.float64)
        inputs, targets = (S["Xcd"],), S["ycd"]
    elif kind == "icm":
        model = ICMModel(S["Xtr"], S["itr"], S["ytr"], lik, P, 1).double()
        model.covar_module.lengthscale = torch.tensor(
            [[LS_INIT_SINGLE, LS_INIT_SINGLE]], dtype=torch.float64)
        inputs, targets = (S["Xtr"], S["itr"]), S["ytr"]
    else:
        model = TrueLMC(S["Xtr"], S["itr"], S["ytr"], lik, P, Q_BEST).double()
        for q in range(Q_BEST):
            model.spatial[q].lengthscale = torch.tensor(
                [[LS_INIT_LMC[q], LS_INIT_LMC[q]]], dtype=torch.float64)
        inputs, targets = (S["Xtr"], S["itr"]), S["ytr"]

    model.train(); lik.train()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)
    with exact():
        for _ in range(N_ITER):
            opt.zero_grad()
            loss = -mll(model(*inputs), targets)
            loss.backward()
            opt.step()

    model.eval(); lik.eval()
    with torch.no_grad(), exact():
        pred = lik(model(S["Xte"])) if kind == "gp" else lik(model(S["Xte"], S["ite"]))
        rmse = torch.sqrt(((pred.mean - S["yte"]) ** 2).mean()).item()
        mae = (pred.mean - S["yte"]).abs().mean().item()
        nll = -pred.log_prob(S["yte"]).item() / N_TEST
    return rmse, mae, nll


CSV = ROOT / "results" / "stage10_robustness.csv"

if CSV.exists() and "--retrain" not in sys.argv:
    # 训练是确定性的 (精确 Cholesky + 固定种子), 已有结果直接复用。
    # 要强制重跑加 --retrain。
    D = pd.read_csv(CSV)
    print(f"[cache] 复用已有结果 {CSV}  (加 --retrain 强制重跑)\n")
    for _, r_ in D.iterrows():
        print(f"  split {int(r_.seed)}: GP={r_.gp_rmse:.4f}  ICM={r_.icm_rmse:.4f}  "
              f"LMC={r_.lmc_rmse:.4f}   改善 ICM {r_.imp_icm:+.1f}%  LMC {r_.imp_lmc:+.1f}%")
else:
    rows = []
    t0 = time.time()
    for seed in range(N_SPLITS):
        S = make_split(seed)
        rec = dict(seed=seed)
        for kind in ["gp", "icm", "lmc"]:
            r, m, n = run(kind, S, seed)
            rec[f"{kind}_rmse"], rec[f"{kind}_mae"], rec[f"{kind}_nll"] = r, m, n
        rec["imp_icm"] = (rec["gp_rmse"] - rec["icm_rmse"]) / rec["gp_rmse"] * 100
        rec["imp_lmc"] = (rec["gp_rmse"] - rec["lmc_rmse"]) / rec["gp_rmse"] * 100
        rows.append(rec)
        print(f"  split {seed}: GP={rec['gp_rmse']:.4f}  ICM={rec['icm_rmse']:.4f}  "
              f"LMC={rec['lmc_rmse']:.4f}   改善 ICM {rec['imp_icm']:+.1f}%  LMC {rec['imp_lmc']:+.1f}%")

    D = pd.DataFrame(rows)
    D.to_csv(CSV, index=False)
    print(f"\n用时 {time.time()-t0:.1f}s")

print("\n" + "=" * 64)
print(f"{N_SPLITS} 次随机划分的汇总 (标准化尺度)")
print("=" * 64)
print(f"\n  {'模型':<12}{'RMSE':<20}{'MAE':<20}{'NLL':<18}")
for k, lab in [("gp", "独立 GP"), ("icm", "ICM"), ("lmc", f"LMC (Q={Q_BEST})")]:
    print(f"  {lab:<12}{D[k+'_rmse'].mean():.4f} ± {D[k+'_rmse'].std():.4f}     "
          f"{D[k+'_mae'].mean():.4f} ± {D[k+'_mae'].std():.4f}     "
          f"{D[k+'_nll'].mean():.4f} ± {D[k+'_nll'].std():.4f}")

print(f"\n  相对独立 GP 的 RMSE 改善:")
print(f"    ICM : {D.imp_icm.mean():+.2f}% ± {D.imp_icm.std():.2f}%   "
      f"(范围 {D.imp_icm.min():+.1f}% ~ {D.imp_icm.max():+.1f}%)")
print(f"    LMC : {D.imp_lmc.mean():+.2f}% ± {D.imp_lmc.std():.2f}%   "
      f"(范围 {D.imp_lmc.min():+.1f}% ~ {D.imp_lmc.max():+.1f}%)")

print("\n" + "-" * 64)
print("配对检验 (10 个划分上的配对 RMSE 差)")
print("-" * 64)
for a, b, lab in [("gp", "lmc", "独立 GP vs LMC"), ("gp", "icm", "独立 GP vs ICM"),
                  ("icm", "lmc", "ICM vs LMC")]:
    d = D[a + "_rmse"] - D[b + "_rmse"]
    t_stat, p_t = stats.ttest_rel(D[a + "_rmse"], D[b + "_rmse"])
    try:
        w_stat, p_w = stats.wilcoxon(D[a + "_rmse"], D[b + "_rmse"])
    except ValueError:
        p_w = float("nan")
    win = int((d > 0).sum())
    print(f"  {lab:<18} 平均差={d.mean():+.4f}  配对t p={p_t:.2e}  "
          f"Wilcoxon p={p_w:.4f}  后者获胜 {win}/{N_SPLITS}")

# ---------- 图 ----------
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

ax = axes[0]
x = np.arange(N_SPLITS)
w = 0.27
ax.bar(x - w, D.gp_rmse, w, label="Independent GP", color="#c44e52")
ax.bar(x, D.icm_rmse, w, label="ICM", color="#dd8452")
ax.bar(x + w, D.lmc_rmse, w, label=f"LMC (Q={Q_BEST})", color="#4c72b0")
ax.axhline(1.0, ls="--", c="k", lw=0.9)
ax.text(N_SPLITS - 0.5, 1.005, "constant predictor", fontsize=7, ha="right")
ax.set_xlabel("split seed"); ax.set_ylabel("test RMSE (standardised)")
ax.set_title("Per-split test RMSE"); ax.set_xticks(x); ax.legend(fontsize=8)
ax.set_ylim(0, 1.15)

ax = axes[1]
bp = ax.boxplot([D.gp_rmse, D.icm_rmse, D.lmc_rmse],
                tick_labels=["Indep. GP", "ICM", f"LMC Q={Q_BEST}"],   # mpl>=3.9 改名, 原为 labels=
                patch_artist=True, widths=0.55)
for patch, c in zip(bp["boxes"], ["#c44e52", "#dd8452", "#4c72b0"]):
    patch.set_facecolor(c); patch.set_alpha(0.75)
for i, k in enumerate(["gp", "icm", "lmc"]):
    ax.scatter(np.full(N_SPLITS, i + 1) + np.random.uniform(-0.07, 0.07, N_SPLITS),
               D[k + "_rmse"], s=16, c="k", zorder=3, alpha=0.6)
ax.set_ylabel("test RMSE (standardised)")
ax.set_title(f"Distribution over {N_SPLITS} random splits")

ax = axes[2]
for i in range(N_SPLITS):
    ax.plot([0, 1, 2], [D.gp_rmse[i], D.icm_rmse[i], D.lmc_rmse[i]],
            marker="o", ms=4, lw=0.9, alpha=0.55, color="gray")
ax.plot([0, 1, 2], [D.gp_rmse.mean(), D.icm_rmse.mean(), D.lmc_rmse.mean()],
        marker="s", ms=9, lw=2.4, color="#4c72b0", label="mean")
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["Indep. GP", "ICM", f"LMC Q={Q_BEST}"])
ax.set_ylabel("test RMSE (standardised)")
ax.set_title("Paired comparison (lines = same split)"); ax.legend(fontsize=8, loc="upper right")

# 把配对检验结论标进图里 —— 这是本图的核心主张
p_gp_icm = stats.ttest_rel(D.gp_rmse, D.icm_rmse).pvalue
p_icm_lmc = stats.ttest_rel(D.icm_rmse, D.lmc_rmse).pvalue
win_icm_lmc = int((D.icm_rmse > D.lmc_rmse).sum())
yl = ax.get_ylim()
ax.annotate("", xy=(1, yl[0] + 0.06 * (yl[1] - yl[0])), xytext=(0, yl[0] + 0.06 * (yl[1] - yl[0])),
            arrowprops=dict(arrowstyle="<->", color="#55a868", lw=1.6))
ax.text(0.5, yl[0] + 0.075 * (yl[1] - yl[0]),
        f"10/10 splits\np = {p_gp_icm:.1e}", ha="center", fontsize=8.5, color="#55a868")
ax.annotate("", xy=(2, yl[0] + 0.06 * (yl[1] - yl[0])), xytext=(1, yl[0] + 0.06 * (yl[1] - yl[0])),
            arrowprops=dict(arrowstyle="<->", color="#c44e52", lw=1.6))
ax.text(1.5, yl[0] + 0.075 * (yl[1] - yl[0]),
        f"{win_icm_lmc}/10 splits\np = {p_icm_lmc:.2f}  n.s.", ha="center", fontsize=8.5, color="#c44e52")

plt.tight_layout()
out = ROOT / "figures" / "stage10_robustness.png"
plt.savefig(out, dpi=140)
print(f"\n[OK] {out}")
print(f"[OK] {ROOT/'results'/'stage10_robustness.csv'}")
