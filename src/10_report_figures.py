"""
报告用图 (替换两张作废的旧图)

替换:
  figures/lmc_vs_independent.png  —— CG 旧数字 + 中文豆腐块 + 模型名标错 (是 ICM 不是 LMC),
                                     且源码已丢失
  figures/true_lmc.png            —— 空白图 (plt.show() 写在 plt.savefig() 之前),
                                     且当时按训练 loss 选 Q, 选出了 Q=4

产出:
  figures/pred_vs_true.png     三模型预测散点 + B 相关性对比 (报告结果图 / 汇报第 7 页)
  figures/model_selection.png  BIC/AIC 选 Q + 拟合项与惩罚项的权衡 (汇报第 5 页)

LMC 直接加载阶段 8 存的权重; GP 和 ICM 重训 (确定性, 约 40 秒, 与阶段 8 逐位一致)。
所有图注用英文, 避免中文字体缺失。
"""

import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import gpytorch
from pathlib import Path

from models import exact, SingleGP, ICMModel, TrueLMC

ROOT = Path(r"E:\Research")
DATA = np.load(ROOT / "data" / "jura_prepared.npz", allow_pickle=True)

N_ITER = 500
LR = 0.05
LS_INIT = 0.5

C_GP, C_ICM, C_LMC, C_EMP = "#c44e52", "#dd8452", "#4c72b0", "#55a868"

Xtr = torch.tensor(DATA["X_train"], dtype=torch.float64)
itr = torch.tensor(DATA["i_train"], dtype=torch.long)
ytr = torch.tensor(DATA["y_train"], dtype=torch.float64)
Xte = torch.tensor(DATA["X_test"], dtype=torch.float64)
ite = torch.tensor(DATA["i_test"], dtype=torch.long)
yte = torch.tensor(DATA["y_test"], dtype=torch.float64)
mask = itr == 0
Xcd, ycd = Xtr[mask], ytr[mask]
NAMES = [str(n) for n in DATA["target_names"]]
P = len(NAMES)

PAR = json.load(open(ROOT / "results" / "stage8_best_params.json", encoding="utf-8"))
ICMP = json.load(open(ROOT / "results" / "stage8_icm_params.json", encoding="utf-8"))
ALL = pd.read_csv(ROOT / "results" / "stage8_all_models.csv")


def train(model, lik, inputs, targets):
    model.train(); lik.train()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)
    with exact():
        for _ in range(N_ITER):
            opt.zero_grad()
            loss = -mll(model(*inputs), targets)
            loss.backward()
            opt.step()
    return model, lik


print("重训独立 GP ...")
torch.manual_seed(0)
lik_gp = gpytorch.likelihoods.GaussianLikelihood().double()
gp = SingleGP(Xcd, ycd, lik_gp).double()
gp.covar_module.base_kernel.lengthscale = torch.tensor([[LS_INIT, LS_INIT]], dtype=torch.float64)
gp, lik_gp = train(gp, lik_gp, (Xcd,), ycd)

print("重训 ICM ...")
torch.manual_seed(0)
lik_icm = gpytorch.likelihoods.GaussianLikelihood().double()
icm = ICMModel(Xtr, itr, ytr, lik_icm, P, rank=1).double()
icm.covar_module.lengthscale = torch.tensor([[LS_INIT, LS_INIT]], dtype=torch.float64)
icm, lik_icm = train(icm, lik_icm, (Xtr, itr), ytr)

print("加载阶段 8 的 LMC 权重 ...")
ckpt = torch.load(ROOT / "results" / "stage8_best_lmc.pt", weights_only=False)
Q = ckpt["Q"]
lik_lmc = gpytorch.likelihoods.GaussianLikelihood().double()
lmc = TrueLMC(Xtr, itr, ytr, lik_lmc, P, Q).double()
lmc.load_state_dict(ckpt["model"])
lik_lmc.load_state_dict(ckpt["likelihood"])

preds = {}
for name, m, l, multi in [("gp", gp, lik_gp, False), ("icm", icm, lik_icm, True),
                          ("lmc", lmc, lik_lmc, True)]:
    m.eval(); l.eval()
    with torch.no_grad(), exact():
        d = l(m(Xte, ite)) if multi else l(m(Xte))
        preds[name] = (d.mean.numpy(), d.stddev.numpy())

y = yte.numpy()
stats = {}
for k, (mn, _) in preds.items():
    stats[k] = dict(rmse=float(np.sqrt(((mn - y) ** 2).mean())),
                    r2=float(1 - ((mn - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()))

# 与阶段 8 核对, 防止漂移
for k, ref in [("gp", PAR["baseline_gp"]["rmse"]), ("icm", ICMP["rmse"]),
               ("lmc", PAR["rmse_std"])]:
    d = abs(stats[k]["rmse"] - ref)
    print(f"  {k:<4} RMSE {stats[k]['rmse']:.6f}  阶段8 {ref:.6f}  差 {d:.2e}"
          f"{'  <-- 不一致!' if d > 1e-5 else ''}")


# ============================================================
# 图 1: 预测 vs 真值 + B 相关性对比
# ============================================================
fig, axes = plt.subplots(1, 4, figsize=(19, 5.4))   # 高一点, 否则三行标题会被裁掉
lim = (-3.0, 3.0)

for ax, (key, title, col) in zip(
    axes[:3],
    [("gp", "Independent GP\n(259 Cd only)", C_GP),
     ("icm", "ICM\n(+ Ni/Zn, one lengthscale)", C_ICM),
     ("lmc", f"LMC Q={Q}\n(+ Ni/Zn, two lengthscales)", C_LMC)],
):
    mn, sdv = preds[key]
    ax.errorbar(y, mn, yerr=1.96 * sdv, fmt="none", ecolor=col, alpha=0.22, lw=0.8)
    ax.scatter(y, mn, s=26, color=col, edgecolor="white", lw=0.5, zorder=3)
    ax.plot(lim, lim, ls="--", c="k", lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel("True Cd (standardised log)")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{title}\nRMSE = {stats[key]['rmse']:.4f}   $R^2$ = {stats[key]['r2']:.3f}",
                 fontsize=10)
    ax.grid(alpha=0.2)

# 第四联: 经验 vs ICM vs LMC 各分量的跨输出相关性
ax = axes[3]


def corr_of(B):
    B = np.array(B)
    d = np.sqrt(np.diag(B))
    return B / np.outer(d, d)


EMP = np.array([0.654, 0.686, 0.699])                      # EDA 实测 (log 尺度 Pearson)
pairs = [(0, 1), (0, 2), (1, 2)]
labels = ["Cd–Ni", "Cd–Zn", "Ni–Zn"]
icm_c = corr_of(ICMP["B"])
q1_c = corr_of(PAR["components"][0]["B"])
q2_c = corr_of(PAR["components"][1]["B"])

series = [
    ("Empirical (EDA)", EMP, C_EMP),
    ("ICM — single B", np.array([icm_c[a, b] for a, b in pairs]), C_ICM),
    (f"LMC $B_1$ (short, ℓ≈{(PAR['components'][0]['ls_x']+PAR['components'][0]['ls_y'])/2:.3f} km)",
     np.array([q1_c[a, b] for a, b in pairs]), "#8172b3"),
    (f"LMC $B_2$ (long, ℓ≈{(PAR['components'][1]['ls_x']+PAR['components'][1]['ls_y'])/2:.3f} km)",
     np.array([q2_c[a, b] for a, b in pairs]), C_LMC),
]
x = np.arange(3)
w = 0.2
for n, (lab, vals, col) in enumerate(series):
    ax.bar(x + (n - 1.5) * w, vals, w, label=lab, color=col, edgecolor="white", lw=0.6)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("cross-output correlation")
ax.set_ylim(0, 1.05)
ax.set_title("ICM's single correlation is a blend\nof two physically distinct couplings", fontsize=10)
ax.legend(fontsize=7.4, loc="lower right")
ax.grid(axis="y", alpha=0.25)

plt.tight_layout()
out1 = ROOT / "figures" / "pred_vs_true.png"
plt.savefig(out1, dpi=140)      # 注意: savefig 必须在 show 之前, 否则存下来是空白
print(f"\n[OK] {out1}")


# ============================================================
# 图 2: 用 BIC / AIC 选 Q
# ============================================================
sel = (ALL[ALL.kind == "lmc"].sort_values("bic")
       .groupby("param", as_index=False).first().sort_values("param"))
Qs = sel["param"].astype(int).to_numpy()
bic, aic = sel.bic.to_numpy(), sel.aic.to_numpy()
fit = -2 * sel.logml.to_numpy()
pen_bic = bic - fit
qbest = int(Qs[bic.argmin()])

fig2, axs = plt.subplots(1, 2, figsize=(13, 4.7))

ax = axs[0]
ax.plot(Qs, bic, "o-", color=C_LMC, lw=2, ms=7, label="BIC")
ax.plot(Qs, aic, "s--", color=C_ICM, lw=2, ms=6, label="AIC")
for arr, col in [(bic, C_LMC), (aic, C_ICM)]:
    i = arr.argmin()
    ax.scatter([Qs[i]], [arr[i]], s=260, marker="*", color=col,
               edgecolor="k", lw=0.7, zorder=5)
ax.set_xticks(Qs)
ax.set_xlabel("Q  (number of latent functions)")
ax.set_ylabel("information criterion  (lower is better)")
ax.set_title(f"Both criteria select Q = {qbest}", fontsize=11)
ax.legend(fontsize=9)
ax.grid(alpha=0.25)

# 右图: 每加一个潜在函数, 买到多少拟合改善, 付出多少复杂度代价。
# 直接画 misfit 的绝对值没用 —— 1841~1962 的差异在 0 起点的坐标轴上看不出来,
# 而 BIC 判的本来就是"增量收益是否抵得过增量惩罚"。
ax = axs[1]
gain = fit[:-1] - fit[1:]                       # 从 Q 到 Q+1 减少的 misfit
cost = pen_bic[1:] - pen_bic[:-1]               # 增加的惩罚 = 8*ln(N), 每步恒定
steps = [f"{a}→{b}" for a, b in zip(Qs[:-1], Qs[1:])]
xs = np.arange(len(steps))

ax.bar(xs - 0.2, gain, 0.4, label="misfit reduction gained", color=C_LMC)
ax.bar(xs + 0.2, cost, 0.4, label="BIC penalty paid  ($8\\ln N$)", color=C_GP)
for i, (g, c) in enumerate(zip(gain, cost)):
    ax.text(i - 0.2, g + 2, f"{g:.1f}", ha="center", fontsize=8.5, color=C_LMC)
    ax.text(i + 0.2, c + 2, f"{c:.1f}", ha="center", fontsize=8.5, color=C_GP)
    verdict = "worth it" if g > c else "not worth it"
    ax.text(i, max(g, c) + 14, verdict, ha="center", fontsize=8.5,
            style="italic", color="#333" if g > c else C_GP)
ax.set_xticks(xs); ax.set_xticklabels(steps)
ax.set_xlabel("adding one more latent function")
ax.set_ylabel("BIC units")
ax.set_ylim(0, max(gain.max(), cost.max()) * 1.35)
ax.set_title("Only the first extra latent function pays for itself", fontsize=11)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.25)

plt.tight_layout()
out2 = ROOT / "figures" / "model_selection.png"
plt.savefig(out2, dpi=140)
print(f"[OK] {out2}")

print("\n" + "=" * 60)
print("旧图已被取代, 可以删除:")
print("  figures/lmc_vs_independent.png   -> figures/pred_vs_true.png")
print("  figures/true_lmc.png             -> figures/model_selection.png")
print("=" * 60)
