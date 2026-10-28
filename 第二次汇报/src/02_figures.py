"""
第二次汇报的图 (读 01_compare.py 的结果, 不重新训练)。规划见 ../PPT规划.md

输出 (../figures/):
  fig0_split_map.png       第 3 页: 训练点 / 测试点在哪里 (划分 0)
  fig1_paired_lines.png    第 6 页: 10 次划分的配对折线
  fig2_pred_vs_true.png    备用页: 预测值 vs 真实值 (划分 0)
  fig3_neighbour_errors.png 第 8 页: 相邻点上回归 B 的错误方向是否一致 (划分 0)
  fig3b_residual_map_B.png  备用页: 回归 B 的残差地图 (划分 0)

图上文字用英文, 与第一次汇报的图一致。
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

HERE = Path(__file__).resolve().parent.parent
RES = HERE / "results"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

# 配色 (dataviz 参考色板, 浅色模式)
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#b5b4ae"
BLUE, ORANGE, RED, MID = "#2a78d6", "#eb6834", "#e34948", "#f0efec"

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.8,
    "savefig.dpi": 160, "savefig.bbox": "tight",
})

MODELS = ["reg_A", "reg_B", "reg_C", "mogp"]
NAMES = {"reg_A": "Regression A\n(Ni, Zn)",
         "reg_B": "Regression B\n(+ location trend)",
         "reg_C": "Regression C\n(+ rock, land use)",
         "mogp": "MOGP\n(+ neighbours)"}
SHORT = {"reg_A": "Regression A", "reg_B": "Regression B",
         "reg_C": "Regression C", "mogp": "MOGP"}

D = pd.read_csv(RES / "per_split.csv")
P0 = pd.read_csv(RES / "seed0_test_predictions.csv")
R0 = pd.read_csv(RES / "seed0_train_residuals_B.csv")


def square_map(ax):
    ax.set_aspect("equal")
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")


# ---------- 图 0: 划分示意 ----------
fig, ax = plt.subplots(figsize=(6.2, 6.2))
ax.scatter(R0.x, R0.y, s=22, c=MUTED, edgecolors="white", linewidths=0.6,
           label=f"Cd visible to models ({len(R0)} points)")
ax.scatter(P0.x, P0.y, s=30, c=ORANGE, edgecolors="white", linewidths=0.6,
           label=f"Cd hidden, used to score ({len(P0)} points)")
square_map(ax)
ax.set_title("One of the 10 splits (split 0)\nNi and Zn are known at all 359 points", fontsize=12)
ax.legend(loc="upper left", bbox_to_anchor=(0, -0.12), frameon=False, fontsize=10)
fig.savefig(FIG / "fig0_split_map.png")
plt.close(fig)

# ---------- 图 1: 配对折线 ----------
fig, ax = plt.subplots(figsize=(8.2, 5.2))
x = np.arange(len(MODELS))
for _, r in D.iterrows():
    ax.plot(x, [r[f"{m}_rmse"] for m in MODELS], color=MUTED, lw=1.4, marker="o", ms=4.5,
            zorder=2)
means = [D[f"{m}_rmse"].mean() for m in MODELS]
ax.plot(x, means, color=BLUE, lw=2.5, marker="o", ms=9, zorder=3,
        markeredgecolor="white", markeredgewidth=1.5, label="Average of 10 splits")
for xi, mv in zip(x, means):
    ax.annotate(f"{mv:.3f}", (xi, mv), xytext=(10, 10), textcoords="offset points",
                va="bottom", fontsize=10.5, color=INK, fontweight="bold")
ax.plot([], [], color=MUTED, lw=1.4, marker="o", ms=4.5, label="One split (same 100 hidden points)")
ax.set_xticks(x)
ax.set_xticklabels([NAMES[m] for m in MODELS], fontsize=10)
ax.set_xlim(-0.35, len(MODELS) - 0.45)
ax.set_ylabel("Test RMSE  (lower = more accurate)")
ax.set_title("Every model takes the same 10 exams", fontsize=12.5)
ax.legend(frameon=False, fontsize=9.5, loc="upper right")
ax.grid(axis="x", visible=False)
fig.savefig(FIG / "fig1_paired_lines.png")
plt.close(fig)

# ---------- 图 2: 预测值 vs 真实值 ----------
fig, axes = plt.subplots(1, 4, figsize=(15, 4.3), sharex=True, sharey=True)
lim = [min(P0.true.min(), *(P0[f"{m}_pred"].min() for m in MODELS)) - 0.2,
       max(P0.true.max(), *(P0[f"{m}_pred"].max() for m in MODELS)) + 0.2]
for ax, m in zip(axes, MODELS):
    rmse = np.sqrt(np.mean((P0[f"{m}_pred"] - P0.true) ** 2))
    ax.plot(lim, lim, ls="--", color=INK2, lw=1, zorder=1)
    ax.scatter(P0.true, P0[f"{m}_pred"], s=22, c=BLUE, edgecolors="white", linewidths=0.6,
               zorder=2)
    ax.set_title(f"{SHORT[m]}   RMSE = {rmse:.3f}", fontsize=11)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel("True Cd")
axes[0].set_ylabel("Predicted Cd")
fig.suptitle("Split 0, 100 hidden points  (closer to the dashed line = better)", fontsize=12, y=0.93)
fig.savefig(FIG / "fig2_pred_vs_true.png")
plt.close(fig)

# ---------- 图 3: 回归 B 残差地图 ----------
cmap = LinearSegmentedColormap.from_list("blue_gray_red", [BLUE, MID, RED])
vmax = np.quantile(np.abs(R0.resid), 0.95)
fig, ax = plt.subplots(figsize=(7.2, 6.4))
sc = ax.scatter(R0.x, R0.y, c=R0.resid, cmap=cmap, norm=TwoSlopeNorm(0, -vmax, vmax),
                s=34, edgecolors="white", linewidths=0.6)
square_map(ax)
cb = fig.colorbar(sc, ax=ax, shrink=0.82, pad=0.03)
cb.set_label("Regression B error at each point\nred = predicted too low, blue = too high",
             color=INK2)
cb.outline.set_visible(False)
ax.set_title("Where does Regression B go wrong?  (split 0, 259 points)", fontsize=12)
fig.savefig(FIG / "fig3b_residual_map_B.png")
plt.close(fig)

# ---------- 图 3: 邻居的错误方向是否一致 ----------
# 地图上看不出来 (簇只有几十米, 整张图 5 km), 所以直接数:
# 两个点的回归错误同为"猜低"或同为"猜高", 算"方向一致"。
# 如果回归没漏掉邻居信息, 方向一致的比例应该像抛硬币, 约 50%。
from scipy.spatial.distance import pdist

NEAR_KM = 0.05
dist = pdist(R0[["x", "y"]].to_numpy())
i, j = np.triu_indices(len(R0), 1)
same = np.sign(R0.resid.to_numpy()[i]) == np.sign(R0.resid.to_numpy()[j])
groups = [("Neighbours\n(within 50 m)", same[dist <= NEAR_KM], (dist <= NEAR_KM).sum()),
          ("Points farther apart\n(more than 50 m)", same[dist > NEAR_KM], (dist > NEAR_KM).sum())]

fig, ax = plt.subplots(figsize=(6.4, 5.0))
vals = [g[1].mean() * 100 for g in groups]
bars = ax.bar([0, 1], vals, width=0.55, color=[RED, MUTED], zorder=2)
for b, v, g in zip(bars, vals, groups):
    ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%", ha="center",
            fontsize=13, fontweight="bold", color=INK)
    ax.text(b.get_x() + b.get_width() / 2, 3, f"{g[2]:,} pairs", ha="center",
            fontsize=9.5, color=INK2)
ax.axhline(50, ls="--", color=INK2, lw=1.2, zorder=3)
ax.text(0.5, 51.5, "coin flip (50%)", ha="center", fontsize=9.5, color=INK2)
ax.set_xticks([0, 1])
ax.set_xticklabels([g[0] for g in groups], fontsize=10.5)
ax.set_ylim(0, 100)
ax.set_ylabel("Pairs where Regression B errs\nin the same direction (%)")
ax.set_title("Regression B makes the same mistake at neighbouring points\n(split 0, 259 training points)",
             fontsize=11.5)
ax.grid(axis="x", visible=False)
fig.savefig(FIG / "fig3_neighbour_errors.png")
plt.close(fig)

print("[OK]", *sorted(p.name for p in FIG.glob("fig*.png")))
