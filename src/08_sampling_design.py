"""
采样设计分析 (补充阶段 3 的 EDA)

起因: 03/04 笔记本里判读 lengthscale 时引用的"平均采样间距 0.27 km"是
      sqrt(包围盒面积 / N) 的密度估计, 不是实测值。Jura 用的是
      规则网格 + 嵌套加密簇的设计, 密度估计会严重高估典型间距。

本脚本给出实测值, 并说明为什么模型能同时辨识 0.05 km 和 0.5 km 两个尺度。

输出: figures/sampling_design.png, results/sampling_design.json
"""

import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(r"E:\Research")
df = pd.read_csv(ROOT / "data" / "jura.csv")
low = {c.lower(): c for c in df.columns}
X_COL = low.get("xloc", low.get("x"))
Y_COL = low.get("yloc", low.get("y"))
coords = df[[X_COL, Y_COL]].to_numpy(float)
N = len(coords)

nn = cKDTree(coords).query(coords, k=2)[0][:, 1]
pairs = pdist(coords)
area = np.ptp(coords[:, 0]) * np.ptp(coords[:, 1])
density_est = np.sqrt(area / N)

# 阶段 8 学到的两个 lengthscale
PAR = json.load(open(ROOT / "results" / "stage8_best_params.json", encoding="utf-8"))
LS = [(c["ls_x"] + c["ls_y"]) / 2 for c in PAR["components"]]

print("=" * 66)
print("Jura 采样设计")
print("=" * 66)
print(f"\n  N = {N} 个位置, 包围盒 {np.ptp(coords[:,0]):.2f} x {np.ptp(coords[:,1]):.2f} km"
      f" = {area:.2f} km^2")
print(f"\n  密度估计 sqrt(A/N)     = {density_est:.4f} km   <- 03/04 笔记本引用的 0.27 来源")
print(f"  实测最近邻距离 mean    = {nn.mean():.4f} km")
print(f"                 median  = {np.median(nn):.4f} km")
print(f"                 min/max = {nn.min():.4f} / {nn.max():.4f} km")

print("\n  最近邻距离分位数 —— 注意这是双峰的:")
for p in [5, 10, 25, 50, 75, 90, 100]:
    print(f"    p{p:<4} = {np.percentile(nn, p):.4f} km")
print(f"\n  >> p75 起就贴在 {np.percentile(nn,75):.3f} km 不动 = 规则网格间距")
print(f"  >> p25 只有 {np.percentile(nn,25):.4f} km          = 嵌套加密簇")
for thr in [0.01, 0.02, 0.05]:
    print(f"     有邻居在 {thr*1000:>2.0f} m 内的点: {(nn < thr).sum():>3} / {N} "
          f"({(nn < thr).mean()*100:.0f}%)")

print("\n" + "=" * 66)
print("为什么两个 lengthscale 都可辨识")
print("=" * 66)
print(f"\n  模型学到 q=1 ell≈{LS[0]:.3f} km, q=2 ell≈{LS[1]:.3f} km")
print(f"  一个尺度能否被估计, 取决于该 lag 附近有多少点对:\n")
print(f"    {'区间 (km)':<20}{'点对数':<12}{'占全部 %':<10}")
for a, b, lab in [(0, LS[0], f"[0, {LS[0]:.3f})  q=1"),
                  (LS[0], 0.125, f"[{LS[0]:.3f}, 0.125)"),
                  (0.125, 0.25, "[0.125, 0.25)"),
                  (0.25, LS[1], f"[0.25, {LS[1]:.3f})"),
                  (LS[1] * 0.7, LS[1] * 1.3, f"q=2 附近 ±30%")]:
    n = ((pairs >= a) & (pairs < b)).sum()
    print(f"    {lab:<20}{n:<12}{n/len(pairs)*100:<10.2f}")

n_short = (pairs < LS[0]).sum()
print(f"\n  >> q=1 (ell={LS[0]:.3f}) 只由 {n_short} 个点对 ({n_short/len(pairs)*100:.2f}%) 确定,")
print(f"     而这些点对几乎全部来自加密簇。若 Jura 只有规则网格, 这个尺度不可辨识。")
print(f"  >> 报告中应作为局限说明: 短程分量的估计依赖少量簇内点对。")

# ---------- 图 ----------
fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))

ax = axes[0]
ax.scatter(coords[:, 0], coords[:, 1], s=9, c="#4c72b0", alpha=0.75)
close = nn < 0.05
ax.scatter(coords[close, 0], coords[close, 1], s=26, facecolors="none",
           edgecolors="#c44e52", lw=0.9, label=f"has neighbour < 50 m ({close.sum()})")
ax.set_xlabel("X (km)"); ax.set_ylabel("Y (km)"); ax.set_aspect("equal")
ax.set_title("Sampling layout: regular grid + nested clusters")
ax.legend(fontsize=8)

ax = axes[1]
ax.hist(nn, bins=60, color="#4c72b0", edgecolor="white", lw=0.4)
ax.axvline(nn.mean(), c="#c44e52", lw=1.8, label=f"mean {nn.mean():.3f} km")
ax.axvline(density_est, c="k", ls="--", lw=1.5,
           label=f"$\\sqrt{{A/N}}$ = {density_est:.3f} km (over-estimate)")
for i, l in enumerate(LS):
    ax.axvline(l, ls=":", lw=1.6, c=["#55a868", "#8172b3"][i], label=f"q={i+1} $\\ell$ = {l:.3f} km")
ax.set_xlabel("nearest-neighbour distance (km)"); ax.set_ylabel("count")
ax.set_title("NN distance is bimodal:\nclusters at ~10 m, grid at ~250 m")
ax.legend(fontsize=7.5)

ax = axes[2]
bins = np.logspace(np.log10(0.004), np.log10(pairs.max()), 60)
ax.hist(pairs, bins=bins, color="#dd8452", edgecolor="white", lw=0.3)
ax.set_xscale("log"); ax.set_yscale("log")
for i, l in enumerate(LS):
    ax.axvline(l, ls=":", lw=1.8, c=["#55a868", "#8172b3"][i], label=f"q={i+1} $\\ell$ = {l:.3f} km")
ax.set_xlabel("pairwise distance (km, log scale)"); ax.set_ylabel("number of pairs (log)")
ax.set_title(f"Pairs available at each lag\n({n_short} pairs below q=1 $\\ell$, "
             f"{n_short/len(pairs)*100:.2f}% of all)")
ax.legend(fontsize=8)

plt.tight_layout()
out = ROOT / "figures" / "sampling_design.png"
plt.savefig(out, dpi=140)

res = dict(
    n_locations=int(N), area_km2=float(area),
    density_estimate_sqrt_A_over_N=float(density_est),
    nn_mean=float(nn.mean()), nn_median=float(np.median(nn)),
    nn_min=float(nn.min()), nn_max=float(nn.max()),
    nn_p25=float(np.percentile(nn, 25)), nn_p75=float(np.percentile(nn, 75)),
    n_with_neighbour_under_50m=int((nn < 0.05).sum()),
    grid_spacing_km=float(np.percentile(nn, 90)),
    n_pairs_total=int(len(pairs)),
    n_pairs_below_q1_lengthscale=int(n_short),
    learned_lengthscales=[float(l) for l in LS],
    note="0.27 km quoted in notebooks 03/04 was sqrt(A/N); actual mean NN is 0.125 km",
)
with open(ROOT / "results" / "sampling_design.json", "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2, ensure_ascii=False)

print(f"\n[OK] {out}")
print(f"[OK] {ROOT/'results'/'sampling_design.json'}")
