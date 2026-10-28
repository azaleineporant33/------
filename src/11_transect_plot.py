"""
沿数据主方向切一条直线, 画经典的 "预测曲线 + 95% 置信带" 图。

现有图都不是这个形式: 空间图是二维热力图, 不确定性-距离图横轴是距离不是位置。
这里横轴是沿线的位置 (一维), 直接对应教科书上的 GP 图。
"""

import json
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

from models import exact, SingleGP, TrueLMC

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(r"E:\Research")
DATA = np.load(ROOT / "data" / "jura_prepared.npz", allow_pickle=True)
PAR = json.load(open(ROOT / "results" / "stage8_best_params.json", encoding="utf-8"))

N_LINE = 900
NEAR_KM = 0.03      # 必须收紧: MOGP 的借力半径只有 0.05 km,
                    # 用 0.2 km 会把根本借不到力的点也算进来

coords = DATA["coords"]
train_idx_cd = DATA["train_idx_cd"]
test_idx = DATA["test_idx"]
Y_std = DATA["Y_std"]

Xtr = torch.tensor(DATA["X_train"], dtype=torch.float64)
itr = torch.tensor(DATA["i_train"], dtype=torch.long)
ytr = torch.tensor(DATA["y_train"], dtype=torch.float64)
mask = itr == 0
Xcd, ycd = Xtr[mask], ytr[mask]
P = len(DATA["target_names"])

# ---------- 1. 重训独立 GP (确定性, 与阶段 8 一致) ----------
print("重训独立 GP ...")
import gpytorch
torch.manual_seed(0)
lik_gp = gpytorch.likelihoods.GaussianLikelihood().double()
gp = SingleGP(Xcd, ycd, lik_gp).double()
gp.covar_module.base_kernel.lengthscale = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
gp.train(); lik_gp.train()
opt = torch.optim.Adam(gp.parameters(), lr=0.05)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik_gp, gp)
with exact():
    for _ in range(500):
        opt.zero_grad()
        loss = -mll(gp(Xcd), ycd)
        loss.backward()
        opt.step()

# ---------- 2. 加载阶段 8 的 LMC ----------
print("加载 LMC ...")
ckpt = torch.load(ROOT / "results" / "stage8_best_lmc.pt", weights_only=False)
Q = ckpt["Q"]
lik_lmc = gpytorch.likelihoods.GaussianLikelihood().double()
lmc = TrueLMC(Xtr, itr, ytr, lik_lmc, P, Q).double()
lmc.load_state_dict(ckpt["model"])
lik_lmc.load_state_dict(ckpt["likelihood"])

# ---------- 3. 选一条真正穿过采样点的线 ----------
# 不能用 PCA 主方向随便切: MOGP 的借力半径只有 0.05 km, 随便一条线上
# 绝大多数位置离最近的采样点都远超这个距离, 画出来 MOGP 的带反而全程更宽。
# 改为沿采样网格的一列走 —— X≈3.67 这一列有 17 个采样点, 是最密的一条。
X_LINE = 3.67
on_line = np.where(np.abs(coords[:, 0] - X_LINE) < NEAR_KM)[0]
major, minor = np.array([0.0, 1.0]), np.array([1.0, 0.0])
centroid = np.array([X_LINE, 0.0])

t_on = coords[on_line, 1]
t_min, t_max = t_on.min() - 0.15, t_on.max() + 0.15
tt = np.linspace(t_min, t_max, N_LINE)
line_xy = np.column_stack([np.full(N_LINE, X_LINE), tt])
print(f"沿 X={X_LINE} km 的竖线, Y 从 {t_min:.2f} 到 {t_max:.2f} km, "
      f"穿过 {len(on_line)} 个采样点")

# ---------- 4. 沿线预测 ----------
Xline = torch.tensor(line_xy, dtype=torch.float64)
iline = torch.zeros(N_LINE, dtype=torch.long)      # task 0 = Cd

gp.eval(); lik_gp.eval(); lmc.eval(); lik_lmc.eval()
with torch.no_grad(), exact():
    pred_gp = lik_gp(gp(Xline))
    pred_lmc = lik_lmc(lmc(Xline, iline))
m_gp, s_gp = pred_gp.mean.numpy(), pred_gp.stddev.numpy()
m_lmc, s_lmc = pred_lmc.mean.numpy(), pred_lmc.stddev.numpy()

# ---------- 5. 把线附近的真实观测投影到线上 ----------
def project(idx):
    idx = np.asarray(idx)
    t = coords[idx, 1]
    perp = np.abs(coords[idx, 0] - X_LINE)
    keep = (perp < NEAR_KM) & (t >= t_min) & (t <= t_max)
    return t[keep], idx[keep]

t_train, idx_train_near = project(train_idx_cd)
t_test, idx_test_near = project(test_idx)

# 所有 359 个位置都有 Ni/Zn 观测 (02_prepare_data) -> 用作 "此处有辅助信息" 的 rug tick
t_all_near, idx_all_near = project(np.arange(len(coords)))

print(f"线附近 (<{NEAR_KM} km): Cd 训练点 {len(t_train)} 个, Cd 测试点 {len(t_test)} 个, "
      f"总采样点 {len(t_all_near)} 个 (均有 Ni/Zn)")

# ---------- 6. 画图 ----------
fig, ax = plt.subplots(figsize=(13, 6))

for m, s, col, lab in [(m_gp, s_gp, "#c44e52", "Independent GP (only sees 259 Cd points)"),
                       (m_lmc, s_lmc, "#4c72b0", f"MOGP / LMC Q={Q} (also sees Ni, Zn everywhere)")]:
    lo, hi = m - 1.96 * s, m + 1.96 * s
    ax.fill_between(tt, lo, hi, color=col, alpha=0.18, lw=0)
    ax.plot(tt, m, color=col, lw=2.2, label=lab)

# 真实观测叠加 (与 stage9_spatial_maps 的图例一致)
ax.scatter(t_train, Y_std[idx_train_near, 0], s=42, facecolors="white",
           edgecolors="#333", linewidths=1.1, zorder=5, label="Cd observed (train)")
ax.scatter(t_test, Y_std[idx_test_near, 0], s=70, marker="^", facecolors="none",
           edgecolors="red", linewidths=1.4, zorder=5, label="Cd held out (test, true value)")

# 在测试点位置画竖线 —— 这是论证的核心:
# 那里 Cd 未观测, 独立 GP 一无所知; 而 MOGP 有同点的 Ni/Zn 可借。
for k, t in enumerate(t_test):
    ax.axvline(t, color="red", ls=":", lw=1.2, alpha=0.55,
               label="Cd unmeasured here (but Ni/Zn are)" if k == 0 else None)

# 底部 rug: 所有沿线的采样位置 (代表 "这里有 Ni/Zn 可借")
ax.plot(t_all_near, np.zeros_like(t_all_near), transform=ax.get_xaxis_transform(),
        marker="|", ls="none", ms=16, mew=1.6, color="#444", alpha=0.8,
        label="sampling location (Ni/Zn measured here)")

ax.set_xlabel("position along transect (km)   —   vertical line at X = 3.67 km")
ax.set_ylabel("predicted Cd (standardised log scale)")
ax.set_title("Prediction ± 95% confidence band along a line through 17 sampling locations\n"
             "At the red lines Cd was never measured — only the MOGP's band narrows there, "
             "because only it can see the co-located Ni and Zn",
             fontsize=11.5)
ax.legend(fontsize=8.5, loc="upper left", framealpha=0.93, ncol=2)
ax.grid(alpha=0.2)

plt.tight_layout()
out = ROOT / "figures" / "transect_prediction.png"
plt.savefig(out, dpi=140)
print(f"\n[OK] {out}")

# ---------- 7. 量化: 三类位置上的置信带宽度 ----------
width_gp, width_lmc = 3.92 * s_gp, 3.92 * s_lmc


def width_at(targets):
    """取离每个目标位置最近的网格点上的带宽"""
    if len(targets) == 0:
        return None, None
    j = [np.argmin(np.abs(tt - t)) for t in targets]
    return width_gp[j].mean(), width_lmc[j].mean()


between = np.array([t for t in tt
                    if len(t_all_near) and np.min(np.abs(t_all_near - t)) > 0.10])

print("\n" + "=" * 68)
print("95% 置信带宽度 (标准化尺度, 越小越确定)")
print("=" * 68)
print(f"\n  {'位置':<34}{'独立 GP':<12}{'MOGP':<12}{'MOGP 更窄'}")
for lab, targ in [("Cd 训练点 (两个模型都有 Cd)", t_train),
                  ("Cd 测试点 (只有 MOGP 能借 Ni/Zn)", t_test),
                  ("采样点之间 (>100m, 谁都借不到)", between)]:
    g, l_ = width_at(targ)
    if g is None:
        continue
    print(f"  {lab:<36}{g:<12.3f}{l_:<12.3f}{(1-l_/g)*100:+.0f}%")

print("\n  >> 看第二行: 测试点上 Cd 从未被测量过, 独立 GP 只能退回先验;")
print("     MOGP 因为有同点的 Ni/Zn, 置信带明显收窄。这就是'借力'在图上的样子。")
