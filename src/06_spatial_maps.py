"""
阶段 9: 空间后验热力图 (报告主视觉)

加载阶段 8 存下的最优 LMC (results/stage8_best_lmc.pt), 在研究区网格上
预测 Cd 的后验均值和不确定性, 与独立 GP 基线并排对比。

要让读者一眼看到的三件事:
  1. LMC 的后验均值有结构, 独立 GP 基本是常数背景
  2. LMC 的后验标准差在"有 Ni/Zn 观测的位置"显著更低 —— 这就是跨输出借力的空间证据
  3. 100 个测试位置正好落在那些低不确定性的点上

输出: figures/stage9_spatial_maps.png
      figures/stage9_latent_decomposition.png  (Q>=2 时)
"""

import contextlib
import json
import numpy as np
import torch
import gpytorch
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from pathlib import Path

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(r"E:\Research")
DATA = np.load(ROOT / "data" / "jura_prepared.npz", allow_pickle=True)
(ROOT / "figures").mkdir(exist_ok=True)

GRID_N = 140          # 网格分辨率 (每个方向)
N_ITER = 500
LR = 0.05


@contextlib.contextmanager
def exact():
    with gpytorch.settings.max_cholesky_size(10_000), \
         gpytorch.settings.fast_computations(
             covar_root_decomposition=False, log_prob=False, solves=False
         ):
        yield


# ---------- 数据 ----------
Xtr = torch.tensor(DATA["X_train"], dtype=torch.float64)
itr = torch.tensor(DATA["i_train"], dtype=torch.long)
ytr = torch.tensor(DATA["y_train"], dtype=torch.float64)
mask = itr == 0
Xcd, ycd = Xtr[mask], ytr[mask]

coords = DATA["coords"]
train_idx_cd = DATA["train_idx_cd"]
test_idx = DATA["test_idx"]
mu, sd = DATA["mu"], DATA["sd"]
NAMES = [str(n) for n in DATA["target_names"]]
P = len(NAMES)


# ---------- 模型定义 (与阶段 8 一致) ----------
class TrueLMC(gpytorch.models.ExactGP):
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


class SingleGP(gpytorch.models.ExactGP):
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


# ---------- 加载最优 LMC ----------
ckpt = torch.load(ROOT / "results" / "stage8_best_lmc.pt", weights_only=False)
Q = ckpt["Q"]
with open(ROOT / "results" / "stage8_best_params.json", encoding="utf-8") as f:
    PAR = json.load(f)

lik_lmc = gpytorch.likelihoods.GaussianLikelihood().double()
lmc = TrueLMC(Xtr, itr, ytr, lik_lmc, P, Q).double()
lmc.load_state_dict(ckpt["model"])
lik_lmc.load_state_dict(ckpt["likelihood"])
print(f"已加载最优 LMC: Q={Q}, RMSE={PAR['rmse_std']:.4f}")
for c in PAR["components"]:
    print(f"  q={c['q']}: lengthscale=({c['ls_x']:.3f}, {c['ls_y']:.3f}) km, "
          f"方差占比 " + ", ".join(f"{n}={f*100:.1f}%" for n, f in zip(NAMES, c["var_frac"])))

# ---------- 重训独立 GP 基线 (259 点, 很快) ----------
print("\n重训独立 GP 基线 ...")
torch.manual_seed(0)
lik_gp = gpytorch.likelihoods.GaussianLikelihood().double()
gp = SingleGP(Xcd, ycd, lik_gp).double()
gp.covar_module.base_kernel.lengthscale = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
gp.train(); lik_gp.train()
opt = torch.optim.Adam(gp.parameters(), lr=LR)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik_gp, gp)
with exact():
    for _ in range(N_ITER):
        opt.zero_grad()
        loss = -mll(gp(Xcd), ycd)
        loss.backward()
        opt.step()
print(f"  baseline loss={loss.item():.4f}  (阶段 8 记录 {PAR['baseline_gp']['loss']:.4f})")


# ---------- 网格 ----------
pad = 0.15
x_min, x_max = coords[:, 0].min() - pad, coords[:, 0].max() + pad
y_min, y_max = coords[:, 1].min() - pad, coords[:, 1].max() + pad
gx = np.linspace(x_min, x_max, GRID_N)
gy = np.linspace(y_min, y_max, GRID_N)
GX, GY = np.meshgrid(gx, gy)
grid = np.column_stack([GX.ravel(), GY.ravel()])
Xg = torch.tensor(grid, dtype=torch.float64)
ig = torch.zeros(len(grid), dtype=torch.long)      # task 0 = Cd
print(f"\n网格 {GRID_N}x{GRID_N} = {len(grid)} 个预测点")


def predict_grid(model, lik, kind, chunk=4000):
    """分块预测, 避免一次性构造 19600x977 的矩阵。返回标准化尺度的 mean/std。"""
    model.eval(); lik.eval()
    means, stds = [], []
    with torch.no_grad(), exact():
        for a in range(0, len(Xg), chunk):
            b = min(a + chunk, len(Xg))
            pred = lik(model(Xg[a:b])) if kind == "gp" else lik(model(Xg[a:b], ig[a:b]))
            means.append(pred.mean.numpy())
            stds.append(pred.stddev.numpy())
    return np.concatenate(means), np.concatenate(stds)


print("预测 LMC 网格 ...")
m_lmc, s_lmc = predict_grid(lmc, lik_lmc, "lmc")
print("预测独立 GP 网格 ...")
m_gp, s_gp = predict_grid(gp, lik_gp, "gp")


def to_ppm(m_std):
    """标准化 -> log 浓度 -> ppm (中位数还原)"""
    return np.exp(m_std * sd[0] + mu[0])


M_LMC, S_LMC = to_ppm(m_lmc).reshape(GRID_N, GRID_N), s_lmc.reshape(GRID_N, GRID_N)
M_GP, S_GP = to_ppm(m_gp).reshape(GRID_N, GRID_N), s_gp.reshape(GRID_N, GRID_N)
REDUCTION = (S_GP - S_LMC) / S_GP * 100


# ---------- 绘图 ----------
extent = [x_min, x_max, y_min, y_max]
tr_xy = coords[train_idx_cd]
te_xy = coords[test_idx]

fig, axes = plt.subplots(2, 3, figsize=(17, 10.5))

vmin_m = min(M_LMC.min(), M_GP.min())
vmax_m = max(np.percentile(M_LMC, 99.5), np.percentile(M_GP, 99.5))
norm_m = Normalize(vmin_m, vmax_m)
vmax_s = max(S_LMC.max(), S_GP.max())
norm_s = Normalize(0, vmax_s)


def base(ax, title, show_pts=True):
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("X (km)"); ax.set_ylabel("Y (km)")
    ax.set_aspect("equal")
    if show_pts:
        ax.scatter(tr_xy[:, 0], tr_xy[:, 1], s=7, facecolors="none",
                   edgecolors="white", linewidths=0.5, label="Cd observed (259)")
        ax.scatter(te_xy[:, 0], te_xy[:, 1], s=22, marker="^", facecolors="none",
                   edgecolors="red", linewidths=0.9, label="Cd held out (100)")


# --- 第一行: 后验均值 ---
ax = axes[0, 0]
im = ax.imshow(M_GP, extent=extent, origin="lower", cmap="viridis", norm=norm_m)
base(ax, "Independent GP — posterior mean Cd (ppm)")
plt.colorbar(im, ax=ax, fraction=0.046)

ax = axes[0, 1]
im = ax.imshow(M_LMC, extent=extent, origin="lower", cmap="viridis", norm=norm_m)
base(ax, f"LMC (Q={Q}) — posterior mean Cd (ppm)")
plt.colorbar(im, ax=ax, fraction=0.046)
ax.legend(loc="upper right", fontsize=7, framealpha=0.8)

ax = axes[0, 2]
im = ax.imshow(M_LMC - M_GP, extent=extent, origin="lower", cmap="RdBu_r",
               norm=Normalize(-np.abs(M_LMC - M_GP).max(), np.abs(M_LMC - M_GP).max()))
base(ax, "Difference: LMC − Independent GP (ppm)", show_pts=False)
plt.colorbar(im, ax=ax, fraction=0.046)

# --- 第二行: 不确定性 ---
ax = axes[1, 0]
im = ax.imshow(S_GP, extent=extent, origin="lower", cmap="magma", norm=norm_s)
base(ax, "Independent GP — posterior std (standardised)")
plt.colorbar(im, ax=ax, fraction=0.046)

ax = axes[1, 1]
im = ax.imshow(S_LMC, extent=extent, origin="lower", cmap="magma", norm=norm_s)
base(ax, f"LMC (Q={Q}) — posterior std (standardised)")
plt.colorbar(im, ax=ax, fraction=0.046)

ax = axes[1, 2]
rlim = np.abs(REDUCTION).max()
im = ax.imshow(REDUCTION, extent=extent, origin="lower", cmap="RdBu_r",
               norm=Normalize(-rlim, rlim))   # 有正有负, 必须用发散色图并以 0 为中心
base(ax, "Uncertainty reduction from borrowing Ni/Zn (%)\nred = LMC more certain, blue = less")
plt.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()
out = ROOT / "figures" / "stage9_spatial_maps.png"
plt.savefig(out, dpi=140)
print(f"\n[OK] {out}")


# ---------- 定量: 测试位置上的不确定性下降 ----------
Xte_t = torch.tensor(coords[test_idx], dtype=torch.float64)
ite_t = torch.zeros(len(test_idx), dtype=torch.long)
lmc.eval(); lik_lmc.eval(); gp.eval(); lik_gp.eval()
with torch.no_grad(), exact():
    s_lmc_te = lik_lmc(lmc(Xte_t, ite_t)).stddev.numpy()
    s_gp_te = lik_gp(gp(Xte_t)).stddev.numpy()

print("\n" + "=" * 62)
print("100 个测试位置上的后验标准差 (标准化尺度)")
print("=" * 62)
print(f"  独立 GP : {s_gp_te.mean():.4f} ± {s_gp_te.std():.4f}")
print(f"  LMC     : {s_lmc_te.mean():.4f} ± {s_lmc_te.std():.4f}")
print(f"  平均下降: {(s_gp_te.mean()-s_lmc_te.mean())/s_gp_te.mean()*100:.1f}%")

# ---------- 不确定性下降 vs 到最近采样点的距离 ----------
# Ni/Zn 在全部 359 个位置都有观测, 所以"到最近采样点的距离"就是
# "到最近的可借力信息的距离"。这决定了 LMC 能不能借到力。
from scipy.spatial import cKDTree

d_all = cKDTree(coords).query(grid)[0]                     # 到最近的 359 个位置
d_cd = cKDTree(coords[train_idx_cd]).query(grid)[0]        # 到最近的 259 个 Cd 观测

print("\n" + "=" * 62)
print("不确定性下降按 '到最近采样点距离' 分档")
print("=" * 62)
bins = [0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.60, 10.0]
print(f"\n  {'距离档 (km)':<16}{'网格点数':<10}{'GP std':<10}{'LMC std':<10}{'下降 %':<10}")
band_rows = []
for a, b in zip(bins[:-1], bins[1:]):
    sel = (d_all >= a) & (d_all < b)
    if sel.sum() == 0:
        continue
    g, l_ = s_gp.reshape(-1)[sel].mean(), s_lmc.reshape(-1)[sel].mean()
    red = (g - l_) / g * 100
    lab = f"[{a:.2f}, {b:.2f})" if b < 10 else f"[{a:.2f}, +inf)"
    print(f"  {lab:<16}{sel.sum():<10}{g:<10.4f}{l_:<10.4f}{red:<+10.1f}")
    band_rows.append(((a + min(b, 1.0)) / 2, red, g, l_))

print("\n>> 三段式结构 (这是核心论点, 比单一平均数有信息量得多):")
print("   1. 采样点上 (d≈0)      : Ni/Zn 同点可借 -> LMC 大幅更确定")
print("   2. 采样点之间 (d~0.1-0.3): 短程分量够不着, LMC 反而更不确定")
print("   3. 研究区之外 (d 大)    : 两者都退回先验, 差异来自先验方差本身")
print("   独立 GP 用较长的 lengthscale + 较高噪声换来'处处中等确定',")
print("   LMC 则是'该确定的地方很确定, 该不确定的地方老实承认' —— ")
print(f"   这也是 LMC 的 NLL ({PAR['nll']:.4f}) 明显优于 GP "
      f"({PAR['baseline_gp']['nll']:.4f}) 的原因。")

fig3, axs3 = plt.subplots(1, 2, figsize=(12.5, 4.6))
ax = axs3[0]
edges = np.linspace(0, 0.8, 41)
ctr = (edges[:-1] + edges[1:]) / 2
for arr, lab, col in [(s_gp.reshape(-1), "Independent GP", "#c44e52"),
                      (s_lmc.reshape(-1), f"LMC (Q={Q})", "#4c72b0")]:
    med, lo, hi = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        v = arr[(d_all >= a) & (d_all < b)]
        if len(v) == 0:
            med.append(np.nan); lo.append(np.nan); hi.append(np.nan)
        else:
            med.append(np.median(v)); lo.append(np.percentile(v, 25)); hi.append(np.percentile(v, 75))
    ax.fill_between(ctr, lo, hi, color=col, alpha=0.22)
    ax.plot(ctr, med, lw=2.0, color=col, label=lab)
ax.axvline(0.052, ls=":", c="#4c72b0", lw=1.4)
ax.text(0.058, 0.42, "q=1 lengthscale\n0.052 km", fontsize=8, color="#4c72b0")
ax.axvline(0.125, ls="-.", c="gray", lw=1)
ax.text(0.13, 0.55, "mean NN\n0.125 km", fontsize=8, color="gray")
ax.axvline(0.25, ls="--", c="gray", lw=1)
ax.text(0.26, 0.42, "grid spacing\n0.25 km", fontsize=8, color="gray")
ax.set_xlabel("distance to nearest sampling location (km)")
ax.set_ylabel("posterior std (standardised)")
ax.set_title("Posterior uncertainty vs distance\n(median, band = IQR over grid cells)")
ax.set_xlim(0, 0.8); ax.legend(fontsize=9, loc="lower right"); ax.grid(alpha=0.25)

ax = axs3[1]
bc = np.array([r[0] for r in band_rows])
br = np.array([r[1] for r in band_rows])
ax.axhline(0, c="k", lw=1)
ax.plot(bc, br, marker="o", lw=1.8, color="#4c72b0")
ax.fill_between(bc, 0, br, where=br > 0, color="#4c72b0", alpha=0.25, label="LMC more certain")
ax.fill_between(bc, 0, br, where=br <= 0, color="#c44e52", alpha=0.25, label="LMC less certain")
ax.axvline(0.125, ls="-.", c="gray", lw=1)
ax.text(0.132, ax.get_ylim()[1] * 0.55, "mean NN\n0.125 km", fontsize=8, color="gray")
ax.axvline(0.25, ls="--", c="gray", lw=1)
ax.text(0.26, ax.get_ylim()[1] * 0.85, "grid spacing\n0.25 km", fontsize=8, color="gray")
ax.set_xlabel("distance to nearest sampling location (km)")
ax.set_ylabel("uncertainty reduction (%)")
ax.set_title("Borrowing strength is strictly local")
ax.legend(fontsize=9); ax.grid(alpha=0.25)
plt.tight_layout()
out3 = ROOT / "figures" / "stage9_uncertainty_vs_distance.png"
plt.savefig(out3, dpi=140)
print(f"\n[OK] {out3}")


# ---------- 潜在函数分解 (Q>=2) ----------
# 后验均值是各潜在函数贡献的精确可加分解:
#   f*(x) = m + sum_q [ k_q(x, X) .* B_q[Cd, i] ] @ alpha,   alpha = (K + s^2 I)^{-1} (y - m)
# 不是近似 —— 下面会断言各分量之和等于 GPyTorch 自己算的后验均值。
if Q >= 2:
    print(f"\n计算 {Q} 个潜在函数对后验均值的精确分解 ...")
    lmc.eval(); lik_lmc.eval()
    with torch.no_grad(), exact():
        # 必须用 .forward() 拿先验协方差。eval 模式下 lmc(Xtr, itr) 返回的是
        # 训练点上的后验, 用它算 alpha 会错。
        Kxx = lmc.forward(Xtr, itr).covariance_matrix
        Kn = Kxx + lik_lmc.noise * torch.eye(len(ytr), dtype=torch.float64)
        m_const = lmc.mean_module(Xtr)
        alpha = torch.linalg.solve(Kn, (ytr - m_const).unsqueeze(-1)).squeeze(-1)

        contribs = []
        for q in range(Q):
            Bq = lmc.task[q].covar_matrix.to_dense()        # (P, P)
            bq = Bq[0, itr]                                 # Cd 与每个训练观测所属任务的耦合
            parts = []
            for a in range(0, len(Xg), 4000):
                b = min(a + 4000, len(Xg))
                kq = lmc.spatial[q](Xg[a:b], Xtr).to_dense()    # (chunk, 977)
                parts.append(((kq * bq) @ alpha).numpy())
            contribs.append(np.concatenate(parts))

    total = np.sum(contribs, axis=0) + lmc.mean_module.constant.item()
    err = np.abs(total - m_lmc).max()
    print(f"  分解校验: max|sum_q contrib + m  -  GPyTorch 后验均值| = {err:.2e}")
    assert err < 1e-6, "分解与 GPyTorch 的后验均值不一致, 检查实现"

    fig2, axs = plt.subplots(1, Q + 1, figsize=(5.0 * (Q + 1), 4.6))
    axs = np.atleast_1d(axs)
    vlim = max(np.abs(c).max() for c in contribs)
    for q in range(Q):
        c = PAR["components"][q]
        ax = axs[q]
        im = ax.imshow(contribs[q].reshape(GRID_N, GRID_N), extent=extent, origin="lower",
                       cmap="RdBu_r", norm=Normalize(-vlim, vlim))
        ax.scatter(tr_xy[:, 0], tr_xy[:, 1], s=4, c="k", alpha=0.35)
        rng = "short-range" if (c["ls_x"] + c["ls_y"]) / 2 < 0.27 else "long-range"
        ax.set_title(f"Latent q={c['q']} ({rng})\n"
                     f"ℓ = ({c['ls_x']:.3f}, {c['ls_y']:.3f}) km\n"
                     + "var share: " + ", ".join(f"{n} {f*100:.0f}%"
                                                 for n, f in zip(NAMES, c["var_frac"])),
                     fontsize=9)
        ax.set_xlabel("X (km)"); ax.set_ylabel("Y (km)"); ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axs[Q]
    im = ax.imshow(total.reshape(GRID_N, GRID_N), extent=extent, origin="lower",
                   cmap="RdBu_r", norm=Normalize(-vlim, vlim))
    ax.scatter(tr_xy[:, 0], tr_xy[:, 1], s=4, c="k", alpha=0.35)
    ax.set_title("Sum = posterior mean\n(standardised log-Cd)", fontsize=9)
    ax.set_xlabel("X (km)"); ax.set_ylabel("Y (km)"); ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    out2 = ROOT / "figures" / "stage9_latent_decomposition.png"
    plt.savefig(out2, dpi=140)
    print(f"[OK] {out2}")

    print("\n各潜在函数在网格上的贡献幅度 (标准差):")
    for q in range(Q):
        c = PAR["components"][q]
        print(f"  q={c['q']}  ℓ=({c['ls_x']:.3f}, {c['ls_y']:.3f}) km  "
              f"贡献 std={contribs[q].std():.4f}  幅度占比={contribs[q].std()/sum(x.std() for x in contribs)*100:.1f}%")

print("\n完成。")
