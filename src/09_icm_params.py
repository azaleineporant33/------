"""
补齐 ICM 的学到参数 (阶段 8 只存了 ICM 的指标, 没存参数)

报告里要论证 "ICM 被迫在双尺度中二选一, 选了短程", 就需要:
  ICM 的单一 lengthscale   vs   LMC 的 (ell_1, ell_2)
  ICM 的 B                 vs   LMC 的 B_1, B_2

精确 Cholesky 下训练是确定性的, 所以这里重训得到的就是阶段 8 的那个 ICM。
脚本会与 results/stage8_all_models.csv 记录的指标核对一致性。

输出: results/stage8_icm_params.json
"""

import contextlib
import json
import numpy as np
import pandas as pd
import torch
import gpytorch
from pathlib import Path

ROOT = Path(r"E:\Research")
DATA = np.load(ROOT / "data" / "jura_prepared.npz", allow_pickle=True)

N_ITER = 500
LR = 0.05
LS_INIT = 0.5          # 与阶段 8 seed=0 一致
SEED = 0
RANK = 1               # 阶段 8 证实 rank>1 完全冗余


@contextlib.contextmanager
def exact():
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
NAMES = [str(n) for n in DATA["target_names"]]
P = len(NAMES)


class ICMModel(gpytorch.models.ExactGP):
    def __init__(self, x, i, y, lik, num_tasks, rank):
        super().__init__((x, i), y, lik)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.RBFKernel(ard_num_dims=2)
        self.task_covar_module = gpytorch.kernels.IndexKernel(num_tasks=num_tasks, rank=rank)

    def forward(self, x, i):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x),
            self.covar_module(x).mul(self.task_covar_module(i)))


torch.manual_seed(SEED)
lik = gpytorch.likelihoods.GaussianLikelihood().double()
icm = ICMModel(Xtr, itr, ytr, lik, P, RANK).double()
icm.covar_module.lengthscale = torch.tensor([[LS_INIT, LS_INIT]], dtype=torch.float64)

icm.train(); lik.train()
opt = torch.optim.Adam(icm.parameters(), lr=LR)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, icm)
print("训练 ICM (rank=1, 精确 Cholesky) ...")
with exact():
    for _ in range(N_ITER):
        opt.zero_grad()
        loss = -mll(icm(Xtr, itr), ytr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        final_loss = -mll(icm(Xtr, itr), ytr).item()

icm.eval(); lik.eval()
with torch.no_grad(), exact():
    pred = lik(icm(Xte, ite))
    rmse = torch.sqrt(((pred.mean - yte) ** 2).mean()).item()
    nll = -pred.log_prob(yte).item() / len(yte)

# ---------- 与阶段 8 核对 ----------
ref = pd.read_csv(ROOT / "results" / "stage8_all_models.csv")
ref = ref[(ref.kind == "icm") & (ref["param"] == RANK) & (ref.seed == SEED)].iloc[0]
print(f"\n一致性核对 (应与阶段 8 完全相同):")
print(f"  loss: 本次 {final_loss:.6f}  阶段8 {ref['loss']:.6f}  差 {abs(final_loss-ref['loss']):.2e}")
print(f"  RMSE: 本次 {rmse:.6f}  阶段8 {ref['rmse']:.6f}  差 {abs(rmse-ref['rmse']):.2e}")
assert abs(final_loss - ref["loss"]) < 1e-6, "与阶段 8 不一致, 检查随机种子或超参"

ls = icm.covar_module.lengthscale.detach().numpy().ravel()
B = icm.task_covar_module.covar_matrix.to_dense().detach().numpy()
D = np.sqrt(np.diag(B))
Bc = B / np.outer(D, D)

print("\n" + "=" * 62)
print("ICM 学到的参数")
print("=" * 62)
print(f"\n  唯一的 lengthscale: X={ls[0]:.4f} km, Y={ls[1]:.4f} km  (几何均值 {np.sqrt(ls[0]*ls[1]):.4f})")
print(f"  噪声方差: {lik.noise.item():.4f}")
print(f"  常数均值: {icm.mean_module.constant.item():.4f}")
print(f"\n  B 的对角 (各任务先验方差): " + ", ".join(f"{n}={B[i,i]:.4f}" for i, n in enumerate(NAMES)))
print("\n  B 相关矩阵:")
for r in range(P):
    print(f"    {NAMES[r]:<4}" + "  ".join(f"{Bc[r,c]:+.3f}" for c in range(P)))

# ---------- 与 LMC 对照 ----------
LMC = json.load(open(ROOT / "results" / "stage8_best_params.json", encoding="utf-8"))
print("\n" + "=" * 62)
print("ICM vs LMC(Q=2): ICM 被迫二选一, 选了短程")
print("=" * 62)
def corr_of(Bmat):
    Bm = np.array(Bmat)
    d = np.sqrt(np.diag(Bm))
    return Bm / np.outer(d, d)


print(f"\n  {'模型':<14}{'lengthscale (km)':<24}{'Cd-Ni 相关性':<16}")
print(f"  {'ICM':<14}{'(%.3f, %.3f)' % (ls[0], ls[1]):<24}{Bc[0, 1]:<16.3f}")
for c in LMC["components"]:
    lab = "LMC q=%d" % c["q"]
    lsq = "(%.3f, %.3f)" % (c["ls_x"], c["ls_y"])
    print(f"  {lab:<14}{lsq:<24}{corr_of(c['B'])[0, 1]:<16.3f}")

c1, c2 = LMC["components"]
print(f"\n  >> ICM 的 ({ls[0]:.3f}, {ls[1]:.3f}) 紧贴 LMC 的短程分量 "
      f"({c1['ls_x']:.3f}, {c1['ls_y']:.3f}),")
print(f"     长程分量 ({c2['ls_x']:.3f}, {c2['ls_y']:.3f}) 完全丢失。")
print(f"  >> 代价: ICM 只有一个 B, 相关性 {Bc[0,1]:.3f} 是两个尺度的加权混合,")
print(f"     掩盖了长程分量高达 {corr_of(c2['B'])[0,1]:.3f} 的近乎完美相关 "
      f"(短程分量只有 {corr_of(c1['B'])[0,1]:.3f})。")

out = dict(
    rank=RANK, seed=SEED, loss=final_loss, rmse=rmse, nll=nll,
    lengthscale=[float(ls[0]), float(ls[1])],
    noise=float(lik.noise.item()),
    mean_constant=float(icm.mean_module.constant.item()),
    B=B.tolist(), B_corr=Bc.tolist(), target_names=NAMES,
)
with open(ROOT / "results" / "stage8_icm_params.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"\n[OK] {ROOT/'results'/'stage8_icm_params.json'}")
