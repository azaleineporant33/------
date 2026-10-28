"""
第二次汇报: 公平比较 MOGP 与线性回归 (规则见 ../方案.md)

参赛者:
  回归 A  Cd ~ Ni + Zn
  回归 B  Cd ~ Ni + Zn + X + Y + X² + Y² + X·Y          (主要对手)
  回归 C  回归 B + 岩石类型 + 土地用途                   (附加比较: 比 MOGP 多用了信息)
  MOGP    第一次汇报的 LMC (Q=2), 设置不变

公平规则落实在代码里的位置:
  规则 1 同一张考卷      make_split() 与 ../../src/07_robustness.py 完全相同, seed 0-9
  规则 2 同样的复习资料  标准化只用训练数据; 测试点的 Cd 只在打分时使用
  规则 3 赛前定死        MOGP 设置照抄 07; 回归公式和类别合并规则写死在下面
  规则 4 同样的评分      RMSE + 95% 区间覆盖率 (另记 MAE、区间平均宽度)
  规则 5 配对比较        每个划分算 "回归 RMSE - MOGP RMSE", 10 个差值做配对 t 检验
  规则 6 赛前判定标准    judge() 照 ../方案.md 第 5 节写死

输出 (../results/):
  per_split.csv            每次划分、每个模型的成绩
  summary.csv              汇总表 (PPT 表 1)
  paired_tests.csv         MOGP 对每个回归的配对检验
  ftest_quadratic.csv      回归 B 平方项的 F 检验 (只用训练数据)
  coefficients_seed0.csv   三个回归的系数表 (划分 0)
  seed0_test_predictions.csv  划分 0 测试点的预测 (画散点图用)
  seed0_train_residuals_B.csv 划分 0 回归 B 的训练残差 (画残差地图用)
"""

import sys
import time
from pathlib import Path

import gpytorch
import numpy as np
import pandas as pd
import torch
from scipy import stats

HERE = Path(__file__).resolve().parent.parent          # 第二次汇报/
ROOT = HERE.parent                                     # E:\Research
sys.path.insert(0, str(ROOT / "src"))
from models import exact, TrueLMC                      # 与第一次汇报同一个模型定义

OUT = HERE / "results"
OUT.mkdir(exist_ok=True)

# ---------- 与 07_robustness.py 完全一致的设置 (规则 1、3) ----------
N_SPLITS = 10
N_TEST = 100
N_ITER = 500
LR = 0.05
Q_BEST = 2
LS_INIT_LMC = [0.08, 3.0]
TARGET_NAMES = ["Cd", "Ni", "Zn"]

# ---------- 回归的赛前设定 (规则 3) ----------
MIN_CATEGORY_N = 10        # 少于 10 个点的类别, 并入该组点最多的那一类
ALPHA = 0.05               # 判定标准 (规则 6)
Z95 = stats.norm.ppf(0.975)

MODELS = ["reg_A", "reg_B", "reg_C", "mogp"]
LABELS = {"reg_A": "回归 A", "reg_B": "回归 B", "reg_C": "回归 C", "mogp": "MOGP"}

# ---------- 数据 ----------
df = pd.read_csv(ROOT / "data" / "jura.csv")
coords = df[["Xloc", "Yloc"]].to_numpy(dtype=np.float64)
Y_log = np.log(df[TARGET_NAMES].to_numpy(dtype=np.float64))
N, P = Y_log.shape

# 坐标减去全部 359 个点的平均位置: 不改变预测, 只让一次项系数更好解释
# (坐标不含 Cd 信息, 用全部点算不构成泄漏)
xc = coords[:, 0] - coords[:, 0].mean()
yc = coords[:, 1] - coords[:, 1].mean()
SPATIAL = np.column_stack([xc, yc, xc ** 2, yc ** 2, xc * yc])
SPATIAL_NAMES = ["X", "Y", "X²", "Y²", "X·Y"]


def category_dummies(prefix):
    """0/1 列; 小类别并入最大类别, 最大类别作为参照类 (不单独成列)。"""
    block = df.filter(like=prefix).copy()
    counts = block.sum(axis=0)
    biggest = counts.idxmax()
    merged = [c for c in counts.index if counts[c] < MIN_CATEGORY_N]
    for c in merged:
        block[biggest] += block[c]
        block = block.drop(columns=c)
    assert (block.sum(axis=1) == 1).all()
    return block.drop(columns=biggest), biggest, merged, counts


ROCK, ROCK_REF, ROCK_MERGED, ROCK_COUNTS = category_dummies("Rock_")
LAND, LAND_REF, LAND_MERGED, LAND_COUNTS = category_dummies("Landuse_")
CATEG = np.column_stack([ROCK.to_numpy(float), LAND.to_numpy(float)])
CATEG_NAMES = list(ROCK.columns) + list(LAND.columns)

print(f"{N} 个位置。岩石参照类 {ROCK_REF}, 并入它的小类别 {ROCK_MERGED}; "
      f"土地用途参照类 {LAND_REF}, 并入它的小类别 {LAND_MERGED}\n")


def make_split(seed):
    """照抄 07_robustness.py 的划分 + 防泄漏标准化 (规则 1、2)。"""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    test_idx = np.sort(perm[:N_TEST])
    train_idx = np.sort(perm[N_TEST:])

    mu, sd = np.zeros(P), np.zeros(P)
    mu[0], sd[0] = Y_log[train_idx, 0].mean(), Y_log[train_idx, 0].std()   # Cd 只用训练位置
    for p in [1, 2]:
        mu[p], sd[p] = Y_log[:, p].mean(), Y_log[:, p].std()               # Ni/Zn 全部可见
    Y_std = (Y_log - mu) / sd
    return train_idx, test_idx, Y_std


# ============================================================
# 线性回归 (普通最小二乘 + 课本上的预测区间)
# ============================================================
def design(kind, Y_std):
    """返回全部 359 个点的设计矩阵和列名, 训练/测试再按下标取行。"""
    cols = [np.ones(N), Y_std[:, 1], Y_std[:, 2]]
    names = ["截距", "Ni", "Zn"]
    if kind in ("reg_B", "reg_C", "reg_B_linear"):
        k = 2 if kind == "reg_B_linear" else 5
        cols += list(SPATIAL[:, :k].T)
        names += SPATIAL_NAMES[:k]
    if kind == "reg_C":
        cols += list(CATEG.T)
        names += CATEG_NAMES
    return np.column_stack(cols), names


def ols(Xd, y):
    n, k = Xd.shape
    XtX_inv = np.linalg.inv(Xd.T @ Xd)
    beta = XtX_inv @ Xd.T @ y
    resid = y - Xd @ beta
    df_res = n - k
    s2 = resid @ resid / df_res
    return dict(beta=beta, XtX_inv=XtX_inv, s2=s2, df=df_res, resid=resid,
                rss=resid @ resid, tss=((y - y.mean()) ** 2).sum(), n=n, k=k)


def ols_predict(fit, X0):
    """预测值 ± t 分位数 × sqrt(s²·(1 + x0ᵀ(XᵀX)⁻¹x0)) —— 新观测的 95% 预测区间。"""
    mean = X0 @ fit["beta"]
    lev = np.einsum("ij,jk,ik->i", X0, fit["XtX_inv"], X0)
    half = stats.t.ppf(0.975, fit["df"]) * np.sqrt(fit["s2"] * (1 + lev))
    return mean, mean - half, mean + half


# ============================================================
# MOGP (与 07_robustness.py 的 LMC 完全一致)
# ============================================================
def run_mogp(train_idx, test_idx, Y_std, seed):
    t = lambda a: torch.tensor(a, dtype=torch.float64)
    Xtr = t(np.concatenate([coords[train_idx], coords, coords]))
    itr = torch.tensor(np.concatenate([np.zeros(len(train_idx)), np.ones(N),
                                       np.full(N, 2)]).astype(np.int64))
    ytr = t(np.concatenate([Y_std[train_idx, 0], Y_std[:, 1], Y_std[:, 2]]))
    Xte = t(coords[test_idx])
    ite = torch.zeros(len(test_idx), dtype=torch.long)

    torch.manual_seed(seed)
    lik = gpytorch.likelihoods.GaussianLikelihood().double()
    model = TrueLMC(Xtr, itr, ytr, lik, P, Q_BEST).double()
    for q in range(Q_BEST):
        model.spatial[q].lengthscale = torch.tensor(
            [[LS_INIT_LMC[q], LS_INIT_LMC[q]]], dtype=torch.float64)

    model.train(); lik.train()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)
    with exact():
        for _ in range(N_ITER):
            opt.zero_grad()
            loss = -mll(model(Xtr, itr), ytr)
            loss.backward()
            opt.step()

    model.eval(); lik.eval()
    with torch.no_grad(), exact():
        pred = lik(model(Xte, ite))           # 含观测噪声: 预测"新测一次"的值, 与回归口径一致
        mean = pred.mean.numpy()
        half = Z95 * np.sqrt(pred.variance.numpy())
    return mean, mean - half, mean + half


def score(y, mean, lo, hi):
    return dict(rmse=np.sqrt(np.mean((mean - y) ** 2)),
                mae=np.mean(np.abs(mean - y)),
                cover=np.mean((y >= lo) & (y <= hi)),
                width=np.mean(hi - lo))


# ============================================================
# 主循环
# ============================================================
rows, ftests = [], []
t0 = time.time()
for seed in range(N_SPLITS):
    train_idx, test_idx, Y_std = make_split(seed)
    y_tr, y_te = Y_std[train_idx, 0], Y_std[test_idx, 0]
    rec = dict(seed=seed)
    preds = {}

    fits = {}
    for kind in ["reg_A", "reg_B", "reg_C", "reg_B_linear"]:
        Xd, names = design(kind, Y_std)
        fits[kind] = (ols(Xd[train_idx], y_tr), names)
        if kind == "reg_B_linear":
            continue
        preds[kind] = ols_predict(fits[kind][0], Xd[test_idx])

    # 回归 B 平方项的 F 检验 (只用训练数据; 结果不影响回归 B 的公式)
    small, big = fits["reg_B_linear"][0], fits["reg_B"][0]
    q_extra = big["k"] - small["k"]
    F = ((small["rss"] - big["rss"]) / q_extra) / (big["rss"] / big["df"])
    ftests.append(dict(seed=seed, F=F, df1=q_extra, df2=big["df"],
                       p=stats.f.sf(F, q_extra, big["df"])))

    preds["mogp"] = run_mogp(train_idx, test_idx, Y_std, seed)

    for m in MODELS:
        for key, val in score(y_te, *preds[m]).items():
            rec[f"{m}_{key}"] = val
    rows.append(rec)
    print(f"  划分 {seed}: " + "  ".join(f"{LABELS[m]}={rec[m + '_rmse']:.4f}" for m in MODELS)
          + f"   ({time.time() - t0:.0f}s)", flush=True)

    if seed == 0:
        # 画图用: 测试点预测 + 回归 B 训练残差
        out = pd.DataFrame(dict(idx=test_idx, x=coords[test_idx, 0], y=coords[test_idx, 1],
                                true=y_te))
        for m in MODELS:
            out[f"{m}_pred"], out[f"{m}_lo"], out[f"{m}_hi"] = preds[m]
        out.to_csv(OUT / "seed0_test_predictions.csv", index=False)
        pd.DataFrame(dict(idx=train_idx, x=coords[train_idx, 0], y=coords[train_idx, 1],
                          resid=fits["reg_B"][0]["resid"])).to_csv(
            OUT / "seed0_train_residuals_B.csv", index=False)
        pd.DataFrame(dict(idx=test_idx, x=coords[test_idx, 0], y=coords[test_idx, 1])).to_csv(
            OUT / "seed0_test_points.csv", index=False)

        # 系数表 (备用页)
        coef_rows = []
        for kind in ["reg_A", "reg_B", "reg_C"]:
            fit, names = fits[kind]
            se = np.sqrt(np.diag(fit["XtX_inv"]) * fit["s2"])
            tval = fit["beta"] / se
            r2 = 1 - fit["rss"] / fit["tss"]
            adj = 1 - (1 - r2) * (fit["n"] - 1) / fit["df"]
            for nm, b, s, tv in zip(names, fit["beta"], se, tval):
                coef_rows.append(dict(model=LABELS[kind], term=nm, coef=b, se=s, t=tv,
                                      p=2 * stats.t.sf(abs(tv), fit["df"]),
                                      r2=r2, adj_r2=adj, n=fit["n"]))
        pd.DataFrame(coef_rows).to_csv(OUT / "coefficients_seed0.csv", index=False,
                                       encoding="utf-8-sig")

D = pd.DataFrame(rows)
D.to_csv(OUT / "per_split.csv", index=False)
FT = pd.DataFrame(ftests)
FT.to_csv(OUT / "ftest_quadratic.csv", index=False)
print(f"\n用时 {time.time() - t0:.0f}s")

# ---------- 自检: MOGP 必须与第一次汇报逐位一致 ----------
old = pd.read_csv(ROOT / "results" / "stage10_robustness.csv")
diff = np.abs(D["mogp_rmse"].to_numpy() - old["lmc_rmse"].to_numpy()).max()
print(f"\n自检: MOGP RMSE 与 stage10_robustness.csv 的最大差 = {diff:.2e}  "
      f"({'一致' if diff < 1e-6 else '>> 不一致, 需要检查 <<'})")

# ---------- 汇总表 (PPT 表 1) ----------
summary = []
for m in MODELS:
    summary.append(dict(model=LABELS[m],
                        rmse_mean=D[f"{m}_rmse"].mean(), rmse_sd=D[f"{m}_rmse"].std(),
                        mae_mean=D[f"{m}_mae"].mean(),
                        cover_mean=D[f"{m}_cover"].mean(), cover_min=D[f"{m}_cover"].min(),
                        cover_max=D[f"{m}_cover"].max(),
                        width_mean=D[f"{m}_width"].mean()))
S = pd.DataFrame(summary)
S.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")

# ---------- 配对检验 (规则 5) ----------
tests = []
for m in ["reg_A", "reg_B", "reg_C"]:
    d = D[f"{m}_rmse"] - D["mogp_rmse"]          # > 0 表示 MOGP 错得更少
    t_res = stats.ttest_rel(D[f"{m}_rmse"], D["mogp_rmse"])
    ci = t_res.confidence_interval(0.95)
    tests.append(dict(vs=LABELS[m], mean_diff=d.mean(), ci_low=ci.low, ci_high=ci.high,
                      rel_pct=(d / D[f"{m}_rmse"]).mean() * 100,
                      p_t=t_res.pvalue, p_wilcoxon=stats.wilcoxon(d).pvalue,
                      mogp_wins=int((d > 0).sum())))
T = pd.DataFrame(tests)
T.to_csv(OUT / "paired_tests.csv", index=False, encoding="utf-8-sig")

# ---------- 打印 ----------
print("\n" + "=" * 72)
print("表 1  10 次划分汇总 (Cd 的标准化 log 尺度)")
print("=" * 72)
print(f"  {'模型':<8}{'RMSE 均值±标准差':<20}{'MAE':<9}{'95%区间覆盖率':<18}{'区间平均宽度'}")
for _, r in S.iterrows():
    print(f"  {r.model:<8}{r.rmse_mean:.4f} ± {r.rmse_sd:.4f}     {r.mae_mean:.4f}   "
          f"{r.cover_mean:.1%} ({r.cover_min:.0%}-{r.cover_max:.0%})     {r.width_mean:.3f}")

print("\n配对检验: 回归 RMSE − MOGP RMSE  (> 0 表示 MOGP 更准)")
for _, r in T.iterrows():
    print(f"  MOGP vs {r.vs}: 平均差 {r.mean_diff:+.4f}  95%CI [{r.ci_low:+.4f}, {r.ci_high:+.4f}]"
          f"  相对 {r.rel_pct:+.1f}%  配对t p={r.p_t:.2g}  Wilcoxon p={r.p_wilcoxon:.2g}"
          f"  MOGP 赢 {r.mogp_wins}/10")

print(f"\n回归 B 平方项 F 检验 (训练数据): 划分 0 的 F={FT.F[0]:.2f}, p={FT.p[0]:.2g}; "
      f"10 次里 p<0.05 的有 {(FT.p < ALPHA).sum()} 次")


# ---------- 判定 (规则 6, 照 ../方案.md 第 5 节) ----------
def judge(row):
    if row.p_t < ALPHA and row.mean_diff > 0:
        return "MOGP 比回归更准"
    if row.p_t < ALPHA and row.mean_diff < 0:
        return "回归比 MOGP 更准"
    return "两者准确度没有明显差别"


print("\n判定 (赛前标准: 配对 t 检验 p < 0.05 且平均差 > 0):")
for _, r in T.iterrows():
    tag = "主比较" if r.vs == "回归 B" else ("附加比较, 回归 C 多用了岩石和土地用途"
                                         if r.vs == "回归 C" else "参考")
    print(f"  MOGP vs {r.vs} ({tag}): {judge(r)}")
best = S.iloc[(S.cover_mean - 0.95).abs().argmin()].model
print(f"  覆盖率最接近 95% 的: {best}")
