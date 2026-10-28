"""
Jura 数据加载 + 探索性分析
输出：data/jura.csv, figures/jura_eda.png, 以及打印的相关性矩阵

跑这个之前确认 smoke_test.py 已经跑通。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.datasets import fetch_openml

# ---------- 目录准备 ----------
ROOT = Path(__file__).resolve().parent.parent   # 假设脚本在 src/ 下
for d in ["data", "figures", "results"]:
    (ROOT / d).mkdir(exist_ok=True)

CSV_PATH = ROOT / "data" / "jura.csv"

# ---------- 1. 拉数据（第一次会联网下载，之后走本地缓存） ----------
if CSV_PATH.exists():
    df = pd.read_csv(CSV_PATH)
    print(f"[cache] 从本地读取 {CSV_PATH}")
else:
    print("[download] 从 OpenML 拉取 Jura (id=41554)...")
    data = fetch_openml(data_id=41554, as_frame=True)
    df = data.frame
    df.to_csv(CSV_PATH, index=False)
    print(f"[saved] 已保存到 {CSV_PATH}")

print(f"\nshape: {df.shape}")
print(f"columns: {list(df.columns)}")
print(df.head())

# ---------- 2. 找出坐标列和金属列（列名大小写可能有出入，先自适应） ----------
cols_lower = {c.lower(): c for c in df.columns}

def pick(*candidates):
    for c in candidates:
        if c.lower() in cols_lower:
            return cols_lower[c.lower()]
    return None

X_COL = pick("Xloc", "X", "x")
Y_COL = pick("Yloc", "Y", "y")
METALS = [pick(m) for m in ["Cd", "Co", "Cr", "Cu", "Ni", "Pb", "Zn"]]
METALS = [m for m in METALS if m is not None]

print(f"\n坐标列: {X_COL}, {Y_COL}")
print(f"金属列: {METALS}")

# 本项目选定的三个输出（MOGP 文献标准组合）
TARGETS = [pick("Cd"), pick("Ni"), pick("Zn")]
print(f"选定输出: {TARGETS}")

# ---------- 3. 基础统计 + 偏度检查 ----------
print("\n" + "=" * 60)
print("原始浓度的描述统计（注意偏度 skew）")
print("=" * 60)
stats = df[METALS].describe().T
stats["skew"] = df[METALS].skew()
print(stats[["mean", "std", "min", "max", "skew"]].round(3))

# ---------- 4. log 变换前后的偏度对比 ----------
log_df = np.log(df[METALS])

print("\n" + "=" * 60)
print("log 变换对偏度的影响（三个选定输出）")
print("=" * 60)
print(f"  {'金属':<6}{'原始 skew':<14}{'log skew':<14}{'效果'}")
for t in TARGETS:
    s0, s1 = df[t].skew(), log_df[t].skew()
    verdict = "改善" if abs(s1) < abs(s0) else ">> 变差 <<"
    print(f"  {t:<6}{s0:<+14.3f}{s1:<+14.3f}{verdict}")

print("\n>> 注意 Ni: 原始分布本来就接近对称, log 之后反而变成左偏。")
print(">> 所以不能笼统说'浓度右偏所以取 log'。采用 log 的实际理由是:")
print(">>   (1) Cd 和 Zn 确有明显右偏, log 后接近对称")
print(">>   (2) 地球化学过程本质是乘性的, log 尺度上才是加性")
print(">>   (3) 跨输出协方差 B 需要三个输出在同一尺度上才有意义")
print(">> 报告里应按这三条写, 而不是'因为都右偏'。")

# ---------- 5. 相关性矩阵（这是报告里的核心图之一） ----------
print("\n" + "=" * 60)
print("log 浓度的 Pearson 相关矩阵（三个选定输出）")
print("=" * 60)
corr = log_df[TARGETS].corr()
print(corr.round(3))
print("\n>> 这个矩阵就是 MOGP 要利用的跨输出相关性，")
print(">> 也是你在报告里论证'为什么该用 MOGP'的直接证据")

# ---------- 6. 可视化 ----------
# 三行布局: 空间分布 / 变换前后直方图 / 相关性热图
fig = plt.figure(figsize=(15, 11.5))

# (a) 第 1 行: 三个金属的空间分布
for i, m in enumerate(TARGETS):
    ax = fig.add_subplot(3, 3, i + 1)
    sc = ax.scatter(df[X_COL], df[Y_COL], c=np.log(df[m]), cmap="viridis", s=18)
    ax.set_title(f"log({m}) spatial distribution")
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_aspect("equal")
    plt.colorbar(sc, ax=ax, fraction=0.046)

# (b) 第 2 行: 三个金属各自的 原始 vs log 直方图
#     只画 Cd 会误导 —— Cd 恰好是 log 效果最好的那个, 而 Ni 的 log 反而变差。
#     三个都画才是诚实的呈现, 也免得报告被抓。
for i, m in enumerate(TARGETS):
    s0, s1 = df[m].skew(), log_df[m].skew()
    worse = abs(s1) >= abs(s0)

    ax = fig.add_subplot(3, 6, 7 + 2 * i)
    ax.hist(df[m], bins=25, color="salmon", edgecolor="k", linewidth=0.4)
    ax.set_title(f"{m} raw\nskew {s0:+.2f}", fontsize=9.5)
    ax.tick_params(labelsize=7)

    ax = fig.add_subplot(3, 6, 8 + 2 * i)
    ax.hist(log_df[m], bins=25, color="lightcoral" if worse else "skyblue",
            edgecolor="k", linewidth=0.4)
    ax.set_title(f"log({m})\nskew {s1:+.2f}" + ("   worse" if worse else ""),
                 fontsize=9.5, color="firebrick" if worse else "black")
    ax.tick_params(labelsize=7)

# (c) 第 3 行居中: 相关性热图
ax = fig.add_subplot(3, 3, 8)
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(TARGETS)), TARGETS)
ax.set_yticks(range(len(TARGETS)), TARGETS)
for i in range(len(TARGETS)):
    for j in range(len(TARGETS)):
        ax.text(j, i, f"{corr.iloc[i, j]:.3f}", ha="center", va="center", fontsize=11)
ax.set_title("Cross-output correlation (log scale)\n— the structure MOGP exploits")
plt.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()
out = ROOT / "figures" / "jura_eda.png"
plt.savefig(out, dpi=130)
print(f"\n[OK] EDA 图已保存: {out}")