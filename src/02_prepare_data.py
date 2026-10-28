"""
Jura 数据准备：log 变换 -> 标准化 -> heterotopic 划分 -> 存成 npz

输出 data/jura_prepared.npz，含长表格式的训练/测试张量，可直接喂给 GPyTorch。

实验设定（Jura 在 MOGP 文献里的标准设定）：
  - Cd 是"昂贵"的主变量：只在 259 个位置观测，100 个位置藏起来做测试
  - Ni / Zn 是"便宜"的辅助变量：全部 359 个位置都观测
  - 任务：在那 100 个位置上预测 Cd
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "jura.csv"
OUT_PATH = ROOT / "data" / "jura_prepared.npz"

SEED = 0
N_TEST = 100                      # 藏起来的 Cd 观测数
TARGET_NAMES = ["Cd", "Ni", "Zn"] # 索引 0 = Cd(主), 1 = Ni, 2 = Zn

rng = np.random.default_rng(SEED)

# ---------- 1. 读数据 ----------
df = pd.read_csv(CSV_PATH)
cols_lower = {c.lower(): c for c in df.columns}
def pick(name):
    return cols_lower[name.lower()]

X_COL, Y_COL = pick("Xloc") if "xloc" in cols_lower else pick("X"), \
               pick("Yloc") if "yloc" in cols_lower else pick("Y")
TARGETS = [pick(t) for t in TARGET_NAMES]

coords = df[[X_COL, Y_COL]].to_numpy(dtype=np.float64)   # (359, 2)
Y_raw = df[TARGETS].to_numpy(dtype=np.float64)           # (359, 3)
N, P = Y_raw.shape
print(f"载入 {N} 个位置, {P} 个输出: {TARGET_NAMES}")
print(f"坐标范围: X [{coords[:,0].min():.2f}, {coords[:,0].max():.2f}], "
      f"Y [{coords[:,1].min():.2f}, {coords[:,1].max():.2f}]")

# ---------- 2. log 变换 ----------
# 浓度右偏 -> log 后近似对称, 匹配 GP 的高斯似然假设
assert (Y_raw > 0).all(), "有非正浓度值, log 会出 NaN"
Y_log = np.log(Y_raw)

# ---------- 3. heterotopic 划分 ----------
# 随机选 100 个位置, 把它们的 Cd 藏起来
perm = rng.permutation(N)
test_idx = np.sort(perm[:N_TEST])        # Cd 被藏起来的位置
train_idx_cd = np.sort(perm[N_TEST:])    # Cd 可见的 259 个位置

print(f"\nCd 训练位置: {len(train_idx_cd)}, Cd 测试位置: {len(test_idx)}")
print(f"Ni/Zn 训练位置: {N} (全部, 包括测试位置)")

# ---------- 4. 标准化 ----------
# 关键: 每个输出各自标准化, 且只用该输出的"训练观测"来算均值/标准差
#       (Cd 用 259 个, Ni/Zn 用 359 个) -> 避免测试信息泄漏
mu = np.zeros(P)
sd = np.zeros(P)

# Cd (task 0): 只用训练位置
mu[0] = Y_log[train_idx_cd, 0].mean()
sd[0] = Y_log[train_idx_cd, 0].std()
# Ni, Zn: 全部位置都是训练数据
for p in [1, 2]:
    mu[p] = Y_log[:, p].mean()
    sd[p] = Y_log[:, p].std()

Y_std = (Y_log - mu) / sd
print("\n标准化参数 (log 尺度):")
for p, name in enumerate(TARGET_NAMES):
    print(f"  {name}: mean={mu[p]:.4f}, std={sd[p]:.4f}")

# ---------- 5. 组装长表 ----------
# 每一行 = (x坐标, y坐标, 任务编号, 观测值)
X_list, i_list, y_list = [], [], []

# task 0: Cd, 只有 259 个训练观测
X_list.append(coords[train_idx_cd])
i_list.append(np.zeros(len(train_idx_cd), dtype=np.int64))
y_list.append(Y_std[train_idx_cd, 0])

# task 1, 2: Ni, Zn, 全部 359 个
for p in [1, 2]:
    X_list.append(coords)
    i_list.append(np.full(N, p, dtype=np.int64))
    y_list.append(Y_std[:, p])

X_train = np.concatenate(X_list, axis=0)     # (977, 2)
i_train = np.concatenate(i_list, axis=0)     # (977,)
y_train = np.concatenate(y_list, axis=0)     # (977,)

# 测试集: 只有 Cd, 在被藏起来的 100 个位置
X_test = coords[test_idx]                    # (100, 2)
i_test = np.zeros(N_TEST, dtype=np.int64)
y_test = Y_std[test_idx, 0]                  # 标准化尺度的真值

print(f"\n长表训练集: X {X_train.shape}, i {i_train.shape}, y {y_train.shape}")
print(f"  各任务观测数: {np.bincount(i_train)}")
print(f"长表测试集: X {X_test.shape}, y {y_test.shape}")

# ---------- 6. 保存 ----------
np.savez(
    OUT_PATH,
    X_train=X_train, i_train=i_train, y_train=y_train,
    X_test=X_test, i_test=i_test, y_test=y_test,
    # 反标准化 / 反 log 需要的东西
    mu=mu, sd=sd,
    # 完整数据, 画图和做独立 GP 基线时会用到
    coords=coords, Y_log=Y_log, Y_std=Y_std,
    train_idx_cd=train_idx_cd, test_idx=test_idx,
    target_names=np.array(TARGET_NAMES),
)
print(f"\n[OK] 已保存: {OUT_PATH}")

# ---------- 7. 自检 ----------
print("\n" + "=" * 55)
print("自检")
print("=" * 55)
print(f"训练集里 Cd 的观测数 = {(i_train == 0).sum()}  (应为 259)")
print(f"测试位置有没有混进 Cd 训练集: "
      f"{len(np.intersect1d(train_idx_cd, test_idx))} 个重叠  (应为 0)")
print(f"标准化后 Cd 训练值: mean={y_train[i_train==0].mean():.6f} "
      f"std={y_train[i_train==0].std():.6f}  (应为 0 和 1)")
print(f"测试集 Cd 真值范围: [{y_test.min():.2f}, {y_test.max():.2f}]")