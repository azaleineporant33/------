"""
共享的模型定义与数值上下文。

存在的理由: 05/06/07/09 各自复制了一份模型类, 有静默漂移的风险 ——
改了一处忘了另一处, 图和表就会来自不同的模型。新脚本一律从这里 import。
(05-09 暂未重构, 它们当前已验证一致: 07 的 split 0 精确复现了 08 的数字。)
"""

import contextlib

import gpytorch
import torch

__all__ = ["exact", "SingleGP", "ICMModel", "TrueLMC"]


@contextlib.contextmanager
def exact():
    """
    强制精确 Cholesky, 关掉所有随机近似。

    GPyTorch 的 max_cholesky_size 默认 800。本项目训练集 N=977 > 800,
    不套这个上下文就会走 CG + Lanczos 随机估计: 同一组参数重复求值极差 0.0095,
    且系统性高估对数似然约 0.03。训练和预测都必须套。
    """
    with gpytorch.settings.max_cholesky_size(10_000), \
         gpytorch.settings.fast_computations(
             covar_root_decomposition=False, log_prob=False, solves=False
         ):
        yield


class SingleGP(gpytorch.models.ExactGP):
    """单输出 GP 基线: 只看 Cd。

    这里要 ScaleKernel —— 没有 B 来提供 output scale, 必须单独有一个。
    多输出模型则不能加 (与 B 的对角元不可辨识)。这个不对称是有意的。
    """

    def __init__(self, x, y, likelihood):
        super().__init__(x, y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=2)
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


class ICMModel(gpytorch.models.ExactGP):
    """ICM: K[(x,i),(x',j)] = K_RBF(x,x') * B[i,j]。单一空间尺度, 所有任务共享。

    rank 对 P=3 无影响: B = W W^T + diag(kappa) 在 rank=1 时已有 6 个自由参数,
    等于 3x3 PSD 矩阵的自由度, 已张满可行空间。实测 rank=1/2/3 的 loss
    极差 1.3e-9。
    """

    def __init__(self, x, i, y, likelihood, num_tasks, rank=1):
        super().__init__((x, i), y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.RBFKernel(ard_num_dims=2)
        self.task_covar_module = gpytorch.kernels.IndexKernel(
            num_tasks=num_tasks, rank=rank
        )

    def forward(self, x, i):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x),
            self.covar_module(x).mul(self.task_covar_module(i)),
        )


class TrueLMC(gpytorch.models.ExactGP):
    """LMC: K[(x,i),(x',j)] = sum_q K_q(x,x') * B_q[i,j]。

    每个 K_q 有独立的 ARD lengthscale —— 这是与 ICM 的唯一区别, 也是 LMC 的定义。
    每个 B_q 取 IndexKernel(rank=1), 即 B_q = a_q a_q^T + diag(kappa_q)。

    自洽性检验: Q=1 时应与 ICM(rank=1) 完全相同。实测两者 loss 均为 1.004174。

    注意: 各 q 的 lengthscale 初值必须拉开, 否则梯度对称,
          所有 q 收敛到同一个解, Q>1 会静默退化成 ICM。
    """

    def __init__(self, x, i, y, likelihood, num_tasks, Q):
        super().__init__((x, i), y, likelihood)
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
