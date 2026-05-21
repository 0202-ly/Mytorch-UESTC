import sys
import os
import unittest
import numpy as np
import matplotlib.pyplot as plt

# 1. 后端检测
try:
    import cupy as cp

    GPU_AVAILABLE = True
except ImportError:
    cp = None
    GPU_AVAILABLE = False

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mytorch import Tensor
from mytorch.function import (
    Add, MatMul, ReLU, ELU, Sigmoid,
    Conv2dOp, MaxPoolOp, MinPoolOp, AvgPoolOp,
    ReshapeOp, MSE, LogSoftmaxOp, NLLLossOp, Sum
)

VIS_RESULTS = []


def to_numpy(data):
    """安全转换数据到 numpy，支持 GPU 和内存视图"""
    if data is None: return np.array([0.0])
    if GPU_AVAILABLE and hasattr(data, 'get'):
        return data.get()
    return np.array(data, copy=True)


def numerical_gradient(f, x: Tensor, eps=1e-4):
    """数值梯度计算 (有限差分法)"""
    is_cuda = (x.device() == 'cuda')
    x_numpy = to_numpy(x.data)
    grad_np = np.zeros_like(x_numpy)

    it = np.nditer(x_numpy, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        old_val = x_numpy[idx]

        # f(x + eps)
        x_numpy[idx] = old_val + eps
        x.data = cp.asarray(x_numpy) if is_cuda else x_numpy.copy()
        y1 = f(x).data
        y1_val = float(to_numpy(y1))

        # f(x - eps)
        x_numpy[idx] = old_val - eps
        x.data = cp.asarray(x_numpy) if is_cuda else x_numpy.copy()
        y2 = f(x).data
        y2_val = float(to_numpy(y2))

        grad_np[idx] = (y1_val - y2_val) / (2 * eps)
        x_numpy[idx] = old_val
        it.iternext()

    x.data = cp.asarray(x_numpy) if is_cuda else x_numpy.copy()
    return grad_np


class TestAllFunctionsGrad(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def check_op(self, f, x, name, rtol=1e-3, atol=1e-5):
        x.grad = None
        y = f(x)
        y.backward()

        analytic_grad = to_numpy(x.grad) if x.grad is not None else np.zeros_like(to_numpy(x.data))
        num_grad = numerical_gradient(f, x)

        diff = np.abs(analytic_grad - num_grad)
        denom = np.maximum(np.abs(analytic_grad), np.abs(num_grad)) + 1e-10
        mean_err = np.mean(diff / denom)

        VIS_RESULTS.append({
            'name': name, 'analytic': analytic_grad.flatten(),
            'numeric': num_grad.flatten(), 'mean_error': mean_err
        })

        res = np.allclose(analytic_grad, num_grad, rtol=rtol, atol=atol)
        self.assertTrue(res, f"{name} 检验失败! 误差: {mean_err:.2e}")
        print(f"PASS: {name:15} | 误差: {mean_err:.2e} | 后端: {x.device()}")

    def make_tensor(self, data, requires_grad=True):
        t = Tensor(data, requires_grad=requires_grad)
        if GPU_AVAILABLE: t.cuda()
        return t

    def test_all_ops(self):
        """修正后的算子测试：确保权重/目标值在 lambda 外部定义"""

        # Add
        b_add = self.make_tensor(np.ones((2, 2)))
        self.check_op(lambda t: Add()(t, b_add).sum(), self.make_tensor(np.random.randn(2, 2)), "Add")

        # MatMul
        w_matmul = self.make_tensor(np.random.randn(2, 3))
        self.check_op(lambda t: MatMul()(t, w_matmul).sum(), self.make_tensor(np.random.randn(1, 2)), "MatMul")

        # Sum
        self.check_op(lambda t: Sum()(t), self.make_tensor(np.random.randn(3, 3)), "Sum")

        # Activations
        self.check_op(lambda t: ReLU()(t).sum(), self.make_tensor(np.array([-1.2, 0.5, 2.3])), "ReLU")
        self.check_op(lambda t: ELU()(t).sum(), self.make_tensor(np.random.randn(3)), "ELU")
        self.check_op(lambda t: Sigmoid()(t).sum(), self.make_tensor(np.random.randn(3)), "Sigmoid")

        # Conv2d
        w_conv = self.make_tensor(np.random.randn(1, 1, 2, 2))
        b_conv = self.make_tensor(np.random.randn(1))
        self.check_op(lambda t: Conv2dOp()(t, w_conv, b_conv).sum(), self.make_tensor(np.random.randn(1, 1, 4, 4)),
                      "Conv2d")

        # Pooling
        pool_in = self.make_tensor(np.random.randn(1, 1, 4, 4) * 5)
        self.check_op(lambda t: MaxPoolOp()(t, 2, 2).sum(), pool_in, "MaxPool")
        self.check_op(lambda t: MinPoolOp()(t, 2, 2).sum(), pool_in, "MinPool")
        self.check_op(lambda t: AvgPoolOp()(t, 2, 2).sum(), pool_in, "AvgPool")

        # Reshape & Loss
        self.check_op(lambda t: ReshapeOp()(t, 1, 4).sum(), self.make_tensor(np.random.randn(2, 2)), "Reshape")

        target_mse = self.make_tensor(np.random.randn(2, 2), requires_grad=False)
        self.check_op(lambda t: MSE()(t, target_mse), self.make_tensor(np.random.randn(2, 2)), "MSE")

        target_ce = self.make_tensor(np.array([1]), requires_grad=False)
        self.check_op(lambda t: NLLLossOp()(LogSoftmaxOp()(t), target_ce), self.make_tensor(np.random.randn(1, 3)),
                      "CrossEntropy")


def generate_defense_plots():
    """生成答辩图表"""
    if not VIS_RESULTS: return
    print("\n--- 正在生成可视化图表 ---")
    plt.style.use('ggplot')

    # 图 1: 散点图
    all_a = np.concatenate([r['analytic'] for r in VIS_RESULTS])
    all_n = np.concatenate([r['numeric'] for r in VIS_RESULTS])
    plt.figure(figsize=(7, 7))
    plt.scatter(all_a, all_n, alpha=0.4, s=15, c='#34495e')
    lims = [min(all_a.min(), all_n.min()), max(all_a.max(), all_n.max())]
    plt.plot(lims, lims, 'r--', label='Ideal ($y=x$)')
    plt.title("Autograd Engine Consistency (GPU)")
    plt.xlabel("Analytical Gradient");
    plt.ylabel("Numerical Gradient")
    plt.legend();
    plt.savefig("defense_grad_consistency.png", dpi=300)

    # 图 2: 误差总结
    plt.figure(figsize=(10, 5))
    names = [r['name'] for r in VIS_RESULTS]
    errors = [r['mean_error'] for r in VIS_RESULTS]
    plt.bar(names, errors, color='skyblue', log=True)
    plt.axhline(y=1e-4, color='red', linestyle='--', label='Threshold')
    plt.title("Relative Error per Operator")
    plt.ylabel("Error (Log Scale)");
    plt.xticks(rotation=45)
    plt.tight_layout();
    plt.savefig("defense_ops_precision.png", dpi=300)
    print("已保存: defense_grad_consistency.png, defense_ops_precision.png")


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAllFunctionsGrad)
    unittest.TextTestRunner(verbosity=1).run(suite)
    generate_defense_plots()