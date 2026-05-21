import unittest
import numpy as np
from .tensor import Tensor  # 导入你的Tensor类
from .function import ReLU, Sigmoid, MaxPoolOp, Conv2dOp  # 导入你的算子


# ===================== 通用数值微分函数（适配mytorch.Tensor） =====================
def numerical_gradient(f, x: Tensor, eps=1e-4):
    """
    修正后的数值微分函数：逐元素扰动计算偏导数
    """
    grad = np.zeros_like(x.data)

    # 创建迭代器遍历数组中的每一个元素
    it = np.nditer(x.data, flags=['multi_index'], op_flags=['readwrite'])

    # 保存原始数据引用
    original_data = x.data.copy()

    while not it.finished:
        idx = it.multi_index
        old_val = original_data[idx]

        # 1. 计算 f(x + eps)
        x.data[idx] = old_val + eps
        y1 = f(x).data  # 获取标量值

        # 2. 计算 f(x - eps)
        x.data[idx] = old_val - eps
        y2 = f(x).data

        # 3. 计算该元素的梯度: (f(x+h) - f(x-h)) / 2h
        grad[idx] = (y1 - y2) / (2 * eps)

        # 4. 还原数据，准备下一个元素的计算
        x.data[idx] = old_val
        it.iternext()

    return grad

# ===================== 单个算子的梯度测试类 =====================
class TestReLU梯度检验(unittest.TestCase):
    """ReLU算子的梯度检验测试"""

    def setUp(self):
        """测试前置：生成随机输入（固定种子保证复现）"""
        np.random.seed(0)
        self.x = Tensor(np.random.randn(5, ), requires_grad=True, device="cpu")  # CPU模式先测试

    def test_relu_gradient(self):
        """测试ReLU的梯度是否正确"""

        # 定义目标函数（输出标量：求和）
        def f(x):
            relu = ReLU()
            out = relu.forward(x)
            return out.sum()

        # 1. 解析梯度（反向传播）
        y = f(self.x)
        y.backward()  # 你的Tensor需实现backward方法
        analytic_grad = self.x.grad.data

        # 2. 数值梯度（有限差分）
        num_grad = numerical_gradient(f, self.x, eps=1e-4)

        # 3. 验证梯度是否接近（默认rtol=1e-5, atol=1e-8）
        self.assertTrue(
            np.allclose(analytic_grad, num_grad),
            msg=f"ReLU梯度检验失败！解析梯度：{analytic_grad}, 数值梯度：{num_grad}"
        )


class TestConv2d梯度检验(unittest.TestCase):
    """Conv2d算子的梯度检验测试"""

    def setUp(self):
        """测试前置：构造卷积输入/权重/偏置"""
        np.random.seed(1)
        # 输入：N=1, C=1, H=4, W=4
        self.x = Tensor(np.random.randn(1, 1, 4, 4), requires_grad=True, device="cpu")
        # 权重：Out_C=1, In_C=1, K=2, K=2
        self.w = Tensor(np.random.randn(1, 1, 2, 2), requires_grad=True, device="cpu")
        # 偏置：Out_C=1
        self.b = Tensor(np.random.randn(1), requires_grad=True, device="cpu")

    def test_conv2d_x_gradient(self):
        """测试Conv2d输入x的梯度"""

        def f(x):
            conv = Conv2dOp()
            out = conv.forward(x, self.w, self.b, stride=1, padding=0)
            return out.sum()

        # 解析梯度
        y = f(self.x)
        y.backward()
        analytic_grad = self.x.grad.data

        # 数值梯度
        num_grad = numerical_gradient(f, self.x, eps=1e-4)

        # 验证（卷积允许稍大的误差，调整rtol）
        self.assertTrue(
            np.allclose(analytic_grad, num_grad, rtol=1e-3, atol=1e-5),
            msg="Conv2d输入x梯度检验失败"
        )

    def test_conv2d_w_gradient(self):
        """测试Conv2d权重w的梯度"""

        def f(w):
            conv = Conv2dOp()
            out = conv.forward(self.x, w, self.b, stride=1, padding=0)
            return out.sum()

        y = f(self.w)
        y.backward()
        analytic_grad = self.w.grad.data
        num_grad = numerical_gradient(f, self.w, eps=1e-4)

        self.assertTrue(
            np.allclose(analytic_grad, num_grad, rtol=1e-3, atol=1e-5),
            msg="Conv2d权重w梯度检验失败"
        )

    def test_conv2d_b_gradient(self):
        """测试Conv2d偏置b的梯度"""

        def f(b):
            conv = Conv2dOp()
            out = conv.forward(self.x, self.w, b, stride=1, padding=0)
            return out.sum()

        y = f(self.b)
        y.backward()
        analytic_grad = self.b.grad.data
        num_grad = numerical_gradient(f, self.b, eps=1e-4)

        self.assertTrue(
            np.allclose(analytic_grad, num_grad),
            msg="Conv2d偏置b梯度检验失败"
        )


class TestMaxPool梯度检验(unittest.TestCase):
    """MaxPool算子的梯度检验测试"""

    def setUp(self):
        np.random.seed(2)
        self.x = Tensor(np.random.randn(1, 1, 4, 4), requires_grad=True, device="cpu")
        self.kernel_size = 2
        self.stride = 2

    def test_maxpool_gradient(self):
        def f(x):
            pool = MaxPoolOp()
            out = pool.forward(x, self.kernel_size, self.stride, padding=0)
            return out.sum()

        # 解析梯度
        y = f(self.x)
        y.backward()
        analytic_grad = self.x.grad.data

        # 数值梯度
        num_grad = numerical_gradient(f, self.x, eps=1e-4)

        # 池化算子误差允许稍大
        self.assertTrue(
            np.allclose(analytic_grad, num_grad, rtol=1e-3, atol=1e-5),
            msg="MaxPool梯度检验失败"
        )


# ===================== 批量运行入口（可选） =====================
if __name__ == '__main__':
    # 运行当前文件的所有测试用例
    unittest.main(verbosity=2)  # verbosity=2显示详细测试日志