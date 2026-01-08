from graphviz import Digraph
import numpy as np
from .function import Function

import matplotlib.pyplot as plt
from .tensor import Tensor, GPU_AVAILABLE

# 如果存在 cupy 则导入
if GPU_AVAILABLE:
    import cupy as cp


def grad_check_model(model, loss_fn, inputs, targets, eps=1e-4, threshold=1e-3):
    """
    对模型的可训练参数进行首批次梯度检验。
    修正了排版截断问题，并将汇总逻辑移出循环。
    """
    print("\n" + "=" * 50)
    print("梯度检验")
    print("=" * 50)

    model.train()
    # 强制使用 float64 提高检验精度
    for p in model.parameters():
        p.data = p.xp.array(p.data, dtype=np.float64)

    outputs = model(inputs)
    loss = loss_fn(outputs, targets)

    for p in model.parameters():
        p.grad = None
    loss.backward()

    all_analytic = []
    all_numeric = []

    # 1. 遍历参数计算数值梯度
    for p_idx, param in enumerate(model.parameters()):
        if not param.requires_grad: continue

        p_data_cpu = param.data.get() if hasattr(param.data, 'get') else param.data.copy()
        p_grad_ana_cpu = param.grad.get() if hasattr(param.grad, 'get') else param.grad.copy()
        p_grad_num_cpu = np.zeros_like(p_data_cpu)

        it = np.nditer(p_data_cpu, flags=['multi_index'], op_flags=['readwrite'])
        count, max_check = 0, 50
        xp = param.xp

        while not it.finished and count < max_check:
            idx = it.multi_index
            old_val = p_data_cpu[idx]

            p_data_cpu[idx] = old_val + eps
            param.data = xp.asarray(p_data_cpu)
            loss_plus = float(loss_fn(model(inputs), targets).data)

            p_data_cpu[idx] = old_val - eps
            param.data = xp.asarray(p_data_cpu)
            loss_minus = float(loss_fn(model(inputs), targets).data)

            p_grad_num_cpu[idx] = (loss_plus - loss_minus) / (2 * eps)
            p_data_cpu[idx] = old_val
            it.iternext()
            count += 1

        check_mask = p_grad_num_cpu != 0
        if np.any(check_mask):
            all_analytic.extend(p_grad_ana_cpu[check_mask].ravel())
            all_numeric.extend(p_grad_num_cpu[check_mask].ravel())

        print(f"模块参数 [{p_idx}] 检验完成. 形状: {param.shape()}")

    # 2. 结果汇总（移出循环，只执行一次）
    all_a, all_n = np.array(all_analytic), np.array(all_numeric)
    rel_error = np.mean(np.abs(all_a - all_n) / (np.maximum(np.abs(all_a), np.abs(all_n)) + 1e-10))

    print("-" * 50)
    print(f"检验完成: 全模型平均相对误差 = {rel_error:.2e}")

    # 3. 可视化优化
    plt.figure(figsize=(7, 6))  # 稍微调大画布
    plt.scatter(all_a, all_n, alpha=0.6, c='#2c3e50', label='Gradient Pairs')
    lims = [min(all_a.min(), all_n.min()), max(all_a.max(), all_n.max())]
    plt.plot(lims, lims, 'r--', lw=2, label='Perfect Match (y=x)')

    plt.title(f"Real-time Gradient Check (Global)\nRelative Error: {rel_error:.2e}")
    plt.xlabel("Analytical (Mytorch Engine)")
    plt.ylabel("Numerical (Mathematics)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)

    # 解决标签掉到外面去的问题
    plt.tight_layout()
    plt.savefig("check.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"实时对比图已生成: grad_check.png")
    print("=" * 50 + "\n")
def make_dot(root_tensor: Tensor):
    """
    生成计算图的可视化对象
    root_tensor: 通常是 loss 张量
    """
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})  # LR: 从左到右

    visited = set()

    def add_nodes(v):
        if v in visited:
            return
        visited.add(v)

        # 为节点生成唯一 ID
        node_id = str(id(v))

        if isinstance(v, Tensor):
            # Tensor 节点：显示形状，如果是参数则标色
            color = 'lightblue' if v.requires_grad else 'white'
            label = f"Tensor\nshape: {v.shape()}"
            if v.creator is None and v.requires_grad:
                label += "\n(Parameter)"
                color = 'orange'

            dot.node(node_id, label=label, shape='ellipse', style='filled', fillcolor=color)

            # 递归创建创建者的节点（如果有）
            if v.creator is not None:
                add_nodes(v.creator)
                # 边：Op -> Tensor
                dot.edge(str(id(v.creator)), node_id)

        elif isinstance(v, Function):
            # Op 节点：显示操作名称
            op_name = v.__class__.__name__
            dot.node(node_id, label=op_name, shape='box', style='filled', fillcolor='lightgrey')

            # 递归创建所有输入的节点
            for inp in v._get_inputs():
                if inp is not None:
                    add_nodes(inp)
                    # 边：Tensor -> Op
                    dot.edge(str(id(inp)), node_id)

    add_nodes(root_tensor)
    return dot

def compute_numerical_gradient(f, x: Tensor, eps=1e-4):
    """
    通用数值梯度计算函数。
    f: 一个接受 Tensor 并返回标量 Tensor 的函数。
    x: 需要计算梯度的输入 Tensor。
    """
    # 确保在 CPU 上进行高精度计算
    x_data = x.data if isinstance(x.data, np.ndarray) else x.data.get()
    grad = np.zeros_like(x_data)

    it = np.nditer(x_data, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        old_val = x_data[idx]

        # 计算 f(x + eps)
        x_data[idx] = old_val + eps
        x.data = x_data if x.device() == 'cpu' else x.xp.asarray(x_data)
        y1 = f(x).data
        if hasattr(y1, 'get'): y1 = y1.get()  # 处理 cupy

        # 计算 f(x - eps)
        x_data[idx] = old_val - eps
        x.data = x_data if x.device() == 'cpu' else x.xp.asarray(x_data)
        y2 = f(x).data
        if hasattr(y2, 'get'): y2 = y2.get()

        # 有限差分公式: $$g = \frac{f(x+\epsilon) - f(x-\epsilon)}{2\epsilon}$$
        grad[idx] = (y1 - y2) / (2 * eps)

        x_data[idx] = old_val
        it.iternext()

    # 恢复原状
    x.data = x_data if x.device() == 'cpu' else x.xp.asarray(x_data)
    return grad
