from graphviz import Digraph
from .tensor import Tensor
from .function import Function


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