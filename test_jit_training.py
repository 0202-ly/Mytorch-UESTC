import numpy as np

import mytorch.jit as jit
from mytorch.function import MSE
from mytorch.modules import BatchNorm2d, Conv2d, Linear, Module, ReLU
from mytorch.tensor import Tensor


def _cached_backend(compiled):
    return compiled.cache[next(iter(compiled.cache))]["backend"]


def _cached_graph(compiled):
    return compiled.cache[next(iter(compiled.cache))]["optimized_graph"]


class TinyMLP(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(4, 8)
        self.relu = ReLU()
        self.fc2 = Linear(8, 2)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class TinyConvBN(Module):
    def __init__(self):
        super().__init__()
        self.conv = Conv2d(3, 4, 3, stride=1, padding=1)
        self.bn = BatchNorm2d(4)
        self.relu = ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


def test_training_forward_graph_mlp_keeps_autograd_creator():
    assert hasattr(jit, "compile_inference")

    np.random.seed(0)
    model = TinyMLP().train()
    compiled = jit.compile(model)

    x = Tensor(np.random.randn(5, 4).astype(np.float32))
    target = Tensor(np.random.randn(5, 2).astype(np.float32))

    pred = compiled(x)
    assert _cached_backend(compiled) == "training"
    assert "train_plan" in compiled.cache[next(iter(compiled.cache))]
    assert compiled.cache[next(iter(compiled.cache))]["train_plan"]["executors"]
    assert pred.creator is not None

    loss = MSE()(pred, target)
    loss.backward()

    assert model.fc1.weight.grad is not None
    assert model.fc1.bias.grad is not None
    assert model.fc2.weight.grad is not None
    assert model.fc2.bias.grad is not None


def test_training_forward_graph_conv_bn_relu_fuses_and_backpropagates():
    np.random.seed(1)
    model = TinyConvBN().train()
    compiled = jit.compile_train(model)

    x = Tensor(np.random.randn(2, 3, 5, 5).astype(np.float32))
    target = Tensor(np.random.randn(2, 4, 5, 5).astype(np.float32))

    pred = compiled(x)
    graph = _cached_graph(compiled)

    assert _cached_backend(compiled) == "training"
    assert "train_plan" in compiled.cache[next(iter(compiled.cache))]
    assert compiled.cache[next(iter(compiled.cache))]["train_plan"]["executors"]
    assert pred.creator is not None
    assert any(node.op_name == "Conv2dOp" for node in graph.nodes)
    assert any(node.op_name == "FusedBNReLU" for node in graph.nodes)
    assert not any(node.op_name == "FusedConvBNReLU" for node in graph.nodes)

    loss = MSE()(pred, target)
    loss.backward()

    assert model.conv.weight.grad is not None
    assert model.conv.bias.grad is not None
    assert model.bn.weight.grad is not None
    assert model.bn.bias.grad is not None


def test_training_forward_graph_experimental_conv_bn_fusion_is_opt_in():
    np.random.seed(2)
    model = TinyConvBN().train()
    compiled = jit.compile_train(model, experimental_conv_bn_fusion=True)

    x = Tensor(np.random.randn(2, 3, 5, 5).astype(np.float32))
    pred = compiled(x)
    graph = _cached_graph(compiled)

    assert pred.creator is not None
    assert any(node.op_name == "FusedConvBNReLU" for node in graph.nodes)


if __name__ == "__main__":
    test_training_forward_graph_mlp_keeps_autograd_creator()
    test_training_forward_graph_conv_bn_relu_fuses_and_backpropagates()
    test_training_forward_graph_experimental_conv_bn_fusion_is_opt_in()
    print("Training forward graph executor tests passed.")
