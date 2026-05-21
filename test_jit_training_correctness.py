import numpy as np

import mytorch.jit as jit
from mytorch.function import Add, MSE
from mytorch.modules import BatchNorm2d, Conv2d, Linear, Module, ReLU
from mytorch.optim import SGD
from mytorch.tensor import Tensor
from model.resnet import BasicBlockOriginal


try:
    import cupy as cp
except ImportError:
    cp = None


FORWARD_TOL = 1e-5
LOSS_TOL = 1e-6
GRAD_TOL = 1e-5
PARAM_TOL = 1e-5
BN_TOL = 1e-6


def to_numpy(value):
    if isinstance(value, Tensor):
        value = value.data
    if cp is not None and isinstance(value, cp.ndarray):
        value = value.get()
    return np.asarray(value)


def max_abs_diff(a, b):
    a_np = to_numpy(a)
    b_np = to_numpy(b)
    if a_np.shape != b_np.shape:
        raise AssertionError(f"shape mismatch: {a_np.shape} vs {b_np.shape}")
    if a_np.size == 0:
        return 0.0
    return float(np.max(np.abs(a_np - b_np)))


def assert_close(name, a, b, tol):
    diff = max_abs_diff(a, b)
    assert diff <= tol, f"{name} max_abs_diff={diff} > tol={tol}"
    return diff


def named_tensors(module):
    result = []
    seen = set()

    def visit(obj, prefix):
        if isinstance(obj, Tensor):
            if id(obj) not in seen:
                seen.add(id(obj))
                result.append((prefix, obj))
            return

        if isinstance(obj, Module):
            items = obj.__dict__.items()
        elif isinstance(obj, dict):
            items = obj.items()
        elif isinstance(obj, (list, tuple)):
            items = enumerate(obj)
        else:
            return

        for key, value in items:
            if key == "training":
                continue
            name = f"{prefix}.{key}" if prefix else str(key)
            visit(value, name)

    visit(module, "")
    return result


def named_parameters(module):
    return [(name, tensor) for name, tensor in named_tensors(module) if tensor.requires_grad]


def named_bn_buffers(module):
    return [
        (name, tensor)
        for name, tensor in named_tensors(module)
        if name.endswith("running_mean") or name.endswith("running_var")
    ]


def copy_model_tensors(dst, src):
    src_tensors = dict(named_tensors(src))
    dst_tensors = dict(named_tensors(dst))
    assert src_tensors.keys() == dst_tensors.keys()

    for name, src_tensor in src_tensors.items():
        dst_tensor = dst_tensors[name]
        dst_tensor.data[...] = src_tensor.data.copy()
        dst_tensor.grad = None


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


class TinyResidualBlock(Module):
    def __init__(self):
        super().__init__()
        self.conv = Conv2d(3, 3, 3, stride=1, padding=1)
        self.bn = BatchNorm2d(3)
        self.relu = ReLU()

    def forward(self, x):
        identity = x
        out = self.conv(x)
        out = self.bn(out)
        out = Add()(out, identity)
        return self.relu(out)


def run_one_step_case(
    name,
    model_factory,
    x_shape,
    target_shape,
    expected_fused_op=None,
    expected_fused_ops=None,
    x_requires_grad=False,
    compile_kwargs=None,
):
    np.random.seed(2026)
    eager_model = model_factory().train()
    jit_model = model_factory().train()
    copy_model_tensors(jit_model, eager_model)

    x_np = np.random.randn(*x_shape).astype(np.float32)
    target_np = np.random.randn(*target_shape).astype(np.float32)

    x_eager = Tensor(x_np.copy(), requires_grad=x_requires_grad)
    x_jit = Tensor(x_np.copy(), requires_grad=x_requires_grad)
    target_eager = Tensor(target_np.copy(), requires_grad=False)
    target_jit = Tensor(target_np.copy(), requires_grad=False)

    compile_kwargs = {} if compile_kwargs is None else dict(compile_kwargs)
    compiled = jit.compile(jit_model, training=True, **compile_kwargs)

    pred_eager = eager_model(x_eager)
    pred_jit = compiled(x_jit)

    cache = compiled.cache[next(iter(compiled.cache))]
    assert cache["backend"] == "training"
    assert pred_jit.creator is not None
    if expected_fused_op is not None:
        expected_fused_ops = [expected_fused_op]
    if expected_fused_ops is not None:
        graph_ops = [node.op_name for node in cache["optimized_graph"].nodes]
        for op_name in expected_fused_ops:
            assert op_name in graph_ops, f"{name}: missing {op_name}, ops={graph_ops}"

    diffs = {
        "forward": assert_close(f"{name}.forward", pred_eager, pred_jit, FORWARD_TOL),
    }

    loss_eager = MSE()(pred_eager, target_eager)
    loss_jit = MSE()(pred_jit, target_jit)
    diffs["loss"] = assert_close(f"{name}.loss", loss_eager, loss_jit, LOSS_TOL)

    loss_eager.backward()
    loss_jit.backward()

    if x_requires_grad:
        assert x_eager.grad is not None, f"{name}.input: eager grad is None"
        assert x_jit.grad is not None, f"{name}.input: jit grad is None"
        diffs["input_grad"] = assert_close(f"{name}.input_grad", x_eager.grad, x_jit.grad, GRAD_TOL)

    eager_params = named_parameters(eager_model)
    jit_params = named_parameters(jit_model)
    assert [n for n, _ in eager_params] == [n for n, _ in jit_params]

    max_grad_diff = 0.0
    for (param_name, eager_param), (_, jit_param) in zip(eager_params, jit_params):
        assert eager_param.grad is not None, f"{name}.{param_name}: eager grad is None"
        assert jit_param.grad is not None, f"{name}.{param_name}: jit grad is None"
        max_grad_diff = max(
            max_grad_diff,
            assert_close(f"{name}.grad.{param_name}", eager_param.grad, jit_param.grad, GRAD_TOL),
        )
    diffs["max_param_grad"] = max_grad_diff

    eager_opt = SGD(eager_model.parameters(), lr=0.01)
    jit_opt = SGD(compiled.parameters(), lr=0.01)
    eager_opt.step()
    jit_opt.step()

    max_param_diff = 0.0
    for (param_name, eager_param), (_, jit_param) in zip(eager_params, jit_params):
        max_param_diff = max(
            max_param_diff,
            assert_close(f"{name}.param_after_step.{param_name}", eager_param, jit_param, PARAM_TOL),
        )
    diffs["max_param_after_step"] = max_param_diff

    eager_bn = named_bn_buffers(eager_model)
    jit_bn = named_bn_buffers(jit_model)
    assert [n for n, _ in eager_bn] == [n for n, _ in jit_bn]

    max_bn_diff = 0.0
    for (buffer_name, eager_buffer), (_, jit_buffer) in zip(eager_bn, jit_bn):
        max_bn_diff = max(
            max_bn_diff,
            assert_close(f"{name}.bn_buffer.{buffer_name}", eager_buffer, jit_buffer, BN_TOL),
        )
    diffs["max_bn_buffer"] = max_bn_diff

    print(f"{name}: {diffs}")


def test_tiny_mlp_eager_vs_jit_training_step():
    run_one_step_case(
        "TinyMLP",
        TinyMLP,
        x_shape=(5, 4),
        target_shape=(5, 2),
        expected_fused_op="FusedLinearReLU",
    )


def test_tiny_conv_bn_eager_vs_jit_training_step():
    run_one_step_case(
        "TinyConvBN",
        TinyConvBN,
        x_shape=(2, 3, 5, 5),
        target_shape=(2, 4, 5, 5),
        expected_fused_ops=["Conv2dOp", "FusedBNReLU"],
    )


def test_tiny_residual_block_eager_vs_jit_training_step():
    run_one_step_case(
        "TinyResidualBlock",
        TinyResidualBlock,
        x_shape=(2, 3, 5, 5),
        target_shape=(2, 3, 5, 5),
        expected_fused_ops=["Conv2dOp", "FusedBNAddReLU"],
    )


def test_tiny_residual_block_experimental_conv_bn_jit_training_step():
    run_one_step_case(
        "TinyResidualBlockExperimentalConvBN",
        TinyResidualBlock,
        x_shape=(2, 3, 5, 5),
        target_shape=(2, 3, 5, 5),
        expected_fused_op="FusedConvBNAddReLU",
        compile_kwargs={"experimental_conv_bn_fusion": True},
    )


def test_cache_hit_keeps_autograd_and_grad_correctness():
    np.random.seed(2027)
    eager_model = TinyConvBN().train()
    jit_model = TinyConvBN().train()
    copy_model_tensors(jit_model, eager_model)
    compiled = jit.compile(jit_model, training=True)

    def run_pair(step, x_np, target_np):
        x_eager = Tensor(x_np.copy(), requires_grad=False)
        x_jit = Tensor(x_np.copy(), requires_grad=False)
        target_eager = Tensor(target_np.copy(), requires_grad=False)
        target_jit = Tensor(target_np.copy(), requires_grad=False)

        pred_eager = eager_model(x_eager)
        pred_jit = compiled(x_jit)
        assert pred_jit.creator is not None, f"cache step {step}: pred creator is None"

        loss_eager = MSE()(pred_eager, target_eager)
        loss_jit = MSE()(pred_jit, target_jit)
        assert_close(f"cache_hit.step{step}.forward", pred_eager, pred_jit, FORWARD_TOL)
        assert_close(f"cache_hit.step{step}.loss", loss_eager, loss_jit, LOSS_TOL)

        loss_eager.backward()
        loss_jit.backward()

        eager_params = named_parameters(eager_model)
        jit_params = named_parameters(jit_model)
        for (param_name, eager_param), (_, jit_param) in zip(eager_params, jit_params):
            assert_close(f"cache_hit.step{step}.grad.{param_name}", eager_param.grad, jit_param.grad, GRAD_TOL)

    x1 = np.random.randn(2, 3, 5, 5).astype(np.float32)
    y1 = np.random.randn(2, 4, 5, 5).astype(np.float32)
    run_pair(1, x1, y1)
    cache_size = len(compiled.cache)

    eager_model.zero_grad()
    jit_model.zero_grad()

    x2 = np.random.randn(2, 3, 5, 5).astype(np.float32)
    y2 = np.random.randn(2, 4, 5, 5).astype(np.float32)
    run_pair(2, x2, y2)
    assert len(compiled.cache) == cache_size


def test_inference_backend_rejects_training_model():
    x = Tensor(np.random.randn(2, 3, 5, 5).astype(np.float32), requires_grad=False)

    model = TinyConvBN().train()
    compiled = jit.compile_inference(model)
    try:
        compiled(x)
    except RuntimeError as exc:
        assert "without autograd creators" in str(exc)
    else:
        raise AssertionError("compile_inference(model.train()) should reject training mode")

    model = TinyConvBN().train()
    compiled = jit.compile(model, training=False)
    try:
        compiled(x)
    except RuntimeError as exc:
        assert "without autograd creators" in str(exc)
    else:
        raise AssertionError("compile(model, training=False) should reject training mode")


def test_resnet_basicblock_eager_vs_jit_training_step():
    run_one_step_case(
        "ResNetBasicBlock",
        lambda: BasicBlockOriginal(3, 3, stride=1),
        x_shape=(2, 3, 5, 5),
        target_shape=(2, 3, 5, 5),
        expected_fused_ops=["Conv2dOp", "FusedBNReLU", "FusedBNAddReLU"],
        x_requires_grad=True,
    )


def test_resnet_basicblock_experimental_conv_bn_jit_training_step():
    run_one_step_case(
        "ResNetBasicBlockExperimentalConvBN",
        lambda: BasicBlockOriginal(3, 3, stride=1),
        x_shape=(2, 3, 5, 5),
        target_shape=(2, 3, 5, 5),
        expected_fused_ops=["FusedConvBNReLU", "FusedConvBNAddReLU"],
        x_requires_grad=True,
        compile_kwargs={"experimental_conv_bn_fusion": True},
    )


if __name__ == "__main__":
    assert hasattr(jit, "compile_inference")
    test_tiny_mlp_eager_vs_jit_training_step()
    test_tiny_conv_bn_eager_vs_jit_training_step()
    test_tiny_residual_block_eager_vs_jit_training_step()
    test_tiny_residual_block_experimental_conv_bn_jit_training_step()
    test_cache_hit_keeps_autograd_and_grad_correctness()
    test_inference_backend_rejects_training_model()
    test_resnet_basicblock_eager_vs_jit_training_step()
    test_resnet_basicblock_experimental_conv_bn_jit_training_step()
    print("Eager vs training forward graph executor correctness tests passed.")
