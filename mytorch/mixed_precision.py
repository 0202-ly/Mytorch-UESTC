# mytorch/mixed_precision.py

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None

from mytorch.tensor import Tensor
from mytorch.modules import (
    Module,
    Conv2d,
    Linear,
    BatchNorm2d,
    FusedBatchNormReLU,
    FusedBatchNormAddReLU,
    FusedAddReLU,
)

from mytorch.function import Add, ReLU as ReLU_Op

# 如果你的文件名是 constant_folding_dynamic.py，就把下面这一行改成：
# from .constant_folding_dynamic import fold_dynamic_fused_conv_bn_for_inference_
from mytorch.constant_folding import fold_dynamic_fused_conv_bn_for_inference_


class AddReLU(Module):
    """
    dtype 通用版 Add + ReLU。
    用普通 Add 和 ReLU 组合，避免 FusedAddReLUOp 只支持 float32 的问题。
    """

    def forward(self, x1, x2):
        return ReLU_Op()(Add()(x1, x2))


def _is_cupy_array(x):
    return cp is not None and isinstance(x, cp.ndarray)


def _astype_array(data, dtype):
    if _is_cupy_array(data):
        if dtype == np.float16:
            return data.astype(cp.float16)
        if dtype == np.float32:
            return data.astype(cp.float32)
        return data.astype(dtype)

    return np.asarray(data).astype(dtype)


def cast_tensor_(tensor: Tensor, dtype):
    """
    原地转换 Tensor.data dtype。
    推理阶段使用，不保留梯度。
    """
    tensor.data = _astype_array(tensor.data, dtype)

    if tensor.grad is not None:
        tensor.grad = _astype_array(tensor.grad, dtype)

    tensor.requires_grad = False
    return tensor


def tensor_to_dtype(x: Tensor, dtype):
    """
    返回一个新的 Tensor，用于把输入转成 FP16 / FP32。
    """
    return Tensor(
        _astype_array(x.data, dtype),
        device=x.device(),
        requires_grad=False,
    )


def _iter_modules(root):
    """
    递归收集所有 Module，支持 list / tuple / dict。
    """
    modules = []
    seen = set()

    def visit(obj):
        if not isinstance(obj, Module):
            return

        obj_id = id(obj)
        if obj_id in seen:
            return
        seen.add(obj_id)

        modules.append(obj)

        for value in obj.__dict__.values():
            if isinstance(value, Module):
                visit(value)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    visit(item)
            elif isinstance(value, dict):
                for item in value.values():
                    visit(item)

    visit(root)
    return modules


def _replace_fused_add_relu_(module):
    """
    BN 折叠后，FusedBatchNormAddReLU 会变成 FusedAddReLU。
    但你当前 FusedAddReLUOp 的 CUDA ElementwiseKernel 是 float32 版本，
    所以混合精度推理时把它换成普通 Add + ReLU。
    """

    for name in list(module.__dict__.keys()):
        value = getattr(module, name)

        if isinstance(value, FusedAddReLU):
            setattr(module, name, AddReLU())

        elif isinstance(value, Module):
            _replace_fused_add_relu_(value)

        elif isinstance(value, list):
            new_list = []
            for item in value:
                if isinstance(item, FusedAddReLU):
                    new_list.append(AddReLU())
                else:
                    if isinstance(item, Module):
                        _replace_fused_add_relu_(item)
                    new_list.append(item)
            setattr(module, name, new_list)

        elif isinstance(value, tuple):
            new_items = []
            for item in value:
                if isinstance(item, FusedAddReLU):
                    new_items.append(AddReLU())
                else:
                    if isinstance(item, Module):
                        _replace_fused_add_relu_(item)
                    new_items.append(item)
            setattr(module, name, tuple(new_items))

        elif isinstance(value, dict):
            new_dict = {}
            for k, item in value.items():
                if isinstance(item, FusedAddReLU):
                    new_dict[k] = AddReLU()
                else:
                    if isinstance(item, Module):
                        _replace_fused_add_relu_(item)
                    new_dict[k] = item
            setattr(module, name, new_dict)


def prepare_fp16_inference_(
    model: Module,
    fold_bn: bool = True,
    keep_first_conv_fp32: bool = False,
    keep_last_linear_fp32: bool = True,
    verbose: bool = True,
):
    """
    推理阶段 FP16 混合精度准备。

    推荐默认策略：
    - Conv2d 转 FP16
    - 中间 Linear 可转 FP16
    - 最后一层 Linear 默认保留 FP32，减少回归输出精度损失
    - BN 必须先折叠，不建议保留 BN 后直接 FP16
    """

    model.eval()

    if fold_bn:
        folded = fold_dynamic_fused_conv_bn_for_inference_(
            model,
            freeze=True,
            verbose=verbose,
        )
    else:
        folded = 0

    # 替换 residual 里的 FusedAddReLU，避免 float32-only fused kernel 影响半精度
    _replace_fused_add_relu_(model)

    modules = _iter_modules(model)

    convs = [m for m in modules if isinstance(m, Conv2d)]
    linears = [m for m in modules if isinstance(m, Linear)]

    first_conv = convs[0] if convs else None
    last_linear = linears[-1] if linears else None

    converted = 0
    kept_fp32 = 0
    risky_bn_left = 0

    for m in modules:
        if isinstance(m, (BatchNorm2d, FusedBatchNormReLU, FusedBatchNormAddReLU)):
            risky_bn_left += 1
            # 如果还有 BN，保持 FP32，不转半精度
            for name in ("weight", "bias", "running_mean", "running_var"):
                if hasattr(m, name):
                    cast_tensor_(getattr(m, name), np.float32)
            continue

        if isinstance(m, Conv2d):
            if keep_first_conv_fp32 and m is first_conv:
                cast_tensor_(m.weight, np.float32)
                if m.bias is not None:
                    cast_tensor_(m.bias, np.float32)
                kept_fp32 += 1
            else:
                cast_tensor_(m.weight, np.float16)
                if m.bias is not None:
                    cast_tensor_(m.bias, np.float16)
                converted += 1

        elif isinstance(m, Linear):
            if keep_last_linear_fp32 and m is last_linear:
                cast_tensor_(m.weight, np.float32)
                if m.bias is not None:
                    cast_tensor_(m.bias, np.float32)
                kept_fp32 += 1
            else:
                cast_tensor_(m.weight, np.float16)
                if m.bias is not None:
                    cast_tensor_(m.bias, np.float16)
                converted += 1

    if verbose:
        print(
            "[MixedPrecisionInference] "
            f"folded_bn_like={folded}, "
            f"converted_fp16_layers={converted}, "
            f"kept_fp32_layers={kept_fp32}, "
            f"bn_like_left_fp32={risky_bn_left}"
        )

    model.eval()
    return model


def fp16_inference(model: Module, x: Tensor, output_float32: bool = True):
    """
    FP16 推理入口。
    """
    x = tensor_to_dtype(x, np.float16)
    x.requires_grad = False

    y = model(x)

    # 推理输出建议转回 FP32，便于后续指标计算和和 FP32 baseline 比较
    if output_float32:
        y = tensor_to_dtype(y, np.float32)

    y.requires_grad = False
    y.creator = None
    return y