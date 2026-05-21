# mytorch/constant_folding_dynamic.py
# Dynamic-fused-model-aware constant folding.
#
# Supports:
#   1) Conv2d + BatchNorm2d
#        -> Conv2d + Identity
#
#   2) Conv2d + FusedBatchNormReLU
#        -> Folded Conv2d + ReLU
#
#   3) Conv2d + FusedBatchNormAddReLU
#        -> Folded Conv2d + FusedAddReLU
#
# This is intended for inference/eval after training.
# Do NOT use it during training.

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None

from .tensor import Tensor
from .modules import (
    Module,
    Conv2d,
    BatchNorm2d,
    ReLU,
    Identity,
    FusedBatchNormReLU,
    FusedBatchNormAddReLU,
    FusedAddReLU,
)


def _is_bn_like(module):
    """
    BN-like modules whose affine/statistics can be folded into the preceding Conv2d.

    Supported:
        BatchNorm2d
        FusedBatchNormReLU
        FusedBatchNormAddReLU
    """
    return isinstance(module, (BatchNorm2d, FusedBatchNormReLU, FusedBatchNormAddReLU))


def _bn_like_kind(module):
    if isinstance(module, BatchNorm2d):
        return "BatchNorm2d"
    if isinstance(module, FusedBatchNormReLU):
        return "FusedBatchNormReLU"
    if isinstance(module, FusedBatchNormAddReLU):
        return "FusedBatchNormAddReLU"
    return module.__class__.__name__


def _to_float32_array(x, xp):
    if xp is np:
        return np.asarray(x, dtype=np.float32)
    return xp.asarray(x, dtype=xp.float32)


def _get_num_features(bn_like):
    if hasattr(bn_like, "num_features"):
        return int(bn_like.num_features)
    # fallback from weight shape: (1, C, 1, 1)
    return int(bn_like.weight.shape()[1])


def fold_conv_bn_like_pair(conv: Conv2d, bn_like: Module, freeze: bool = True):
    """
    Fold BN-like affine/statistics into Conv2d.

    For inference:
        BN(y) = gamma * (y - running_mean) / sqrt(running_var + eps) + beta

    If y = Conv(x, W, b), then:
        scale  = gamma / sqrt(running_var + eps)
        W_fold = W * scale
        b_fold = beta + (b - running_mean) * scale

    This function only updates conv.weight / conv.bias.
    The caller decides how to replace bn_like:
        BatchNorm2d              -> Identity
        FusedBatchNormReLU       -> ReLU
        FusedBatchNormAddReLU    -> FusedAddReLU
    """
    if not isinstance(conv, Conv2d):
        raise TypeError(f"conv must be Conv2d, got {type(conv)}")

    if not _is_bn_like(bn_like):
        raise TypeError(
            "bn_like must be BatchNorm2d/FusedBatchNormReLU/FusedBatchNormAddReLU, "
            f"got {type(bn_like)}"
        )

    num_features = _get_num_features(bn_like)
    if conv.out_channels != num_features:
        raise ValueError(
            f"Conv2d.out_channels must equal BN-like num_features, "
            f"got conv.out_channels={conv.out_channels}, num_features={num_features}"
        )

    xp = conv.weight.xp
    device = conv.weight.device()

    W = _to_float32_array(conv.weight.data, xp)

    if conv.bias is None:
        b = xp.zeros((conv.out_channels,), dtype=xp.float32)
    else:
        b = _to_float32_array(conv.bias.data, xp).reshape(-1)

    gamma = _to_float32_array(bn_like.weight.data, xp).reshape(-1)
    beta = _to_float32_array(bn_like.bias.data, xp).reshape(-1)
    running_mean = _to_float32_array(bn_like.running_mean.data, xp).reshape(-1)
    running_var = _to_float32_array(bn_like.running_var.data, xp).reshape(-1)

    eps = float(bn_like.eps)

    scale = gamma / xp.sqrt(running_var + eps)

    W_fold = W * scale.reshape((-1, 1, 1, 1))
    b_fold = beta + (b - running_mean) * scale

    conv.weight = Tensor(
        W_fold.astype(xp.float32),
        device=device,
        requires_grad=(False if freeze else conv.weight.requires_grad),
    )

    conv.bias = Tensor(
        b_fold.reshape(1, -1).astype(xp.float32),
        device=device,
        requires_grad=(False if freeze else (conv.bias.requires_grad if conv.bias is not None else False)),
    )

    return conv


def _replacement_for_bn_like(bn_like):
    """
    After folding the BN-like part into the preceding Conv2d, replace the module
    with the remaining operation.
    """
    if isinstance(bn_like, BatchNorm2d):
        # Conv + BN -> Conv
        return Identity()

    if isinstance(bn_like, FusedBatchNormReLU):
        # Conv + Fused(BN+ReLU) -> FoldedConv + ReLU
        return ReLU()

    if isinstance(bn_like, FusedBatchNormAddReLU):
        # Conv + Fused(BN+Add+ReLU) -> FoldedConv + Fused(Add+ReLU)
        # The dynamic BasicBlock forward usually calls this module as:
        #     self.bn2_add_relu(out, identity)
        # FusedAddReLU has the same two-argument signature.
        return FusedAddReLU()

    raise TypeError(f"Unsupported bn_like: {type(bn_like)}")

def _candidate_bn_names_for_conv(conv_name):
    """
    Candidate attribute names paired with a Conv2d attribute.

    Supports:
        conv1 -> bn1
        conv1 -> bn1_relu1       # DynamicFusedBasicBlock
        conv2 -> bn2
        conv2 -> bn2_add_relu    # residual fused branch
        conv  -> bn
    """
    names = []

    if conv_name.startswith("conv"):
        suffix = conv_name[len("conv"):]

        if suffix:
            # DynamicFusedBasicBlock:
            #   conv1 -> bn1_relu1
            #   conv2 -> bn2_add_relu
            if suffix == "1":
                names.append("bn1_relu1")

            if suffix == "2":
                names.append("bn2_add_relu")

            # Original BasicBlock:
            #   conv1 -> bn1
            #   conv2 -> bn2
            names.append("bn" + suffix)

    if conv_name == "conv":
        names.append("bn")

    return names


def _fold_direct_attrs(module: Module, freeze=True, verbose=True):
    folded = 0

    for conv_name, conv in list(module.__dict__.items()):
        if not isinstance(conv, Conv2d):
            continue

        for bn_name in _candidate_bn_names_for_conv(conv_name):
            if not hasattr(module, bn_name):
                continue

            bn_like = getattr(module, bn_name)

            if not _is_bn_like(bn_like):
                continue

            kind = _bn_like_kind(bn_like)

            fold_conv_bn_like_pair(conv, bn_like, freeze=freeze)
            setattr(module, bn_name, _replacement_for_bn_like(bn_like))

            folded += 1

            if verbose:
                print(
                    f"[DynamicConstantFolding] folded "
                    f"{module.__class__.__name__}.{conv_name} + {bn_name}({kind}) "
                    f"-> folded {conv_name} + {getattr(module, bn_name).__class__.__name__}"
                )

            break

    return folded


def _fold_list_container(container, freeze=True, verbose=True):
    """
    Fold adjacent Conv2d + BatchNorm-like modules in list containers.

    This handles simple sequential containers such as:
        downsample = [Conv2d(...), BatchNorm2d(...)]
    """
    if not isinstance(container, list):
        return 0

    folded = 0
    i = 0

    while i < len(container) - 1:
        conv = container[i]
        bn_like = container[i + 1]

        if isinstance(conv, Conv2d) and _is_bn_like(bn_like):
            # FusedBatchNormAddReLU is not valid in a plain sequential list
            # because it requires identity as second input. Skip if encountered.
            if isinstance(bn_like, FusedBatchNormAddReLU):
                i += 1
                continue

            kind = _bn_like_kind(bn_like)

            fold_conv_bn_like_pair(conv, bn_like, freeze=freeze)
            container[i + 1] = _replacement_for_bn_like(bn_like)

            folded += 1

            if verbose:
                print(
                    f"[DynamicConstantFolding] folded list[{i}] Conv2d + "
                    f"list[{i + 1}]({kind}) -> Conv2d + {container[i + 1].__class__.__name__}"
                )

            i += 2
        else:
            i += 1

    return folded


def fold_dynamic_fused_conv_bn_for_inference_(model: Module, freeze=True, verbose=True):
    """
    In-place constant folding for dynamic-fused models.

    It supports both original and dynamic-fused structures:
        Conv2d + BatchNorm2d
        Conv2d + FusedBatchNormReLU
        Conv2d + FusedBatchNormAddReLU

    Requirements:
        1. Call after training.
        2. Call in eval/inference stage.
        3. Do not continue training after folding.
    """
    if not isinstance(model, Module):
        raise TypeError(f"model must be Module, got {type(model)}")

    model.eval()

    folded_count = 0
    visited = set()

    def visit(obj):
        nonlocal folded_count

        if not isinstance(obj, Module):
            return

        obj_id = id(obj)
        if obj_id in visited:
            return

        visited.add(obj_id)

        folded_count += _fold_direct_attrs(
            obj,
            freeze=freeze,
            verbose=verbose,
        )

        for value in list(obj.__dict__.values()):
            if isinstance(value, Module):
                visit(value)

            elif isinstance(value, list):
                folded_count += _fold_list_container(
                    value,
                    freeze=freeze,
                    verbose=verbose,
                )

                for item in value:
                    if isinstance(item, Module):
                        visit(item)

            elif isinstance(value, tuple):
                for item in value:
                    if isinstance(item, Module):
                        visit(item)

            elif isinstance(value, dict):
                for item in value.values():
                    if isinstance(item, Module):
                        visit(item)

    visit(model)

    if verbose:
        print(f"[DynamicConstantFolding] total folded Conv-BN-like pairs: {folded_count}")

    return folded_count


def fold_conv_bn_for_inference_(model: Module, freeze=True, verbose=True):
    """Backward-compatible alias for inference-time Conv/BN constant folding."""
    return fold_dynamic_fused_conv_bn_for_inference_(
        model,
        freeze=freeze,
        verbose=verbose,
    )


def max_abs_diff(a, b):
    """Return max absolute difference between two Tensor/array values."""
    a_data = a.data if isinstance(a, Tensor) else a
    b_data = b.data if isinstance(b, Tensor) else b

    xp = cp if cp is not None and hasattr(a_data, "get") else np
    diff = xp.max(xp.abs(a_data - b_data))
    if hasattr(diff, "get"):
        diff = diff.get()
    return float(diff)
