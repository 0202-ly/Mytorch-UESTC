# mytorch/jit_train_rewrite.py
# Staticized dynamic fusion:
#   dynamic detection + static-style execution.
#
# Goal:
#   Make dynamic fusion closer to the original hand-written ResNet18Fused performance.
#
# Key idea:
#   Do NOT patch BasicBlockOriginal.forward with runtime branches.
#   Replace each BasicBlockOriginal object by a clean DynamicFusedBasicBlock object.
#
# Fusion patterns:
#   1) Stem: bn1 + relu -> FusedBatchNormReLU + Identity
#   2) Block first branch: bn1 + relu1 -> FusedBatchNormReLU
#   3) Block residual branch: bn2 + Add(identity) + relu2 -> FusedBatchNormAddReLU

from .modules import (
    Module,
    BatchNorm2d,
    ReLU,
    Identity,
    FusedBatchNormReLU,
    FusedBatchNormAddReLU,
)


def _is_basicblock_like(module):
    """
    Detect ResNet BasicBlock-like modules by structure rather than class name.

    This catches BasicBlockOriginal without hardcoding the class name.
    """
    required = ("conv1", "bn1", "relu1", "conv2", "bn2", "relu2")
    return all(hasattr(module, name) for name in required)


def _iter_child_modules(module):
    for value in module.__dict__.values():
        if isinstance(value, Module):
            yield value
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, Module):
                    yield item
        elif isinstance(value, dict):
            for item in value.values():
                if isinstance(item, Module):
                    yield item


class DynamicFusedBasicBlock(Module):
    """
    A clean static-style fused block generated dynamically from BasicBlockOriginal.

    It reuses the original modules/parameters:
        conv1      <- old_block.conv1
        bn1_relu1  <- FusedBatchNormReLU(old_block.bn1)
        conv2      <- old_block.conv2
        bn2_add_relu <- FusedBatchNormAddReLU(old_block.bn2)
        downsample <- old_block.downsample

    Forward path is intentionally clean:
        identity = downsample(x) if downsample else x
        out = conv1(x)
        out = bn1_relu1(out)
        out = conv2(out)
        out = bn2_add_relu(out, identity)
        return out

    No Identity call, no runtime branch, no patched MethodType forward.
    """
    def __init__(self, old_block):
        super().__init__()

        if not _is_basicblock_like(old_block):
            raise TypeError(
                "DynamicFusedBasicBlock expects a BasicBlock-like module with "
                "conv1, bn1, relu1, conv2, bn2, relu2."
            )

        if not isinstance(old_block.bn1, BatchNorm2d):
            raise TypeError(f"old_block.bn1 must be BatchNorm2d, got {type(old_block.bn1)}")

        if not isinstance(old_block.bn2, BatchNorm2d):
            raise TypeError(f"old_block.bn2 must be BatchNorm2d, got {type(old_block.bn2)}")

        self.conv1 = old_block.conv1
        self.bn1_relu1 = FusedBatchNormReLU(old_block.bn1)

        self.conv2 = old_block.conv2
        self.bn2_add_relu = FusedBatchNormAddReLU(old_block.bn2)

        self.downsample = getattr(old_block, "downsample", None)

        # Useful metadata for debugging/reporting.
        self.original_block_class = old_block.__class__.__name__

    def forward(self, x):
        identity = self.downsample(x) if self.downsample is not None else x

        out = self.conv1(x)
        out = self.bn1_relu1(out)

        out = self.conv2(out)
        out = self.bn2_add_relu(out, identity)

        return out


def _fuse_stem_inplace(model, verbose=True):
    """
    Fuse ResNet stem:
        model.bn1 + model.relu -> FusedBatchNormReLU + Identity

    We keep model.relu = Identity because the original ResNet.forward probably still
    calls self.relu(x). This is one remaining Python no-op call, but it is only once
    per forward, so the cost is negligible compared with 8 BasicBlock forwards.
    """
    if not (hasattr(model, "bn1") and hasattr(model, "relu")):
        return 0

    bn = getattr(model, "bn1")
    relu = getattr(model, "relu")

    if not isinstance(bn, BatchNorm2d):
        return 0

    if not isinstance(relu, ReLU):
        return 0

    model.bn1 = FusedBatchNormReLU(bn)
    model.relu = Identity()

    if verbose:
        print("[TrainGraphRewrite] fused ResNet stem bn1 + relu -> FusedBatchNormReLU + Identity")

    return 1


def _replace_blocks_in_value(value, verbose=True):
    """
    Replace BasicBlock-like modules inside a value.

    Returns:
        new_value, replace_count
    """
    if isinstance(value, Module):
        if _is_basicblock_like(value):
            new_block = DynamicFusedBasicBlock(value)
            if verbose:
                print(
                    f"[TrainGraphRewrite] replaced {value.__class__.__name__} "
                    f"-> DynamicFusedBasicBlock"
                )
            return new_block, 1

        # Recurse into non-block module attributes.
        count = _replace_basicblocks_in_module(value, verbose=verbose)
        return value, count

    if isinstance(value, list):
        new_list = []
        count = 0
        for item in value:
            new_item, c = _replace_blocks_in_value(item, verbose=verbose)
            new_list.append(new_item)
            count += c
        return new_list, count

    if isinstance(value, tuple):
        new_items = []
        count = 0
        for item in value:
            new_item, c = _replace_blocks_in_value(item, verbose=verbose)
            new_items.append(new_item)
            count += c
        return tuple(new_items), count

    if isinstance(value, dict):
        new_dict = {}
        count = 0
        for k, item in value.items():
            new_item, c = _replace_blocks_in_value(item, verbose=verbose)
            new_dict[k] = new_item
            count += c
        return new_dict, count

    return value, 0


def _replace_basicblocks_in_module(module, verbose=True):
    """
    In-place replace BasicBlockOriginal-like children with DynamicFusedBasicBlock.

    This works for ResNet structures where layers are stored as:
        self.layer1 = [block1, block2]
        self.layer2 = [...]
    or direct module attributes.
    """
    total = 0

    # Snapshot keys to avoid dict-size mutation issues.
    for name in list(module.__dict__.keys()):
        value = getattr(module, name)

        # Do not replace the root module itself here; only its children.
        new_value, count = _replace_blocks_in_value(value, verbose=verbose)

        if count > 0 and new_value is not value:
            setattr(module, name, new_value)

        total += count

    return total


def fuse_bn_relu_for_training(model, verbose=True):
    """
    Public API used by train_donkeycar.py.

    Dynamic fusion strategy:
        1. Detect and fuse ResNet stem.
        2. Replace BasicBlockOriginal-like modules with clean DynamicFusedBasicBlock.

    Expected for ResNet18:
        stem_fused = 1
        block_replaced = 8

    Expected module count after rewrite:
        DynamicFusedBasicBlock: 8
        FusedBatchNormReLU:    9
        FusedBatchNormAddReLU: 8
        Identity:              1   # only stem relu placeholder
        BatchNorm2d:           3   # downsample BNs
        ReLU:                  0 or very low
    """
    stem_fused = _fuse_stem_inplace(model, verbose=verbose)
    block_replaced = _replace_basicblocks_in_module(model, verbose=verbose)

    if verbose:
        print(
            "[TrainGraphRewrite] staticized dynamic fusion summary: "
            f"stem_fused={stem_fused}, "
            f"block_replaced={block_replaced}, "
            f"expected_fused_structures={stem_fused + block_replaced * 2}"
        )

    return model
