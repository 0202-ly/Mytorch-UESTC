from mytorch.modules import (
    Module,
    Conv2d,
    BatchNorm2d,
    ReLU,
    MaxPool,
    AdaptiveAvgPool2d,
    Linear,
    Flatten,
    FusedBatchNormReLU,
    FusedBatchNormAddReLU,
)


class Downsample(Module):
    """
    残差分支下采样：
        1x1 Conv -> BN

    注意：
        Downsample 分支这里保持普通 BN。
        因为它不是简单的 BN + ReLU 结构，也不是单独的 BN + Add + ReLU。
    """
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            padding=0
        )
        self.bn = BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class BasicBlockOriginal(Module):
    """
    真正原始 BasicBlock：

        conv1 -> BN -> ReLU
        conv2 -> BN
        Add
        ReLU

    不包含任何 fused 算子。
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1
        )
        self.bn1 = BatchNorm2d(out_channels)
        self.relu1 = ReLU()

        self.conv2 = Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn2 = BatchNorm2d(out_channels)
        self.relu2 = ReLU()

        self.downsample = None
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = Downsample(
                in_channels,
                out_channels * self.expansion,
                stride
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu2(out)

        return out


class BasicBlockFullFusion(Module):
    """
    完整训练融合 BasicBlock：

        conv1 -> FusedBatchNormReLU
        conv2 -> FusedBatchNormAddReLU

    其中：
        FusedBatchNormReLU 等价于 BN + ReLU
        FusedBatchNormAddReLU 等价于 BN + residual Add + ReLU
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1
        )
        self.bn1_relu = FusedBatchNormReLU(
            BatchNorm2d(out_channels)
        )

        self.conv2 = Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn2_add_relu = FusedBatchNormAddReLU(
            BatchNorm2d(out_channels)
        )

        self.downsample = None
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = Downsample(
                in_channels,
                out_channels * self.expansion,
                stride
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1_relu(out)

        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.bn2_add_relu(out, identity)

        return out


class ResNet(Module):
    """
    通用 ResNet 组装类。

    block:
        BasicBlockOriginal
        BasicBlockFullFusion

    fused_stem:
        False: stem 使用 BN + ReLU
        True:  stem 使用 FusedBatchNormReLU
    """
    def __init__(
        self,
        block,
        num_blocks,
        num_classes=1000,
        fused_stem=False
    ):
        super().__init__()

        self.in_channels = 64
        self.fused_stem = fused_stem

        self.conv1 = Conv2d(
            3,
            64,
            kernel_size=7,
            stride=2,
            padding=3
        )

        if fused_stem:
            self.bn1_relu = FusedBatchNormReLU(
                BatchNorm2d(64)
            )
            self.bn1 = None
            self.relu = None
        else:
            self.bn1 = BatchNorm2d(64)
            self.relu = ReLU()
            self.bn1_relu = None

        self.maxpool = MaxPool(
            kernel_size=3,
            stride=2,
            padding=1
        )

        self.layer1 = self._make_layer(
            block,
            64,
            num_blocks[0],
            stride=1
        )
        self.layer2 = self._make_layer(
            block,
            128,
            num_blocks[1],
            stride=2
        )
        self.layer3 = self._make_layer(
            block,
            256,
            num_blocks[2],
            stride=2
        )
        self.layer4 = self._make_layer(
            block,
            512,
            num_blocks[3],
            stride=2
        )

        self.avgpool = AdaptiveAvgPool2d((1, 1))
        self.flatten = Flatten()
        self.fc = Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for s in strides:
            layers.append(
                block(
                    self.in_channels,
                    out_channels,
                    s
                )
            )
            self.in_channels = out_channels * block.expansion

        return layers

    def forward(self, x):
        x = self.conv1(x)

        if self.fused_stem:
            x = self.bn1_relu(x)
        else:
            x = self.bn1(x)
            x = self.relu(x)

        x = self.maxpool(x)

        for block in self.layer1:
            x = block(x)
        for block in self.layer2:
            x = block(x)
        for block in self.layer3:
            x = block(x)
        for block in self.layer4:
            x = block(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x


def ResNet18Original(num_classes=1000):
    """
    真正原始 ResNet18：

        stem:   Conv -> BN -> ReLU
        block:  conv1 -> BN -> ReLU
                conv2 -> BN -> Add -> ReLU
    """
    return ResNet(
        BasicBlockOriginal,
        [2, 2, 2, 2],
        num_classes=num_classes,
        fused_stem=False
    )


def ResNet18Fused(num_classes=1000):
    """
    完整融合 ResNet18：

        stem:   Conv -> FusedBatchNormReLU
        block:  conv1 -> FusedBatchNormReLU
                conv2 -> FusedBatchNormAddReLU
    """
    return ResNet(
        BasicBlockFullFusion,
        [2, 2, 2, 2],
        num_classes=num_classes,
        fused_stem=True
    )


def ResNet18(num_classes=1000, fused=False):
    """
    兼容接口。

    fused=False:
        返回 ResNet18Original

    fused=True:
        返回 ResNet18Fused
    """
    if fused:
        return ResNet18Fused(num_classes=num_classes)

    return ResNet18Original(num_classes=num_classes)


# 兼容旧代码：
# 如果旧代码写 ResNet(BasicBlock, ...)，默认 BasicBlock 表示原始块。
BasicBlock = BasicBlockOriginal