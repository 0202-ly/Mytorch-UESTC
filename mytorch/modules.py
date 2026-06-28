import numpy as np
import pickle
import json
import os
from datetime import datetime
from .tensor import Tensor
# 引用我们在 function.py 里定义的算子 (确保你的 function.py 中有这些算子)
from .function import (
    Add, MatMul, Conv2dOp, ConvTranspose2dOp, MaxPoolOp, MinPoolOp, AvgPoolOp, ReshapeOp,
    ReLU as ReLU_Op, ELU as ELU_Op, Sigmoid as Sigmoid_Op,
    LogSoftmaxOp, NLLLossOp, BatchNorm2dOp, 
    FusedMSELossOp, FusedBatchNormReLUOp, FusedAddReLUOp, FusedCrossEntropyLossOp, FusedLinearReLUOp,FusedConv2dReLUOp,FusedConvBNReLUOp,
    FusedConvBNAddReLUOp,FusedBatchNormAddReLUOp
)

def _to_2tuple(value):
    """辅助函数：将标量转换为元组，例如 3 -> (3, 3)"""
    if isinstance(value, tuple):
        return value
    return value, value

# ==========================================================
# 核心基类 (终极升级版) - 添加 JSON+NPZ 保存功能
# ==========================================================

class Module:
    def __init__(self):
        self.training = True
        
    def _apply(self, fn):
        """
        内部辅助函数：递归地对所有子模块和张量应用某个函数 (如 .cuda())
        这是处理 ResNet 这种包含 list/dict 结构的网络的关键！
        """
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, Module):
                attr_value._apply(fn)
            elif isinstance(attr_value, Tensor):
                fn(attr_value)
            elif isinstance(attr_value, list):
                for item in attr_value:
                    if isinstance(item, Module):
                        item._apply(fn)
                    elif isinstance(item, Tensor):
                        fn(item)
            elif isinstance(attr_value, dict):
                for item in attr_value.values():
                    if isinstance(item, Module):
                        item._apply(fn)
                    elif isinstance(item, Tensor):
                        fn(item)
        return self
    
    def cuda(self):
        """将模型所有参数转移到 GPU"""
        return self._apply(lambda t: t.cuda())

    def cpu(self):
        """将模型所有参数转移到 CPU"""
        return self._apply(lambda t: t.cpu())

    def train(self):
        """切换到训练模式"""
        self.training = True
        # 递归处理 list 里的子模块
        self._apply_mode(True)
        return self

    def eval(self):
        """切换到评估模式"""
        self.training = False
        self._apply_mode(False)
        return self

    def _apply_mode(self, training):
        """递归切换训练/评估状态的辅助函数"""
        for attr in self.__dict__.values():
            if isinstance(attr, Module):
                attr.training = training
                attr._apply_mode(training)

            elif isinstance(attr, (list, tuple)):
                for item in attr:
                    if isinstance(item, Module):
                        item.training = training
                        item._apply_mode(training)

            elif isinstance(attr, dict):
                for item in attr.values():
                    if isinstance(item, Module):
                        item.training = training
                        item._apply_mode(training)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def zero_grad(self):
        """将模型所有参数的梯度置零"""
        for p in self.parameters():
            p.zero_grad()

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def parameters(self):
        """
        升级版：递归收集所有参数。
        支持识别存储在 list, tuple 或 dict 中的 Tensor 和 Module。
        使用 seen 集合防止参数被重复收集。
        """
        params = []
        seen = set()

        def collect(obj):
            if isinstance(obj, Module):
                # 遍历模块的所有属性
                for v in obj.__dict__.values():
                    collect(v)
            elif isinstance(obj, Tensor):
                # 只收集需要求导的 Tensor
                if obj.requires_grad:
                    obj_id = id(obj)
                    if obj_id not in seen:
                        seen.add(obj_id)
                        params.append(obj)
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    collect(item)
            elif isinstance(obj, dict):
                for item in obj.values():
                    collect(item)

        collect(self)
        return params

    # ==========================================================
    # 新增：named_tensors - 递归遍历所有 Tensor（包括参数和buffer）
    # ==========================================================
    
    def named_tensors(self, include_buffers=True, prefix=''):
        """
        递归遍历模块中的所有 Tensor（参数和buffer）
        
        Args:
            include_buffers: 是否包含非训练参数（如 running_mean/running_var）
            prefix: 名称前缀，用于递归拼接
        
        Returns:
            List of (name, tensor) 元组
        """
        tensors = []
        seen = set()  # 防止同一个Tensor被重复收集（如融合模块中复用的情况）
        
        def collect(obj, name_prefix):
            if isinstance(obj, Module):
                # 遍历模块的所有属性
                for attr_name, attr_value in obj.__dict__.items():
                    # 跳过私有属性
                    if attr_name.startswith('_'):
                        continue
                    collect(attr_value, f"{name_prefix}.{attr_name}" if name_prefix else attr_name)
                    
            elif isinstance(obj, Tensor):
                # 检查是否需要包含
                if obj.requires_grad or include_buffers:
                    obj_id = id(obj)
                    if obj_id not in seen:
                        seen.add(obj_id)
                        tensors.append((name_prefix, obj))
                        
            elif isinstance(obj, (list, tuple)):
                for idx, item in enumerate(obj):
                    collect(item, f"{name_prefix}[{idx}]")
                    
            elif isinstance(obj, dict):
                for key, item in obj.items():
                    collect(item, f"{name_prefix}.{key}")
        
        collect(self, prefix)
        return tensors

    # ==========================================================
    # 新增：state_dict - 返回 {name: numpy_array} 字典
    # ==========================================================
    
    def state_dict(self, include_buffers=True):
        """
        返回模型的完整状态字典
        
        Args:
            include_buffers: 是否包含 buffer（如 BN 的 running_mean/running_var）
        
        Returns:
            dict: {参数名: numpy数组}
        """
        state = {}
        
        for name, tensor in self.named_tensors(include_buffers=include_buffers):
            # 确保数据在CPU上
            if hasattr(tensor.data, 'get'):
                # CuPy 数组
                state[name] = tensor.data.get()
            elif hasattr(tensor.data, 'cpu'):
                # PyTorch 张量
                state[name] = tensor.data.cpu().numpy()
            else:
                # NumPy 数组
                state[name] = tensor.data.copy()
        
        return state

    # ==========================================================
    # 新增：load_state_dict - 从状态字典加载参数
    # ==========================================================
    
    def load_state_dict(self, state, strict=True):
        """
        从状态字典加载参数
        
        Args:
            state: 从 state_dict() 返回的字典
            strict: 是否严格模式（检查缺失和多余的键）
        
        Raises:
            RuntimeError: 当 strict=True 且存在缺失或多余参数时
        """
        # 获取当前模型的所有 Tensor 名称
        current_tensors = {name: tensor for name, tensor in self.named_tensors(include_buffers=True)}
        
        # 检查缺失和多余的键
        missing_keys = set(current_tensors.keys()) - set(state.keys())
        extra_keys = set(state.keys()) - set(current_tensors.keys())
        
        if strict:
            error_messages = []
            if missing_keys:
                error_messages.append(f"Missing keys: {sorted(missing_keys)}")
            if extra_keys:
                error_messages.append(f"Unexpected keys: {sorted(extra_keys)}")
            if error_messages:
                raise RuntimeError("Error(s) in loading state_dict:\n\t" + "\n\t".join(error_messages))
        
        # 加载参数
        loaded_count = 0
        for name, tensor in current_tensors.items():
            if name in state:
                arr = state[name]
                # 检查形状是否匹配
                if arr.shape != tensor.data.shape:
                    if strict:
                        raise RuntimeError(
                            f"Shape mismatch for {name}: expected {tensor.data.shape}, got {arr.shape}"
                        )
                    else:
                        print(f"Warning: Shape mismatch for {name}, skipping...")
                        continue
                
                # 加载数据
                if hasattr(tensor.data, 'get'):
                    # CuPy
                    tensor.data = tensor.xp.array(arr)
                else:
                    # NumPy
                    tensor.data = np.array(arr)
                loaded_count += 1
        
        # 非严格模式时，打印警告信息
        if not strict:
            if missing_keys:
                print(f"Warning: Missing keys: {sorted(missing_keys)}")
            if extra_keys:
                print(f"Warning: Unexpected keys: {sorted(extra_keys)}")
        
        return loaded_count

    # ==========================================================
    # 新增：save_state - JSON + NPZ 格式保存
    # ==========================================================
    
    def save_state(self, prefix, metadata=None, compressed=True):
        """
        以 JSON + NPZ 格式保存模型状态
        
        Args:
            prefix: 文件前缀，将生成 <prefix>.json 和 <prefix>.npz
            metadata: 额外元数据（如训练信息、配置等）
            compressed: 是否使用压缩 NPZ
        
        Returns:
            tuple: (json_path, npz_path)
        """
        json_path = f"{prefix}.json"
        npz_path = f"{prefix}.npz"
        
        # 获取状态字典
        state = self.state_dict(include_buffers=True)
        
        # 准备 JSON 元数据
        tensor_info = []
        arr_keys = []
        
        for idx, (name, arr) in enumerate(state.items()):
            key = f"arr_{idx:06d}"  # arr_000000, arr_000001, ...
            arr_keys.append(key)
            
            tensor_info.append({
                "name": name,
                "key": key,
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "requires_grad": self._get_tensor_requires_grad(name),
                "device": "cpu"  # state_dict 总是 CPU
            })
        
        # 保存 NPZ
        npz_data = {}
        for (name, arr), key in zip(state.items(), arr_keys):
            npz_data[key] = arr
        
        if compressed:
            np.savez_compressed(npz_path, **npz_data)
        else:
            np.savez(npz_path, **npz_data)
        
        # 构建 JSON
        json_data = {
            "format": "mytorch_json_npz_state",
            "version": 1,
            "npz_file": os.path.basename(npz_path),
            "model_class": self.__class__.__name__,
            "created_at": datetime.now().isoformat(),
            "metadata": metadata or {},
            "tensors": tensor_info
        }
        
        # 保存 JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 模型状态已保存:")
        print(f"   JSON: {json_path}")
        print(f"   NPZ:  {npz_path}")
        print(f"   Tensors: {len(state)}")
        
        return json_path, npz_path

    def _get_tensor_requires_grad(self, name):
        """获取指定 Tensor 的 requires_grad 状态"""
        for n, tensor in self.named_tensors(include_buffers=True):
            if n == name:
                return tensor.requires_grad
        return False

    # ==========================================================
    # 新增：load_state - 从 JSON + NPZ 加载模型状态
    # ==========================================================
    
    def load_state(self, prefix, strict=True):
        """
        从 JSON + NPZ 文件加载模型状态
        
        Args:
            prefix: 文件前缀（不含扩展名）
            strict: 是否严格模式
        
        Returns:
            dict: 加载统计信息
        """
        json_path = f"{prefix}.json"
        npz_path = f"{prefix}.npz"
        
        # 检查文件是否存在
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"NPZ file not found: {npz_path}")
        
        # 加载 JSON 元数据
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        # 验证格式
        if json_data.get("format") != "mytorch_json_npz_state":
            raise ValueError(f"Unknown format: {json_data.get('format')}")
        
        # 加载 NPZ 数据
        npz_data = np.load(npz_path, allow_pickle=True)
        
        # 构建状态字典
        state = {}
        for tensor_info in json_data["tensors"]:
            name = tensor_info["name"]
            key = tensor_info["key"]
            if key in npz_data:
                arr = npz_data[key]
                state[name] = arr
        
        # 加载到模型
        loaded_count = self.load_state_dict(state, strict=strict)
        
        print(f"✅ 模型状态已加载:")
        print(f"   JSON: {json_path}")
        print(f"   NPZ:  {npz_path}")
        print(f"   Loaded tensors: {loaded_count}")
        
        return {
            "loaded_count": loaded_count,
            "total_tensors": len(state),
            "model_class": json_data.get("model_class"),
            "created_at": json_data.get("created_at")
        }

    # ==========================================================
    # 兼容旧接口：save_weights / load_weights (pickle 格式)
    # ==========================================================
    
    def save_weights(self, path):
        """保存模型权重到文件 (pickle 格式，保留兼容)"""
        # 如果路径是 .pkl/.pickle，使用旧 pickle 格式
        if path.endswith(('.pkl', '.pickle')):
            params = self.parameters()
            weights_data = [p.data.get() if hasattr(p.data, 'get') else p.data for p in params]
            with open(path, 'wb') as f:
                pickle.dump(weights_data, f)
            print(f"模型权重已保存至: {path} (pickle格式)")
        else:
            # 否则尝试用新格式
            self.save_state(path)

    def load_weights(self, path):
        """从文件加载模型权重 (pickle 格式，保留兼容)"""
        # 如果路径是 .pkl/.pickle，使用旧 pickle 格式
        if path.endswith(('.pkl', '.pickle')):
            with open(path, 'rb') as f:
                weights_data = pickle.load(f)

            params = self.parameters()
            if len(weights_data) != len(params):
                raise ValueError("权重文件与模型结构不匹配！")

            for p, d in zip(params, weights_data):
                p.data = p.xp.array(d) if hasattr(p, 'xp') else np.array(d)
            print(f"模型权重已成功从 {path} 加载 (pickle格式)")
        else:
            # 否则尝试用新格式
            self.load_state(path)


# ==========================================================
# 辅助函数：便捷的模型保存/加载工具
# ==========================================================

def save_model_state(model, prefix, metadata=None, compressed=True):
    """
    便捷函数：保存模型状态
    
    Args:
        model: Module 实例
        prefix: 文件前缀
        metadata: 额外元数据
        compressed: 是否压缩
    """
    return model.save_state(prefix, metadata, compressed)

def load_model_state(model, prefix, strict=True):
    """
    便捷函数：加载模型状态
    
    Args:
        model: Module 实例
        prefix: 文件前缀
        strict: 是否严格模式
    """
    return model.load_state(prefix, strict)

def load_model_state_to_any(model_class, prefix, strict=True, **kwargs):
    """
    创建模型实例并加载状态
    
    Args:
        model_class: 模型类
        prefix: 文件前缀
        strict: 是否严格模式
        **kwargs: 传递给模型构造函数的参数
    
    Returns:
        Module: 加载了状态的模型实例
    """
    model = model_class(**kwargs)
    model.load_state(prefix, strict)
    return model


# ==========================================================
# 原有代码保持不变...
# ==========================================================

# 以下是你原有的所有层定义 (Linear, Conv2d, ...)
# 我为了节省空间在这里省略了，但实际使用时要把它们完整保留
# ... (所有原有的层类保持不变) ...

# 注意：上面的 Module 类已经被替换为增强版，
# 其他层类 (Linear, Conv2d, BatchNorm2d 等) 继承自 Module，
# 因此会自动获得 save_state / load_state 方法。


# ==========================================================
# 线性层
# ==========================================================

class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Xavier 初始化
        limit = np.sqrt(6 / (in_features + out_features))
        self.weight = Tensor(
            np.random.uniform(-limit, limit, (in_features, out_features)).astype(np.float32),
            requires_grad=True
        )

        self.bias = Tensor(
            np.zeros((1, out_features), dtype=np.float32),
            requires_grad=True
        )
    def forward(self, x):
        return Add()(MatMul()(x, self.weight), self.bias)


# ==========================================================
# 卷积层大家族
# ==========================================================

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super().__init__()
        if in_channels % groups != 0:
            raise ValueError("in_channels 必须能被 groups 整除")
        if out_channels % groups != 0:
            raise ValueError("out_channels 必须能被 groups 整除")
            
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _to_2tuple(kernel_size)
        self.stride = _to_2tuple(stride)
        self.padding = _to_2tuple(padding)
        self.dilation = _to_2tuple(dilation)
        self.groups = groups

        kh, kw = self.kernel_size
        
        # He 初始化 (针对 ReLU 优化)
        limit = np.sqrt(6 / (in_channels * kh * kw + out_channels))

        # 权重形状: (out_channels, in_channels // groups, k, k)
        self.weight = Tensor(
            np.random.uniform(
                -limit,
                limit,
                (out_channels, in_channels // groups, kh, kw)
            ).astype(np.float32),
            requires_grad=True
        )

        self.bias = Tensor(
            np.zeros((1, out_channels), dtype=np.float32),
            requires_grad=True
        )

    def forward(self, x):
        return Conv2dOp()(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class GroupConv2d(Conv2d):
    """分组卷积语法糖"""
    def __init__(self, in_channels, out_channels, kernel_size, groups, stride=1, padding=0, dilation=1):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups)


class DilatedConv2d(Conv2d):
    """空洞卷积语法糖"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, stride=1, padding=0, groups=1):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups)


class DepthwiseSeparableConv2d(Module):
    """深度可分离卷积 (Depthwise Separable Convolution)"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, depth_multiplier=1):
        super().__init__()
        mid_channels = in_channels * depth_multiplier
        
        # 1. 逐通道卷积 (Depthwise)
        self.depthwise = Conv2d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels
        )
        
        # 2. 逐点卷积 (Pointwise)
        self.pointwise = Conv2d(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1
        )
        self.relu = ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.pointwise(self.relu(self.depthwise(x)))


class ConvTranspose2d(Module):
    """转置卷积 (反卷积)"""
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        dilation=1,
        groups=1
    ):
        super().__init__()

        if in_channels % groups != 0 or out_channels % groups != 0:
            raise ValueError("in_channels 和 out_channels 必须能被 groups 整除")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _to_2tuple(kernel_size)
        self.stride = _to_2tuple(stride)
        self.padding = _to_2tuple(padding)
        self.output_padding = _to_2tuple(output_padding)
        self.dilation = _to_2tuple(dilation)
        self.groups = groups

        kh, kw = self.kernel_size

        # 与 Conv2d 保持类似初始化口径
        limit = np.sqrt(6 / (in_channels * kh * kw + out_channels))

        # 转置卷积权重形状：
        # (in_channels, out_channels // groups, kh, kw)
        self.weight = Tensor(
            np.random.uniform(
                -limit,
                limit,
                (in_channels, out_channels // groups, kh, kw)
            ).astype(np.float32),
            requires_grad=True
        )

        # bias 对应输出通道
        self.bias = Tensor(
            np.zeros((1, out_channels), dtype=np.float32),
            requires_grad=True
        )

    def forward(self, x: Tensor) -> Tensor:
        return ConvTranspose2dOp()(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation,
            self.groups
        )
# ==========================================================
# 辅助层
# ==========================================================

class Flatten(Module):
    def forward(self, x):
        # 自动推断 batch_size，拉平后面所有维度
        return ReshapeOp()(x, x.shape()[0], -1)


# ==========================================================
# 激活函数层
# ==========================================================

class ReLU(Module):
    def forward(self, x):
        return ReLU_Op()(x)
class Identity(Module):
    def forward(self, x):
        return x
class ELU(Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return ELU_Op(self.alpha)(x)

class Sigmoid(Module):
    def forward(self, x):
        return Sigmoid_Op()(x)


# ==========================================================
# 池化层
# ==========================================================

class MaxPool(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = _to_2tuple(kernel_size)
        # 如果没有指定 stride，默认等于 kernel_size
        self.stride = _to_2tuple(stride if stride is not None else kernel_size)
        self.padding = _to_2tuple(padding)

    def forward(self, x):
        return MaxPoolOp()(x, self.kernel_size, self.stride, self.padding)


class AvgPool(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = _to_2tuple(kernel_size)
        self.stride = _to_2tuple(stride if stride is not None else kernel_size)
        self.padding = _to_2tuple(padding)

    def forward(self, x):
        return AvgPoolOp()(x, self.kernel_size, self.stride, self.padding)


class MinPool(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = _to_2tuple(kernel_size)
        self.stride = _to_2tuple(stride if stride is not None else kernel_size)
        self.padding = _to_2tuple(padding)

    def forward(self, x):
        return MinPoolOp()(x, self.kernel_size, self.stride, self.padding)

# mytorch/modules.py

class AdaptiveAvgPool2d(Module):
    """
    自适应平均池化
    根据输入形状自动计算池化参数，确保输出形状固定
    """
    def __init__(self, output_size):
        super().__init__()
        # 统一转换为 (H, W) 元组，支持单整数输入 
        self.output_size = _to_2tuple(output_size)

    def forward(self, x):
        # 获取输入张量的形状 (N, C, H, W)
        _, _, ih, iw = x.shape()
        oh, ow = self.output_size

        # 计算步长和卷积核大小
        stride_h = ih // oh
        stride_w = iw // ow
        
        kernel_h = ih - (oh - 1) * stride_h
        kernel_w = iw - (ow - 1) * stride_w

        # 调用已有的 AvgPoolOp 算子执行计算
        # 这样可以确保操作被记录在计算图中，支持自动反向传播 [cite: 2, 3]
        return AvgPoolOp()(x, (kernel_h, kernel_w), (stride_h, stride_w), padding=0)

# ==========================================================
# 归一化层
# ==========================================================

class BatchNorm2d(Module):
    """
    二维批量归一化层 (主要配合卷积层使用)
    在组装 ResNet 等现代视觉网络时是必备组件。
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # 将权重 (gamma) 和偏置 (beta) 用 Tensor 包装，形状设为 (1, C, 1, 1)，以支持 4D 张量广播计算。
        self.weight = Tensor(
            np.ones((1, num_features, 1, 1), dtype=np.float32),
            requires_grad=True
        )

        self.bias = Tensor(
            np.zeros((1, num_features, 1, 1), dtype=np.float32),
            requires_grad=True
        )

        self.running_mean = Tensor(
            np.zeros((1, num_features, 1, 1), dtype=np.float32),
            requires_grad=False
        )

        self.running_var = Tensor(
            np.ones((1, num_features, 1, 1), dtype=np.float32),
            requires_grad=False
        )
    def forward(self, x):
        # 将当前的 training 模式传入底层 Op 中，如果在 eval 模式下会自动使用全局 mean/var
        return BatchNorm2dOp(momentum=self.momentum, eps=self.eps, is_train=self.training)(
            x, self.weight, self.bias, self.running_mean, self.running_var
        )

class FusedBatchNormReLU(Module):
    """
    训练阶段使用的 BatchNorm2d + ReLU 融合模块。

    它复用原 BatchNorm2d 的参数和 running_mean/running_var。
    """
    def __init__(self, bn: BatchNorm2d):
        super().__init__()

        self.num_features = bn.num_features
        self.eps = bn.eps
        self.momentum = bn.momentum

        # 直接复用原 BN 的 Tensor，保证 optimizer 仍然能更新同一份参数
        self.weight = bn.weight
        self.bias = bn.bias
        self.running_mean = bn.running_mean
        self.running_var = bn.running_var

        self.training = bn.training

    def forward(self, x):
        return FusedBatchNormReLUOp(
            momentum=self.momentum,
            eps=self.eps,
            is_train=self.training
        )(
            x,
            self.weight,
            self.bias,
            self.running_mean,
            self.running_var
        )

class FusedAddReLU(Module):
    """
    Module 包装版 Add + ReLU。

    用法：
        out = self.add_relu(out, identity)
    """

    def forward(self, x1, x2):
        return FusedAddReLUOp()(x1, x2)

class FusedLinearReLU(Module):
    """
    Linear + ReLU 融合模块。

    用法:
        self.fc_relu = FusedLinearReLU(in_features, out_features)

    或者从已有 Linear 创建:
        self.fc_relu = FusedLinearReLU.from_linear(old_linear)
    """

    def __init__(self, in_features, out_features):
        super().__init__()

        limit = np.sqrt(6 / (in_features + out_features))

        self.weight = Tensor(
            np.random.uniform(
                -limit,
                limit,
                (in_features, out_features)
            ).astype(np.float32),
            requires_grad=True
        )

        self.bias = Tensor(
            np.zeros((1, out_features), dtype=np.float32),
            requires_grad=True
        )

    @classmethod
    def from_linear(cls, linear):
        """
        复用已有 Linear 的 weight / bias。
        这样 optimizer 仍然能拿到同一份参数 Tensor。
        """
        in_features = linear.weight.shape()[0]
        out_features = linear.weight.shape()[1]

        obj = cls(in_features, out_features)
        obj.weight = linear.weight
        obj.bias = linear.bias

        return obj

    def forward(self, x):
        return FusedLinearReLUOp()(x, self.weight, self.bias)
    
class FusedBatchNormAddReLU(Module):
    """
    训练态 BatchNorm2d + residual Add + ReLU 融合模块。

    用于 ResNet BasicBlock:
        conv2 -> bn2 -> add(identity) -> relu2

    它复用原 BatchNorm2d 的 weight / bias / running_mean / running_var。
    """
    def __init__(self, bn: BatchNorm2d):
        super().__init__()

        self.num_features = bn.num_features
        self.eps = bn.eps
        self.momentum = bn.momentum

        self.weight = bn.weight
        self.bias = bn.bias
        self.running_mean = bn.running_mean
        self.running_var = bn.running_var

        self.training = bn.training

    def forward(self, x, identity):
        return FusedBatchNormAddReLUOp(
            momentum=self.momentum,
            eps=self.eps,
            is_train=self.training
        )(
            x,
            identity,
            self.weight,
            self.bias,
            self.running_mean,
            self.running_var
        )
    
class FusedConv2dReLU(Conv2d):
    """
    Module 包装版 Conv2d + ReLU 静态融合层。

    注意：
    - 它继承 Conv2d，复用原来的 weight / bias。
    - forward 调用 FusedConv2dReLUOp。
    - 内部卷积仍然使用 im2col + einsum。
    """

    @classmethod
    def from_conv2d(cls, conv):
        obj = cls(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups
        )

        obj.weight = conv.weight
        obj.bias = conv.bias

        return obj

    def forward(self, x):
        return FusedConv2dReLUOp()(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )
    
class FusedConvBNReLU(Module):
    """
    Module 包装版 Conv2d + BatchNorm2d + ReLU。

    复用原 Conv2d 的 weight/bias。
    复用原 BatchNorm2d 的 gamma/beta/running_mean/running_var。
    """

    def __init__(self, conv: Conv2d, bn: BatchNorm2d):
        super().__init__()

        if not isinstance(conv, Conv2d):
            raise TypeError(f"conv must be Conv2d, got {type(conv)}")

        if not isinstance(bn, BatchNorm2d):
            raise TypeError(f"bn must be BatchNorm2d, got {type(bn)}")

        if conv.out_channels != bn.num_features:
            raise ValueError(
                f"conv.out_channels must equal bn.num_features, "
                f"got {conv.out_channels} vs {bn.num_features}"
            )

        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups

        self.weight = conv.weight
        self.bias = conv.bias

        self.num_features = bn.num_features
        self.eps = bn.eps
        self.momentum = bn.momentum

        self.bn_weight = bn.weight
        self.bn_bias = bn.bias
        self.running_mean = bn.running_mean
        self.running_var = bn.running_var

        self.training = bn.training

    def forward(self, x):
        return FusedConvBNReLUOp(
            momentum=self.momentum,
            eps=self.eps,
            is_train=self.training,
        )(
            x,
            self.weight,
            self.bias,
            self.bn_weight,
            self.bn_bias,
            self.running_mean,
            self.running_var,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class FusedConvBNAddReLU(Module):
    """
    Module 包装版 Conv2d + BatchNorm2d + Add(identity) + ReLU。

    用于 ResNet BasicBlock 第二个分支：
        conv2 -> bn2 -> add(identity) -> relu2
    """

    def __init__(self, conv: Conv2d, bn: BatchNorm2d):
        super().__init__()

        if not isinstance(conv, Conv2d):
            raise TypeError(f"conv must be Conv2d, got {type(conv)}")

        if not isinstance(bn, BatchNorm2d):
            raise TypeError(f"bn must be BatchNorm2d, got {type(bn)}")

        if conv.out_channels != bn.num_features:
            raise ValueError(
                f"conv.out_channels must equal bn.num_features, "
                f"got {conv.out_channels} vs {bn.num_features}"
            )

        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups

        self.weight = conv.weight
        self.bias = conv.bias

        self.num_features = bn.num_features
        self.eps = bn.eps
        self.momentum = bn.momentum

        self.bn_weight = bn.weight
        self.bn_bias = bn.bias
        self.running_mean = bn.running_mean
        self.running_var = bn.running_var

        self.training = bn.training

    def forward(self, x, identity):
        return FusedConvBNAddReLUOp(
            momentum=self.momentum,
            eps=self.eps,
            is_train=self.training,
        )(
            x,
            identity,
            self.weight,
            self.bias,
            self.bn_weight,
            self.bn_bias,
            self.running_mean,
            self.running_var,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
