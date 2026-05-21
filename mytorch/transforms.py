# mytorch/transforms.py
"""
数据增强和变换模块
提供类似 torchvision.transforms 的接口
"""
import numpy as np
from typing import Any, Dict, Optional, Tuple
from mytorch.tensor import Tensor


class Compose:
    """组合多个变换"""
    
    def __init__(self, transforms):
        """
        :param transforms: 变换列表
        """
        self.transforms = transforms
    
    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string


class ToTensor:
    """将numpy数组转换为Tensor"""
    
    def __call__(self, data):
        if isinstance(data, Tensor):
            return data
        if isinstance(data, np.ndarray):
            return Tensor(data)
        raise TypeError(f"不支持的类型: {type(data)}")
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"


class Normalize:
    """标准化：(data - mean) / std"""
    
    def __init__(self, mean, std):
        """
        :param mean: 均值，可以是标量或与数据通道数相同的数组
        :param std: 标准差，可以是标量或与数据通道数相同的数组
        """
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
    
    def __call__(self, data):
        if isinstance(data, Tensor):
            data_np = data.data
        else:
            data_np = data
        
        # 广播机制支持
        if data_np.ndim == 4:  # (N, C, H, W)
            mean = self.mean.reshape(1, -1, 1, 1)
            std = self.std.reshape(1, -1, 1, 1)
        elif data_np.ndim == 3:  # (C, H, W)
            mean = self.mean.reshape(-1, 1, 1)
            std = self.std.reshape(-1, 1, 1)
        else:
            mean = self.mean
            std = self.std
        
        normalized = (data_np - mean) / (std + 1e-8)
        
        if isinstance(data, Tensor):
            return Tensor(normalized)
        return normalized
    
    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean.tolist()}, std={self.std.tolist()})"


class RandomHorizontalFlip:
    """随机水平翻转"""
    
    def __init__(self, p=0.5):
        """
        :param p: 翻转概率
        """
        self.p = p
    
    def __call__(self, data):
        if isinstance(data, Tensor):
            data_np = data.data
            is_tensor = True
        else:
            data_np = data
            is_tensor = False
        
        if np.random.rand() < self.p:
            # 水平翻转：沿着最后一个空间维度翻转
            if data_np.ndim == 3:  # (C, H, W)
                flipped = data_np[:, :, ::-1].copy()
            elif data_np.ndim == 4:  # (N, C, H, W)
                flipped = data_np[:, :, :, ::-1].copy()
            else:
                flipped = data_np
        else:
            flipped = data_np
        
        if is_tensor:
            return Tensor(flipped)
        return flipped
    
    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"


class RandomVerticalFlip:
    """随机垂直翻转"""
    
    def __init__(self, p=0.5):
        """
        :param p: 翻转概率
        """
        self.p = p
    
    def __call__(self, data):
        if isinstance(data, Tensor):
            data_np = data.data
            is_tensor = True
        else:
            data_np = data
            is_tensor = False
        
        if np.random.rand() < self.p:
            # 垂直翻转：沿着倒数第二个空间维度翻转
            if data_np.ndim == 3:  # (C, H, W)
                flipped = data_np[:, ::-1, :].copy()
            elif data_np.ndim == 4:  # (N, C, H, W)
                flipped = data_np[:, :, ::-1, :].copy()
            else:
                flipped = data_np
        else:
            flipped = data_np
        
        if is_tensor:
            return Tensor(flipped)
        return flipped
    
    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"


class RandomCrop:
    """随机裁剪"""
    
    def __init__(self, size, padding=None):
        """
        :param size: 裁剪尺寸 (height, width) 或单个整数表示正方形
        :param padding: 填充大小（可选）
        """
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = tuple(size)
        self.padding = padding
    
    def __call__(self, data):
        if isinstance(data, Tensor):
            data_np = data.data
            is_tensor = True
        else:
            data_np = data
            is_tensor = False
        
        # 如果需要填充
        if self.padding is not None and self.padding > 0:
            if data_np.ndim == 3:  # (C, H, W)
                data_np = np.pad(data_np, ((0, 0), (self.padding, self.padding), 
                                           (self.padding, self.padding)), mode='reflect')
            elif data_np.ndim == 4:  # (N, C, H, W)
                data_np = np.pad(data_np, ((0, 0), (0, 0), (self.padding, self.padding), 
                                           (self.padding, self.padding)), mode='reflect')
        
        h, w = data_np.shape[-2], data_np.shape[-1]
        th, tw = self.size
        
        if h < th or w < tw:
            raise ValueError(f"输入尺寸 ({h}, {w}) 小于裁剪尺寸 ({th}, {tw})")
        
        # 随机选择裁剪位置
        i = np.random.randint(0, h - th + 1)
        j = np.random.randint(0, w - tw + 1)
        
        if data_np.ndim == 3:  # (C, H, W)
            cropped = data_np[:, i:i+th, j:j+tw].copy()
        elif data_np.ndim == 4:  # (N, C, H, W)
            cropped = data_np[:, :, i:i+th, j:j+tw].copy()
        else:
            cropped = data_np
        
        if is_tensor:
            return Tensor(cropped)
        return cropped
    
    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size}, padding={self.padding})"


class CenterCrop:
    """中心裁剪"""
    
    def __init__(self, size):
        """
        :param size: 裁剪尺寸 (height, width) 或单个整数
        """
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = tuple(size)
    
    def __call__(self, data):
        if isinstance(data, Tensor):
            data_np = data.data
            is_tensor = True
        else:
            data_np = data
            is_tensor = False
        
        h, w = data_np.shape[-2], data_np.shape[-1]
        th, tw = self.size
        
        i = (h - th) // 2
        j = (w - tw) // 2
        
        if data_np.ndim == 3:  # (C, H, W)
            cropped = data_np[:, i:i+th, j:j+tw]
        elif data_np.ndim == 4:  # (N, C, H, W)
            cropped = data_np[:, :, i:i+th, j:j+tw]
        else:
            cropped = data_np
        
        if is_tensor:
            return Tensor(cropped)
        return cropped
    
    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size})"


class Resize:
    """调整图像大小"""
    
    def __init__(self, size):
        """
        :param size: 目标尺寸 (height, width) 或单个整数
        """
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = tuple(size)
    
    def __call__(self, data):
        try:
            import cv2
        except ImportError:
            raise ImportError("Resize 需要安装 opencv-python")
        
        if isinstance(data, Tensor):
            data_np = data.data
            is_tensor = True
        else:
            data_np = data
            is_tensor = False
        
        target_h, target_w = self.size
        
        if data_np.ndim == 3:  # (C, H, W)
            # 转换为 (H, W, C) 进行resize
            hwc = np.transpose(data_np, (1, 2, 0))
            resized_hwc = cv2.resize(hwc, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            # 确保形状正确
            if resized_hwc.ndim == 2:  # 单通道图像
                resized_hwc = resized_hwc[:, :, np.newaxis]
            resized = np.transpose(resized_hwc, (2, 0, 1))
        elif data_np.ndim == 4:  # (N, C, H, W)
            resized = np.zeros((data_np.shape[0], data_np.shape[1], target_h, target_w), 
                              dtype=data_np.dtype)
            for i in range(data_np.shape[0]):
                hwc = np.transpose(data_np[i], (1, 2, 0))
                r = cv2.resize(hwc, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                if r.ndim == 2:
                    r = r[:, :, np.newaxis]
                resized[i] = np.transpose(r, (2, 0, 1))
        else:
            resized = data_np
        
        if is_tensor:
            return Tensor(resized)
        return resized
    
    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size})"


class RandomRotation:
    """随机旋转"""
    
    def __init__(self, degrees, fill_value=0):
        """
        :param degrees: 旋转角度范围 (-degrees, +degrees)
        :param fill_value: 填充值
        """
        self.degrees = degrees
        self.fill_value = fill_value
    
    def __call__(self, data):
        try:
            import cv2
        except ImportError:
            raise ImportError("RandomRotation 需要安装 opencv-python")
        
        if isinstance(data, Tensor):
            data_np = data.data
            is_tensor = True
        else:
            data_np = data
            is_tensor = False
        
        angle = np.random.uniform(-self.degrees, self.degrees)
        
        if data_np.ndim == 3:  # (C, H, W)
            h, w = data_np.shape[1], data_np.shape[2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            rotated = np.zeros_like(data_np)
            for c in range(data_np.shape[0]):
                rotated[c] = cv2.warpAffine(data_np[c], M, (w, h), 
                                           borderValue=self.fill_value)
        elif data_np.ndim == 4:  # (N, C, H, W)
            rotated = np.zeros_like(data_np)
            for n in range(data_np.shape[0]):
                h, w = data_np.shape[2], data_np.shape[3]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                for c in range(data_np.shape[1]):
                    rotated[n, c] = cv2.warpAffine(data_np[n, c], M, (w, h), 
                                                  borderValue=self.fill_value)
        else:
            rotated = data_np
        
        if is_tensor:
            return Tensor(rotated)
        return rotated
    
    def __repr__(self):
        return f"{self.__class__.__name__}(degrees={self.degrees})"


class Lambda:
    """应用自定义函数"""
    
    def __init__(self, func):
        """
        :param func: 自定义变换函数
        """
        self.func = func
    
    def __call__(self, data):
        return self.func(data)
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.func.__name__})"


# 添加到 mytorch/transforms.py 末尾

class RandomBrightness:
    """随机调整亮度"""

    def __init__(self, brightness_range=(0.8, 1.2)):
        """
        Args:
            brightness_range: 亮度调整范围
        """
        self.brightness_range = brightness_range

    def __call__(self, data):
        if isinstance(data, Tensor):
            data_np = data.data
            is_tensor = True
        else:
            data_np = data
            is_tensor = False

        factor = np.random.uniform(*self.brightness_range)
        adjusted = data_np * factor
        adjusted = np.clip(adjusted, 0, 1)

        if is_tensor:
            return Tensor(adjusted)
        return adjusted

    def __repr__(self):
        return f"{self.__class__.__name__}(range={self.brightness_range})"


class RandomContrast:
    """随机调整对比度"""

    def __init__(self, contrast_range=(0.8, 1.2)):
        """
        Args:
            contrast_range: 对比度调整范围
        """
        self.contrast_range = contrast_range

    def __call__(self, data):
        if isinstance(data, Tensor):
            data_np = data.data
            is_tensor = True
        else:
            data_np = data
            is_tensor = False

        factor = np.random.uniform(*self.contrast_range)
        mean = np.mean(data_np, axis=tuple(range(1, data_np.ndim)), keepdims=True)
        adjusted = mean + factor * (data_np - mean)
        adjusted = np.clip(adjusted, 0, 1)

        if is_tensor:
            return Tensor(adjusted)
        return adjusted

    def __repr__(self):
        return f"{self.__class__.__name__}(range={self.contrast_range})"


class RandomNoise:
    """添加随机噪声"""

    def __init__(self, noise_std=0.05):
        """
        Args:
            noise_std: 噪声标准差
        """
        self.noise_std = noise_std

    def __call__(self, data):
        if isinstance(data, Tensor):
            data_np = data.data
            is_tensor = True
        else:
            data_np = data
            is_tensor = False

        noise = np.random.normal(0, self.noise_std, data_np.shape)
        noisy = data_np + noise
        noisy = np.clip(noisy, 0, 1)

        if is_tensor:
            return Tensor(noisy)
        return noisy

    def __repr__(self):
        return f"{self.__class__.__name__}(std={self.noise_std})"


def _split_sample(sample: Any) -> Tuple[Any, Optional[Any], bool]:
    if isinstance(sample, tuple) and len(sample) == 2:
        return sample[0], sample[1], True
    return sample, None, False


def _merge_sample(image: Any, label: Optional[Any], had_label: bool) -> Any:
    if had_label:
        return image, label
    return image


def _to_float_label(label: Any) -> float:
    if isinstance(label, Tensor):
        return float(np.asarray(label.data).reshape(-1)[0])
    return float(np.asarray(label).reshape(-1)[0])


def _replace_label_like(label: Any, value: float) -> Any:
    if label is None:
        return None
    if isinstance(label, Tensor):
        return Tensor(np.asarray(label.data, dtype=np.float32) * 0.0 + np.float32(value))
    arr = np.asarray(label)
    if arr.ndim == 0:
        return float(value)
    return np.asarray(arr, dtype=np.float32) * 0.0 + np.float32(value)


def _clip01(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0.0, 1.0).astype(np.float32)


class LabelAwareCompose:
    """Compose transforms that may update both image and steering label.

    It accepts either ``transform(image, label)`` or ``transform((image, label))``.
    Image-only transforms can still return just the image.
    """

    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, image, label=None):
        sample = (image, label) if label is not None else image
        had_label = isinstance(sample, tuple) and len(sample) == 2
        for transform in self.transforms:
            result = transform(sample)
            if had_label and not (isinstance(result, tuple) and len(result) == 2):
                _, current_label, _ = _split_sample(sample)
                result = (result, current_label)
            sample = result
        return sample

    def __repr__(self):
        body = "\n".join(f"    {t}" for t in self.transforms)
        return f"{self.__class__.__name__}(\n{body}\n)"


class DonkeyHorizontalFlip:
    """Horizontal flip for steering regression.

    A horizontal flip mirrors the road scene, so the steering angle sign must
    also be inverted. The default layout is HWC because DonkeyCar images are
    loaded with cv2/PIL style image arrays before they are converted to CHW.
    """

    def __init__(self, p=0.5, layout="HWC"):
        self.p = p
        self.layout = layout.upper()

    def __call__(self, sample, label=None):
        if label is not None:
            sample = (sample, label)
        image, label, had_label = _split_sample(sample)
        if np.random.rand() >= self.p:
            return _merge_sample(image, label, had_label)

        if self.layout == "HWC":
            flipped = np.ascontiguousarray(image[:, ::-1, ...])
        elif self.layout == "CHW":
            flipped = np.ascontiguousarray(image[..., ::-1])
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")

        if had_label:
            label = _replace_label_like(label, -_to_float_label(label))
        return _merge_sample(flipped, label, had_label)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p}, layout='{self.layout}')"


class DonkeyBrightnessContrast:
    """Brightness/contrast jitter for normalized HWC images."""

    def __init__(self, brightness=0.18, contrast=0.18):
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, sample, label=None):
        if label is not None:
            sample = (sample, label)
        image, label, had_label = _split_sample(sample)
        alpha = 1.0 + np.random.uniform(-self.contrast, self.contrast)
        beta = np.random.uniform(-self.brightness, self.brightness)
        image = _clip01(image.astype(np.float32) * alpha + beta)
        return _merge_sample(image, label, had_label)

    def __repr__(self):
        return f"{self.__class__.__name__}(brightness={self.brightness}, contrast={self.contrast})"


class DonkeyGammaJitter:
    """Mild exposure jitter that preserves lane geometry."""

    def __init__(self, gamma_range=(0.9, 1.1)):
        self.gamma_range = gamma_range

    def __call__(self, sample, label=None):
        if label is not None:
            sample = (sample, label)
        image, label, had_label = _split_sample(sample)
        gamma = np.random.uniform(*self.gamma_range)
        image = np.power(np.clip(image.astype(np.float32), 0.0, 1.0), gamma)
        return _merge_sample(_clip01(image), label, had_label)

    def __repr__(self):
        return f"{self.__class__.__name__}(gamma_range={self.gamma_range})"


class DonkeyHSVJitter:
    """HSV color jitter for normalized RGB/HWC images."""

    def __init__(self, brightness=0.22, contrast=0.25, saturation=0.25, value=0.15):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.value = value

    def __call__(self, sample, label=None):
        if label is not None:
            sample = (sample, label)
        try:
            import cv2
        except ImportError:
            raise ImportError("DonkeyHSVJitter requires opencv-python")

        image, label, had_label = _split_sample(sample)
        image = DonkeyBrightnessContrast(self.brightness, self.contrast)(image)
        hsv = cv2.cvtColor((image * 255.0).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] *= 1.0 + np.random.uniform(-self.saturation, self.saturation)
        hsv[:, :, 2] *= 1.0 + np.random.uniform(-self.value, self.value)
        hsv[:, :, 1:] = np.clip(hsv[:, :, 1:], 0, 255)
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
        return _merge_sample(_clip01(image), label, had_label)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(brightness={self.brightness}, "
            f"contrast={self.contrast}, saturation={self.saturation}, value={self.value})"
        )


class DonkeyGaussianNoise:
    """Gaussian pixel noise for normalized images."""

    def __init__(self, std=0.015):
        self.std = std

    def __call__(self, sample, label=None):
        if label is not None:
            sample = (sample, label)
        image, label, had_label = _split_sample(sample)
        noise = np.random.normal(0.0, self.std, size=image.shape).astype(np.float32)
        return _merge_sample(_clip01(image.astype(np.float32) + noise), label, had_label)

    def __repr__(self):
        return f"{self.__class__.__name__}(std={self.std})"


class DonkeyCenterCropResize:
    """Fixed center crop followed by resize without steering-label changes.

    Unlike random crop, this transform keeps the crop center fixed, so it does
    not shift the road scene left or right relative to the image center. The
    default keeps full width and removes a small amount of vertical context,
    which is a conservative DonkeyCar ROI-style crop.
    """

    def __init__(self, height_ratio=0.85, width_ratio=1.0):
        if not 0.0 < height_ratio <= 1.0:
            raise ValueError("height_ratio must be in (0, 1]")
        if not 0.0 < width_ratio <= 1.0:
            raise ValueError("width_ratio must be in (0, 1]")
        self.height_ratio = height_ratio
        self.width_ratio = width_ratio

    def __call__(self, sample, label=None):
        if label is not None:
            sample = (sample, label)
        try:
            import cv2
        except ImportError:
            raise ImportError("DonkeyCenterCropResize requires opencv-python")

        image, label, had_label = _split_sample(sample)
        h, w = image.shape[:2]
        crop_h = max(8, int(round(h * self.height_ratio)))
        crop_w = max(8, int(round(w * self.width_ratio)))
        top = max(0, (h - crop_h) // 2)
        left = max(0, (w - crop_w) // 2)
        crop = image[top:top + crop_h, left:left + crop_w]
        image = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        return _merge_sample(_clip01(image), label, had_label)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(height_ratio={self.height_ratio}, "
            f"width_ratio={self.width_ratio})"
        )


class DonkeyRandomCropResize:
    """Random crop followed by resize with optional steering compensation.

    ``steering_per_shift`` is measured in steering-label units per normalized
    horizontal crop-center shift. It defaults to 0 because the correct value is
    camera- and controller-dependent and should be calibrated experimentally.
    """

    def __init__(self, min_scale=0.82, steering_per_shift=0.0):
        self.min_scale = min_scale
        self.steering_per_shift = steering_per_shift

    def __call__(self, sample, label=None):
        if label is not None:
            sample = (sample, label)
        try:
            import cv2
        except ImportError:
            raise ImportError("DonkeyRandomCropResize requires opencv-python")

        image, label, had_label = _split_sample(sample)
        h, w = image.shape[:2]
        scale = np.random.uniform(self.min_scale, 1.0)
        crop_h = max(8, int(round(h * scale)))
        crop_w = max(8, int(round(w * scale)))
        top = np.random.randint(0, h - crop_h + 1)
        left = np.random.randint(0, w - crop_w + 1)
        crop = image[top:top + crop_h, left:left + crop_w]
        image = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)

        if had_label and self.steering_per_shift != 0.0:
            crop_center_x = left + crop_w / 2.0
            normalized_shift = (w / 2.0 - crop_center_x) / max(1.0, w / 2.0)
            label = _replace_label_like(label, _to_float_label(label) + self.steering_per_shift * normalized_shift)
        return _merge_sample(_clip01(image), label, had_label)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(min_scale={self.min_scale}, "
            f"steering_per_shift={self.steering_per_shift})"
        )


class DonkeyRandomRotation:
    """Random in-plane rotation with optional steering compensation.

    ``steering_per_degree`` defaults to 0 for reproducibility. Set it after
    calibration if rotation is used as a physical steering perturbation.
    """

    def __init__(self, degrees=8.0, steering_per_degree=0.0):
        self.degrees = degrees
        self.steering_per_degree = steering_per_degree

    def __call__(self, sample, label=None):
        if label is not None:
            sample = (sample, label)
        try:
            import cv2
        except ImportError:
            raise ImportError("DonkeyRandomRotation requires opencv-python")

        image, label, had_label = _split_sample(sample)
        h, w = image.shape[:2]
        angle = np.random.uniform(-self.degrees, self.degrees)
        matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        image = cv2.warpAffine(
            image,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        ).astype(np.float32)

        if had_label and self.steering_per_degree != 0.0:
            label = _replace_label_like(label, _to_float_label(label) + self.steering_per_degree * angle)
        return _merge_sample(_clip01(image), label, had_label)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(degrees={self.degrees}, "
            f"steering_per_degree={self.steering_per_degree})"
        )


class MixUpBatch:
    """Batch-level MixUp that mixes images and steering labels together."""

    def __init__(self, alpha=0.4):
        self.alpha = alpha

    def __call__(self, x, y=None):
        if y is None:
            x, y = x
            return_tuple = True
        else:
            return_tuple = False

        lam = float(np.random.beta(self.alpha, self.alpha))
        if x.__class__.__module__.startswith("torch"):
            import torch
            perm = torch.randperm(x.size(0), device=x.device)
            x_out = lam * x + (1.0 - lam) * x[perm]
            y_out = lam * y + (1.0 - lam) * y[perm]
        else:
            perm = np.random.permutation(x.shape[0])
            x_out = lam * x + (1.0 - lam) * x[perm]
            y_out = lam * y + (1.0 - lam) * y[perm]
        return (x_out, y_out) if return_tuple else (x_out, y_out)

    def __repr__(self):
        return f"{self.__class__.__name__}(alpha={self.alpha})"


class LocalMixUpBatch:
    """MixUp only between samples with close steering labels."""

    def __init__(self, alpha=0.2, max_label_diff=0.05):
        self.alpha = alpha
        self.max_label_diff = max_label_diff

    def __call__(self, x, y=None):
        if y is None:
            x, y = x
        lam = float(np.random.beta(self.alpha, self.alpha))

        if x.__class__.__module__.startswith("torch"):
            import torch

            flat_y = y.reshape(y.size(0), -1)[:, 0]
            partners = []
            for i in range(x.size(0)):
                candidates = torch.nonzero(torch.abs(flat_y - flat_y[i]) <= self.max_label_diff, as_tuple=False).flatten()
                candidates = candidates[candidates != i]
                if candidates.numel() == 0:
                    partners.append(i)
                else:
                    j = candidates[torch.randint(candidates.numel(), (1,), device=x.device)].item()
                    partners.append(j)
            perm = torch.tensor(partners, device=x.device, dtype=torch.long)
            x_out = lam * x + (1.0 - lam) * x[perm]
            y_out = lam * y + (1.0 - lam) * y[perm]
        else:
            flat_y = y.reshape(y.shape[0], -1)[:, 0]
            partners = []
            for i in range(x.shape[0]):
                candidates = np.flatnonzero(np.abs(flat_y - flat_y[i]) <= self.max_label_diff)
                candidates = candidates[candidates != i]
                partners.append(i if candidates.size == 0 else int(np.random.choice(candidates)))
            perm = np.asarray(partners, dtype=np.int64)
            x_out = lam * x + (1.0 - lam) * x[perm]
            y_out = lam * y + (1.0 - lam) * y[perm]
        return x_out, y_out

    def __repr__(self):
        return f"{self.__class__.__name__}(alpha={self.alpha}, max_label_diff={self.max_label_diff})"


class CutMixBatch:
    """Batch-level CutMix with area-weighted steering-label mixing."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, x, y=None):
        if y is None:
            x, y = x
            return_tuple = True
        else:
            return_tuple = False

        lam = float(np.random.beta(self.alpha, self.alpha))
        b, _, h, w = x.shape
        cut_ratio = np.sqrt(1.0 - lam)
        cut_w = int(w * cut_ratio)
        cut_h = int(h * cut_ratio)
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        x1 = int(np.clip(cx - cut_w // 2, 0, w))
        y1 = int(np.clip(cy - cut_h // 2, 0, h))
        x2 = int(np.clip(cx + cut_w // 2, 0, w))
        y2 = int(np.clip(cy + cut_h // 2, 0, h))
        patch_area = max(0, x2 - x1) * max(0, y2 - y1)
        lam_adj = 1.0 - patch_area / float(w * h)

        if x.__class__.__module__.startswith("torch"):
            import torch
            perm = torch.randperm(b, device=x.device)
            x_out = x.clone()
            x_out[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
            y_out = lam_adj * y + (1.0 - lam_adj) * y[perm]
        else:
            perm = np.random.permutation(b)
            x_out = x.copy()
            x_out[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
            y_out = lam_adj * y + (1.0 - lam_adj) * y[perm]
        return (x_out, y_out) if return_tuple else (x_out, y_out)

    def __repr__(self):
        return f"{self.__class__.__name__}(alpha={self.alpha})"


def build_donkeycar_transform(
    variant: str,
    crop_steering_gain: float = 0.0,
    rotate_steering_gain: float = 0.0,
    use_random_crop: bool = False,
    use_random_rotation: bool = False,
    center_crop_height_ratio: float = 0.85,
    center_crop_width_ratio: float = 1.0,
) -> LabelAwareCompose:
    """Build the image/label transform used by DonkeyCar augmentation ablations."""

    if variant == "none":
        return LabelAwareCompose([])
    if variant == "flip_only":
        return LabelAwareCompose([
            DonkeyHorizontalFlip(p=0.3),
        ])
    if variant == "center_crop":
        return LabelAwareCompose([
            DonkeyCenterCropResize(
                height_ratio=center_crop_height_ratio,
                width_ratio=center_crop_width_ratio,
            ),
        ])
    if variant == "brightness_only":
        return LabelAwareCompose([
            DonkeyBrightnessContrast(brightness=0.10, contrast=0.0),
        ])
    if variant == "contrast_only":
        return LabelAwareCompose([
            DonkeyBrightnessContrast(brightness=0.0, contrast=0.10),
        ])
    if variant == "gamma_only":
        return LabelAwareCompose([
            DonkeyGammaJitter(gamma_range=(0.9, 1.1)),
        ])
    if variant == "noise_only":
        return LabelAwareCompose([
            DonkeyGaussianNoise(std=0.005),
        ])
    if variant == "hsv_only":
        return LabelAwareCompose([
            DonkeyHSVJitter(brightness=0.0, contrast=0.0, saturation=0.10, value=0.10),
        ])
    if variant in {"mixup_only", "local_mixup_only", "cutmix_only"}:
        return LabelAwareCompose([])
    if variant in {"donkey_safe", "local_mixup"}:
        return LabelAwareCompose([
            DonkeyHorizontalFlip(p=0.3),
            DonkeyGammaJitter(gamma_range=(0.9, 1.1)),
            DonkeyBrightnessContrast(brightness=0.10, contrast=0.10),
            DonkeyGaussianNoise(std=0.005),
        ])
    if variant in {"basic", "mixup", "cutmix"}:
        return LabelAwareCompose([
            DonkeyHorizontalFlip(p=0.3),
            DonkeyBrightnessContrast(brightness=0.10, contrast=0.10),
            DonkeyGaussianNoise(std=0.005),
        ])
    if variant == "crop_rotate_jitter":
        transforms = [
            DonkeyHorizontalFlip(p=0.3),
            DonkeyHSVJitter(brightness=0.10, contrast=0.10, saturation=0.10, value=0.10),
            DonkeyGaussianNoise(std=0.005),
        ]
        if use_random_crop:
            transforms.insert(1, DonkeyRandomCropResize(min_scale=0.82, steering_per_shift=crop_steering_gain))
        if use_random_rotation:
            insert_at = 2 if use_random_crop else 1
            transforms.insert(insert_at, DonkeyRandomRotation(degrees=8.0, steering_per_degree=rotate_steering_gain))
        return LabelAwareCompose(transforms)
    raise ValueError(f"Unknown DonkeyCar augmentation variant: {variant}")


def build_donkeycar_batch_transform(
    variant: str,
    mix_alpha: float = 0.4,
    cutmix_alpha: float = 1.0,
    local_mix_alpha: float = 0.2,
    local_mix_max_diff: float = 0.05,
):
    if variant in {"mixup", "mixup_only"}:
        return MixUpBatch(alpha=mix_alpha)
    if variant in {"local_mixup", "local_mixup_only"}:
        return LocalMixUpBatch(alpha=local_mix_alpha, max_label_diff=local_mix_max_diff)
    if variant in {"cutmix", "cutmix_only"}:
        return CutMixBatch(alpha=cutmix_alpha)
    return None


def describe_donkeycar_augmentation(
    variant: str,
    crop_steering_gain: float = 0.0,
    rotate_steering_gain: float = 0.0,
    mix_alpha: float = 0.4,
    cutmix_alpha: float = 1.0,
    local_mix_alpha: float = 0.2,
    local_mix_max_diff: float = 0.05,
    use_random_crop: bool = False,
    use_random_rotation: bool = False,
    center_crop_height_ratio: float = 0.85,
    center_crop_width_ratio: float = 1.0,
) -> Dict[str, Any]:
    return {
        "variant": variant,
        "image_transform": repr(build_donkeycar_transform(
            variant,
            crop_steering_gain,
            rotate_steering_gain,
            use_random_crop,
            use_random_rotation,
            center_crop_height_ratio,
            center_crop_width_ratio,
        )),
        "batch_transform": repr(build_donkeycar_batch_transform(
            variant,
            mix_alpha,
            cutmix_alpha,
            local_mix_alpha,
            local_mix_max_diff,
        )),
        "crop_steering_gain": crop_steering_gain,
        "rotate_steering_gain": rotate_steering_gain,
        "random_crop_enabled": use_random_crop,
        "random_rotation_enabled": use_random_rotation,
        "center_crop_height_ratio": center_crop_height_ratio,
        "center_crop_width_ratio": center_crop_width_ratio,
        "label_compensation_note": (
            "random crop and random rotation are disabled; this variant keeps only label-safe flip "
            "plus color/noise augmentation."
            if variant == "crop_rotate_jitter" and not use_random_crop and not use_random_rotation
            else "crop/rotation steering compensation is disabled; calibrate non-zero gains before "
            "treating spatial augmentation as physically consistent."
            if variant == "crop_rotate_jitter"
            and (use_random_crop or use_random_rotation)
            and crop_steering_gain == 0.0
            and rotate_steering_gain == 0.0
            else "label-aware transforms are active where configured."
        ),
    }
