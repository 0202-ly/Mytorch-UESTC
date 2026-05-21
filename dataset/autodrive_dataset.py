import os
import numpy as np

# 只有这里需要 cv2
try:
    import cv2
except ImportError:
    cv2 = None

from mytorch import Tensor
from mytorch import Dataset


def resolve_list_path(data_root, list_path):
    if os.path.isabs(list_path):
        return list_path
    candidate = os.path.join(data_root, list_path)
    if os.path.exists(candidate):
        return candidate
    return list_path


class AutoDriveDataset(Dataset):
    """
    适配 MyTorch 框架的自动驾驶数据集加载器
    """

    def __init__(self, mode, transform=None, data_root="./", list_path=None):
        super().__init__()

        if cv2 is None:
            raise ImportError("AutoDriveDataset 需要安装 opencv-python。请运行 pip install opencv-python")

        self.mode = mode.lower()
        self.transform = transform
        self.data_root = data_root

        assert self.mode in {"train", "val"}

        if list_path is None:
            list_path = "train.txt" if self.mode == "train" else "val.txt"
        file_path = resolve_list_path(data_root, list_path)
        self.list_path = file_path

        self.file_list = list()

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到数据列表文件: {file_path}")

        with open(file_path, "r") as f:
            files = f.readlines()
            for file in files:
                line = file.strip()
                if not line:
                    continue
                parts = line.split(" ")
                img_path = parts[0]
                steering = float(parts[1])
                self.file_list.append([img_path, steering])

    def __getitem__(self, i):
        img_rel_path = self.file_list[i][0]
        full_img_path = os.path.join(self.data_root, img_rel_path)

        img = cv2.imread(full_img_path)
        if img is None:
            raise ValueError(f"无法读取图像: {full_img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label_val = self.file_list[i][1]
        if self.transform:
            transformed = self.transform(img, label_val)
            if isinstance(transformed, tuple) and len(transformed) == 2:
                img, label_val = transformed
            else:
                img = transformed
        else:
            # HWC -> CHW 并归一化
            img = img.transpose(2, 0, 1).astype(np.float32) / 255.0

        if isinstance(img, np.ndarray):
            if img.ndim == 3 and img.shape[-1] in (1, 3, 4) and img.shape[0] not in (1, 3, 4):
                img = img.transpose(2, 0, 1)
            img = img.astype(np.float32)
            if img.size and img.max() > 1.5:
                img = img / 255.0

        # 转换为 MyTorch Tensor
        img_tensor = Tensor(img)
        label_tensor = Tensor(np.array([label_val], dtype=np.float32))

        return img_tensor, label_tensor

    def __len__(self):
        return len(self.file_list)
