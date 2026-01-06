# 文件： mytorch/dataset.py
import cv2
import os
import gzip
import struct
import hashlib
import urllib.request
from urllib.error import URLError
import numpy as np
from .tensor import Tensor

class Dataset:
    def __getitem__(self, item):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


# ==========================================================
# 辅助函数 (用于 MD5 校验和 .gz 解析)
# (这部分不变)
# ==========================================================

def _check_md5(filepath: str, expected_md5: str) -> bool:
    """计算文件的 MD5 值并进行比较"""
    if not os.path.exists(filepath):
        return False
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    digest = hash_md5.hexdigest()
    if digest == expected_md5:
        return True
    else:
        print(f"MD5 校验失败: {filepath} (预期: {expected_md5}, 得到: {digest})")
        return False


def _parse_idx_images(filepath: str) -> np.ndarray:
    """一个辅助函数，用于解析 'idx3-ubyte.gz' 图像文件。"""
    with gzip.open(filepath, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        buffer = f.read()
        data = np.frombuffer(buffer, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols)
        data = np.expand_dims(data, axis=1).astype(np.float32) / 255.0
        return data


def _parse_idx_labels(filepath: str) -> np.ndarray:
    """一个辅助函数，用于解析 'idx1-ubyte.gz' 标签文件。"""
    with gzip.open(filepath, 'rb') as f:
        magic, num_items = struct.unpack('>II', f.read(8))
        buffer = f.read()
        data = np.frombuffer(buffer, dtype=np.uint8)
        data = data.astype(np.int64)
        return data


# ==========================================================
# 具备下载功能的 MNIST Dataset
# ==========================================================

class MnistDataset(Dataset):
    """
    一个从 'data/MNIST/raw' 文件夹 加载原始 .gz 文件的数据集。
    如果文件不存在且 download=True，将自动从网络下载。
    它继承自您在 dataset.py 中定义的基类。
    """

    # --- 修正：更新 t10k-labels 的 MD5 哈希值 ---
    resources = [
        ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
        ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
        ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c")
    ]

    # S3 镜像 URL
    base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"

    def __init__(self, data_root="data", train=True, download=False):
        super().__init__()
        self.data_root = data_root
        self.train = train

        self.raw_folder = os.path.join(self.data_root, 'MNIST', 'raw')
        os.makedirs(self.raw_folder, exist_ok=True)

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError(f"数据集未找到或已损坏。您可以在 {self.raw_folder} 中找到它，"
                               "或者通过设置 download=True 重新下载。")

        # (加载数据的逻辑不变)
        if self.train:
            img_file = os.path.join(self.raw_folder, self.resources[0][0])
            lbl_file = os.path.join(self.raw_folder, self.resources[1][0])
        else:
            img_file = os.path.join(self.raw_folder, self.resources[2][0])
            lbl_file = os.path.join(self.raw_folder, self.resources[3][0])

        self.data = _parse_idx_images(img_file)
        self.labels = _parse_idx_labels(lbl_file)

    def _check_integrity(self) -> bool:
        """检查文件是否存在且MD5正确"""
        resources_to_check = self.resources[:2] if self.train else self.resources[2:]

        all_files_ok = True
        for filename, md5 in resources_to_check:
            filepath = os.path.join(self.raw_folder, filename)
            # 现在将使用更新后的 MD5 列表进行检查
            if not _check_md5(filepath, md5):
                print(f"文件 {filename} 未通过完整性检查。")
                all_files_ok = False
                break

        return all_files_ok

    def _download(self):
        """下载所需文件"""
        if self._check_integrity():
            return

        resources_to_download = self.resources[:2] if self.train else self.resources[2:]

        for filename, md5 in resources_to_download:
            filepath = os.path.join(self.raw_folder, filename)
            url = self.base_url + filename

            # 检查本地文件是否 "损坏" (MD5不匹配)
            if os.path.exists(filepath) and not _check_md5(filepath, md5):
                print(f"文件 {filename} MD5不匹配，正在删除...")
                os.remove(filepath)

            if not os.path.exists(filepath):
                try:
                    urllib.request.urlretrieve(url, filepath)
                except URLError as e:
                    print(f"下载失败: {e}")
                    if os.path.exists(filepath):
                        os.remove(filepath)
                    raise RuntimeError(f"无法下载 {filename}")

            # 最终校验
            if not _check_md5(filepath, md5):
                raise RuntimeError(f"下载的文件 {filename} MD5 校验失败。")

    def __getitem__(self, item):
        img = self.data[item]
        label = self.labels[item]
        return img, label

    def __len__(self):
        return self.data.shape[0]


class AutoDriveDataset(Dataset):
    """
    适配 MyTorch 框架的自动驾驶数据集加载器
    """

    def __init__(self, mode, transform=None, data_root="./"):
        """
        :参数 mode: 'train' 或者 'val'
        :参数 transform: 图像预处理函数
        :参数 data_root: 数据集根目录，用于拼接完整路径
        """
        self.mode = mode.lower()
        self.transform = transform
        self.data_root = data_root

        assert self.mode in {"train", "val"}

        # 读取数据集列表文件信息
        if self.mode == "train":
            file_path = os.path.join(data_root, "train.txt")
        else:
            file_path = os.path.join(data_root, "val.txt")

        self.file_list = list()

        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到数据列表文件: {file_path}")

        with open(file_path, "r") as f:
            files = f.readlines()
            for file in files:
                line = file.strip()
                if not line:
                    continue
                # 解析每一行: 图片路径 转向角
                parts = line.split(" ")
                img_path = parts[0]
                steering = float(parts[1])

                # 如果有油门值(throttle)，通常是第三列，可以按需添加
                # throttle = float(parts[2]) if len(parts) > 2 else 0.0

                self.file_list.append([img_path, steering])

    def __getitem__(self, i):
        """
        :参数 i: 图像检索号
        :返回: (图像Tensor, 标签Tensor)
        """
        # 1. 解析路径
        img_rel_path = self.file_list[i][0]
        # 拼接完整路径 (train.txt 里是相对路径)
        full_img_path = os.path.join(self.data_root, img_rel_path)

        # 2. 读取图像
        img = cv2.imread(full_img_path)
        if img is None:
            raise ValueError(f"无法读取图像: {full_img_path}")

        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 3. 预处理
        # 如果传入了外部 transform 函数则使用它，否则使用默认处理
        if self.transform:
            img = self.transform(img)
        else:
            # --- 关键适配 MyTorch ---
            # OpenCV 读取的是 (H, W, C)，但在 mytorch.Conv2d 中我们需要 (C, H, W)
            # 同时也需要将像素值从 0-255 归一化到 0-1

            # Resize (可选，根据模型输入调整，LeNet通常比较小，DonkeyCar常用 (120, 160))
            # img = cv2.resize(img, (160, 120))

            # HWC -> CHW 并归一化
            img = img.transpose(2, 0, 1).astype(np.float32) / 255.0

        # 4. 处理标签 (转向值)
        label_val = self.file_list[i][1]

        # --- 修改 2: 包装为 MyTorch Tensor ---
        # 图像 Tensor
        img_tensor = Tensor(img)

        # 标签 Tensor (形状为 (1,) 的向量，方便计算 MSELoss)
        label_tensor = Tensor(np.array([label_val], dtype=np.float32))

        return img_tensor, label_tensor

    def __len__(self):
        """返回: 图像总数"""
        return len(self.file_list)
