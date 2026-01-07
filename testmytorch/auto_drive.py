import cv2
import numpy as np
import gym
import gym_donkeycar
import pickle
import os


from mytorch.tensor import Tensor
from model.autodrive_net import AutoDriveNet


def load_weights(model, path):
    """
    加载权重的辅助函数
    对应 train_donkey.py 中的 save_model
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到模型权重文件: {path}")

    print(f"正在加载模型权重: {path}")
    with open(path, 'rb') as f:
        # 读取保存的参数列表 (numpy 或 cupy 数组列表)
        saved_params = pickle.load(f)

    # 获取当前模型的参数列表
    model_params = model.parameters()

    # 检查数量是否一致
    if len(saved_params) != len(model_params):
        raise RuntimeError("加载失败：模型结构与保存的权重参数数量不匹配！")

    # 逐个赋值
    for param, saved_data in zip(model_params, saved_params):
        # 如果当前环境是 CPU 但权重是 GPU (cupy) 保存的，需要转换，反之亦然
        # 这里简单起见，直接赋值 data，Tensor 类会自动处理类型
        # 但为了安全，我们确保 data 被正确覆盖
        param.data = saved_data

    print("权重加载完成！")


# =============================================================
# 主程序
# =============================================================

# 1. 设置模拟器环境
# 确保你安装了 gym-donkeycar
# pip install gym-donkeycar
env = gym.make("donkey-generated-roads-v0")

# 重置当前场景
obv = env.reset()

# 2. 初始化模型
model = AutoDriveNet()

# 3. 加载权重
# 请确保路径和文件名与训练时保存的一致 (例如 best_model.pkl)
load_weights(model, "./results/best_model.pkl")


# 切换到评估模式 (关闭 Dropout)
model.eval()

# 5. 开始启动
action = np.array([0, 0.1])  # 初始动作：[转向, 油门]

# 执行第一步获取初始图像
frame, reward, done, info = env.step(action)

print("开始自动驾驶...")

try:
    # 运行 2500 次动作 (或者直到 done)
    for t in range(2500):
        # --- 图像预处理 (NumPy 实现) ---
        # 1. 拷贝并转为 float32
        img_arr = frame.copy().astype(np.float32)

        # 2. 归一化 (0~255 -> 0~1)
        img_arr /= 255.0

        # 3. 调整通道 (H, W, C) -> (C, H, W)
        img_arr = img_arr.transpose(2, 0, 1)

        # 4. 增加 Batch 维度 (C, H, W) -> (1, C, H, W)
        img_arr = np.expand_dims(img_arr, axis=0)

        # 5. 转为 mytorch Tensor
        img_tensor = Tensor(img_arr)

        # 6. 如果模型在 GPU，数据也要搬到 GPU

        # --- 模型推理 ---
        # 不需要 with no_grad()，只要不调用 backward 就行
        pred_tensor = model(img_tensor)

        # --- 获取结果 ---
        # 1. 取出数据 (.data 可能在 GPU)
        # 2. 确保转回 CPU numpy
        # 3. 提取标量值

        pred_val = pred_tensor.data

        # 假设输出 shape 是 (1, 1)，取出具体的 float 值
        steering_angle = float(pred_val.flatten()[0])

        # --- 执行动作 ---
        factor = 1.0  # 转向灵敏度，根据实际情况调整
        throttle = 0.2  # 恒定油门，或者你可以训练模型输出油门

        action = np.array([steering_angle * factor, throttle])

        # 发送给环境
        frame, reward, done, info = env.step(action)

        if done:
            print(f"Episode finished after {t + 1} timesteps")
            obv = env.reset()
            break

except KeyboardInterrupt:
    print("用户手动停止")

finally:
    # 运行完以后重置当前场景并关闭
    obv = env.reset()
    env.close()
    print("模拟器连接已断开")