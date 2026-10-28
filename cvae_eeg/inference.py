import os
import argparse
import numpy as np
import pandas as pd
import torch
from . import config
from .model import CVAE


def load_model(segment_name, device=None):
    """加载训练好的模型checkpoint"""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"cvae_{segment_name}.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到模型: {ckpt_path}，请先训练")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = CVAE().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"加载模型: {ckpt_path}")
    return model, device, ckpt


def generate(segment_name, duration_sec, output_path=None):
    """
    生成指定时长的频段EEG数据
    参数:
        segment_name: 段落名 (prelude/lyrical/playful/intense/meditative)
        duration_sec: 时长（秒）
        output_path: 输出CSV路径
    返回:
        DataFrame
    """
    model, device, ckpt = load_model(segment_name)
    seg_label = config.SEGMENT_MAP[segment_name]
    mean = ckpt["mean"]   # (5,)
    std = ckpt["std"]     # (5,)

    # 计算需要生成多少个时间步（每行0.5秒）
    n_steps = int(duration_sec / 0.5)
    # 每个窗口4步(2秒)，有重叠，需要 ceil(n_steps / WIN_LEN) + 1 个窗口
    n_windows = int(np.ceil(n_steps / config.WIN_LEN)) + 1

    # 从标准正态采样z，用Decoder生成
    all_windows = []
    with torch.no_grad():
        for _ in range(n_windows):
            z = torch.randn(1, config.Z_DIM, device=device)
            c = torch.tensor([seg_label], dtype=torch.long, device=device)
            x_hat = model.decoder(z, c)        # (1, 5, 4)
            all_windows.append(x_hat.squeeze(0).cpu().numpy())

    # 拼接窗口（重叠区域取平均）
    # 简单做法：直接首尾拼接，去掉重叠
    generated = np.concatenate([w for w in all_windows], axis=1)  # (5, T)
    generated = generated[:, :n_steps]   # 截断到目标长度

    # 反标准化：用训练时保存的mean/std精确还原
    # generated: (5, T)，mean/std: (5,)
    mean = mean[:, np.newaxis]   # (5, 1)
    std = std[:, np.newaxis]     # (5, 1)
    generated = generated * std + mean
    generated = np.clip(generated, 0.01, None)   # 频段功率不能为负

    # 构建CSV
    time_s = np.arange(n_steps) * 0.5 + 2.0   # 从2.0秒开始，和原始数据对齐
    timestamps = pd.date_range(start="2026-10-14 13:36:30",
                              periods=n_steps, freq="500ms")

    df = pd.DataFrame({
        "timestamp": timestamps.strftime("%Y-%m-%d %H:%M:%S"),
        "time_s": np.round(time_s, 2),
        "Theta":   np.round(generated[0], 2),
        "Alpha":   np.round(generated[1], 2),
        "LowBeta": np.round(generated[2], 2),
        "HighBeta": np.round(generated[3], 2),
        "Gamma":   np.round(generated[4], 2),
    })

    if output_path is None:
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(config.OUTPUT_DIR, f"generated_{segment_name}.csv")
    df.to_csv(output_path, index=False)
    print(f"已生成: {output_path} ({len(df)}行, {duration_sec}秒)")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--segment", required=True, choices=list(config.SEGMENT_MAP.keys()))
    parser.add_argument("--duration", type=float, required=True, help="生成时长（秒）")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    generate(args.segment, args.duration, args.output)
