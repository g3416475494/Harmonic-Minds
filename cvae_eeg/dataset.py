import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from . import config


def find_segment_files(segment_name, data_dir=None):
    """自动查找 {segment}-1.csv ~ {segment}-3.csv"""
    data_dir = data_dir or config.DATA_DIR
    files = []
    for i in [1, 2, 3]:
        path = os.path.join(data_dir, f"{segment_name}-{i}.csv")
        if os.path.exists(path):
            files.append(path)
    if not files:
        raise FileNotFoundError(f"在 {data_dir}/ 下找不到 {segment_name}-{{1,2,3}}.csv")
    return files


class EEGCSVDataset(Dataset):
    def __init__(self, csv_paths, segment_label):
        all_windows = []
        all_raw = []
        for path in csv_paths:
            df = pd.read_csv(path)
            data = df[config.FEATURES].values.astype(np.float32)  # (T, 5)
            all_raw.append(data)

        # 全局标准化参数（用于推理反变换）
        all_raw = np.concatenate(all_raw, axis=0)
        self.mean = all_raw.mean(axis=0)      # (5,)
        self.std = all_raw.std(axis=0) + 1e-8  # (5,)

        for path in csv_paths:
            df = pd.read_csv(path)
            data = df[config.FEATURES].values.astype(np.float32)
            data = (data - self.mean) / self.std
            data = data.T   # (5, T)

            T = data.shape[1]
            for s in range(0, T - config.WIN_LEN + 1, config.STEP):
                all_windows.append(data[:, s:s + config.WIN_LEN])

        self.windows = np.stack(all_windows)
        self.label = segment_label
        print(f"  加载 {len(csv_paths)} 个文件: {self.windows.shape[0]} 个窗口")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.windows[idx])
        c = torch.tensor(self.label, dtype=torch.long)
        return x, c
