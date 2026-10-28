import os
import torch
from torch.utils.data import DataLoader
from . import config
from .dataset import EEGCSVDataset, find_segment_files
from .model import CVAE, loss_fn


def train_segment(segment_name, data_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seg_label = config.SEGMENT_MAP[segment_name]

    print(f"\n=== 训练段落: {segment_name} (label={seg_label}) ===")
    print(f"设备: {device}")

    csv_files = find_segment_files(segment_name, data_dir)
    print(f"数据文件: {[os.path.basename(f) for f in csv_files]}")

    ds = EEGCSVDataset(csv_files, seg_label)
    dl = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)

    model = CVAE().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.LR,
                            weight_decay=config.WEIGHT_DECAY)

    for epoch in range(config.EPOCHS):
        model.train()
        beta = min(config.BETA_START,
                   config.BETA_START * (epoch + 1) / 100)

        total_loss = 0
        for x, c in dl:
            x, c = x.to(device), c.to(device)
            x_hat, mu, logvar = model(x, c)
            loss, _, _ = loss_fn(x, x_hat, mu, logvar, beta)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        if (epoch + 1) % 30 == 0:
            avg = total_loss / len(dl)
            print(f"Epoch {epoch+1:3d}/{config.EPOCHS} | "
                  f"Loss={avg:.2f} | β={beta:.4f}")

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    save_path = os.path.join(config.CHECKPOINT_DIR, f"cvae_{segment_name}.pth")
    torch.save({
        "model_state": model.state_dict(),
        "segment": segment_name,
        "label": seg_label,
        "z_dim": config.Z_DIM,
        "n_features": config.N_CH,
        "win_len": config.WIN_LEN,
        "mean": ds.mean,
        "std": ds.std,
    }, save_path)
    print(f"模型已保存: {save_path}")
    return save_path
