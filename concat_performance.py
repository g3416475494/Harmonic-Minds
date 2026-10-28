import argparse
import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GEN_DIR = os.path.join(BASE_DIR, "generated_csv")
TIMELINE_PATH = os.path.join(BASE_DIR, "timeline.txt")
OUTPUT_PATH = os.path.join(GEN_DIR, "full_generated.csv")

FEATURES = ["Theta", "Alpha", "LowBeta", "HighBeta", "Gamma"]


def load_timeline(path):
    timeline = []
    end_time = None
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("END_TIME="):
                end_time = int(line.split("=")[1])
                continue
            parts = line.split(",")
            if len(parts) == 2:
                timeline.append((int(parts[0].strip()), parts[1].strip()))
    if end_time is None:
        raise ValueError("timeline.txt 缺少 END_TIME=")
    return timeline, end_time


def load_segment(segment_name):
    if segment_name == "blank":
        return None
    path = os.path.join(GEN_DIR, f"generated_{segment_name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到生成数据: {path}，请先运行推理")
    df = pd.read_csv(path)
    return df[FEATURES].values.astype(np.float32)


def make_blank(n_steps):
    """blank段全0"""
    return np.zeros((n_steps, len(FEATURES)), dtype=np.float32)


def slice_random(data, n_needed):
    if data is None:
        return make_blank(n_needed)
    total = len(data)
    if total <= n_needed:
        repeats = int(np.ceil(n_needed / total))
        data = np.tile(data, (repeats, 1))
        total = len(data)
    start = np.random.randint(0, total - n_needed + 1)
    return data[start:start + n_needed]


def smooth_transitions(full, timeline, transition_sec):
    if transition_sec <= 0:
        return full
    n_fade_total = int(transition_sec / 0.5)   # 总过渡步数
    n_each = n_fade_total // 2                   # 每侧步数

    # 切换点在full中的位置（步数）
    for i in range(1, len(timeline)):
        switch_step = int(timeline[i][0] / 0.5)
        if switch_step >= len(full):
            continue
        # 前一段尾部n_each步 与 后一段头部n_each步 混合
        s = max(0, switch_step - n_each)
        e = min(len(full), switch_step + n_each)
        seg_a = full[s:switch_step].copy()   # 前一段尾部
        seg_b = full[switch_step:e].copy()   # 后一段头部
        if len(seg_a) == 0 or len(seg_b) == 0:
            continue
        n = min(len(seg_a), len(seg_b))
        t = np.linspace(0, 1, n).reshape(-1, 1)
        blended = seg_a[-n:] * (1 - t) + seg_b[:n] * t
        full[switch_step - n:switch_step] = blended
    return full


def build_performance(prep_sec=3, transition_sec=2):
    timeline, end_time = load_timeline(TIMELINE_PATH)

    # 开头补足准备时间
    if timeline[0][1] == "blank":
        blank_end = timeline[1][0] if len(timeline) > 1 else end_time
        if blank_end < prep_sec:
            timeline = [(0, "blank")] + [(max(s, prep_sec), n) for s, n in timeline[1:]]

    segments_data = {}
    for _, name in timeline:
        if name not in segments_data:
            segments_data[name] = load_segment(name)

    # 按时间戳精确拼接，总长度 = END_TIME / 0.5
    parts = []
    for i, (start, name) in enumerate(timeline):
        end = timeline[i + 1][0] if i + 1 < len(timeline) else end_time
        n_steps = int((end - start) / 0.5)
        seg = slice_random(segments_data[name], n_steps)
        parts.append(seg)

    full = np.concatenate(parts, axis=0)

    # 在切换点做平滑过渡
    full = smooth_transitions(full, timeline, transition_sec)

    n = len(full)
    time_s = np.arange(n) * 0.5 + 2.0
    timestamps = pd.date_range(start="00:00:00", periods=n, freq="500ms")

    df = pd.DataFrame({
        "timestamp": timestamps.strftime("%Y-%m-%d %H:%M:%S"),
        "time_s": np.round(time_s, 2),
        "Theta":   np.round(full[:, 0], 2),
        "Alpha":   np.round(full[:, 1], 2),
        "LowBeta": np.round(full[:, 2], 2),
        "HighBeta": np.round(full[:, 3], 2),
        "Gamma":   np.round(full[:, 4], 2),
    })

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"准备时间: {prep_sec}秒, 过渡时间: {transition_sec}秒")
    print(f"时间轴: {len(timeline)}段")
    print(f"总时长: {n}行 ({n*0.5:.1f}秒)")
    print(f"已生成: {OUTPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prep", type=int, default=3, help="开头准备时间（秒）")
    parser.add_argument("--transition", type=int, default=2, help="状态切换过渡时间（秒）")
    args = parser.parse_args()
    build_performance(prep_sec=args.prep, transition_sec=args.transition)
