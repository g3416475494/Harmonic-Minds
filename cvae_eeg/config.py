import os

# 项目根目录（cvae_eeg/ 的上层）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据/输出目录（项目根目录下）
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "generated_csv")

# CSV列
FEATURES = ["Theta", "Alpha", "LowBeta", "HighBeta", "Gamma"]
N_CH = len(FEATURES)

# 窗口
WIN_LEN = 4        # 2秒窗口（每行0.5秒）
STEP = 2           # 步长1秒

# 模型
Z_DIM = 16
N_CLASSES = 5

# 训练
BATCH_SIZE = 16
EPOCHS = 500
LR = 1e-3
BETA_START = 0.01
WEIGHT_DECAY = 1e-5

# 段落映射
SEGMENT_MAP = {
    "lyrical":   1,
    "playful":   2,
    "intense":   3,
    "meditative": 4,
}
