import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cvae_eeg import config
from cvae_eeg.train import train_segment


def main():
    parser = argparse.ArgumentParser(description="CVAE EEG信号生成")
    # 模式
    parser.add_argument("--infer", action="store_true", help="推理模式（生成数据）")
    parser.add_argument("--all", action="store_true", help="训练全部段落")
    # 段落
    parser.add_argument("--segment", type=str,
                        choices=list(config.SEGMENT_MAP.keys()),
                        help="段落名")
    # 推理参数
    parser.add_argument("--duration", type=float, help="生成时长（秒）")
    parser.add_argument("--output", type=str, default=None, help="输出CSV路径")
    # 其他
    parser.add_argument("--data-dir", type=str, default=None)
    args = parser.parse_args()

    # ---- 推理模式 ----
    if args.infer:
        if not args.segment or not args.duration:
            parser.error("推理需要 --segment 和 --duration")
        from cvae_eeg.inference import generate
        generate(args.segment, args.duration, args.output)
        return

    # ---- 训练模式 ----
    if args.all:
        for seg in config.SEGMENT_MAP:
            train_segment(seg, args.data_dir)
    elif args.segment:
        train_segment(args.segment, args.data_dir)
    else:
        parser.print_help()
        print("\n示例:")
        print("  训练: python run.py --segment playful")
        print("  全部: python run.py --all")
        print("  推理: python run.py --infer --segment playful --duration 60")


if __name__ == "__main__":
    main()
