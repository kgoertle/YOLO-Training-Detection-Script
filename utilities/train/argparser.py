import argparse

def get_args():
    """
    Returns a namespace of parsed command-line arguments for YOLO training.
    """
    parser = argparse.ArgumentParser(description="YOLO Training Script")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true", help="Transfer-learning training")
    group.add_argument("--update", action="store_true", help="Update weights from latest best.pt")
    group.add_argument("--scratch", action="store_true", help="Train from scratch on dataset")
    parser.add_argument("--test", action="store_true", help="Debug mode for testing script")
    parser.add_argument("--resume", action="store_true", help="Resume from latest last.pt")
    parser.add_argument("--float32", action="store_true", help="Export TFLite model in float32 format")
    parser.add_argument("--float16", action="store_true", help="Export TFLite model in float16 format")

    args = parser.parse_args()
    mode = "train" if args.train else "update" if args.update else "scratch"

    return args, mode
