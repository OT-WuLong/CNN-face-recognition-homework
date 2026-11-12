"""课程设计项目的命令行入口。"""

import argparse
from importlib.metadata import version


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="cnn-face",
        description="CNN 人脸识别课程设计：当前仅完成项目初始化。",
    )
    parser.add_argument("--version", action="version", version=version("cnn-face-course"))
    parser.parse_args()
    parser.print_help()


if __name__ == "__main__":
    main()
