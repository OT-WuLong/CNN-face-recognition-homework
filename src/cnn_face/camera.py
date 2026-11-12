"""使用 OpenCV 读取视频源，通过 Tkinter 和 Pillow 展示逐帧识别。"""

import math
from pathlib import Path
import tkinter as tk
from tkinter import font as tkfont

import cv2
from PIL import Image, ImageDraw, ImageTk

from cnn_face.contracts import Gallery, validate_threshold


def run_stream(
    source: int | Path, gallery: Gallery, *, threshold: float
) -> None:
    """同步识别原尺寸 BGR 帧；调用方须事先完成模型初始化。

    关闭窗口、按 Esc 或 Q，以及视频正常结束时返回；运行故障抛出
    中文异常。视频有帧数元数据时检查提前中断；缺少元数据时，
    至少读到一帧后的 read=False 只能按结束处理。
    """
    if not (type(source) is int or isinstance(source, Path)):
        raise ValueError("视频源必须是摄像头整数索引或本地视频 Path。")
    if type(source) is int and source < 0:
        raise ValueError("摄像头索引不能为负数。")
    if isinstance(source, Path) and not source.is_file():
        raise ValueError(f"视频文件不存在：{source}")
    if not isinstance(gallery, Gallery):
        raise ValueError("需要有效的特征库。")
    validate_threshold(threshold)

    capture = None
    window = None
    is_camera = type(source) is int
    description = f"摄像头 {source}" if is_camera else f"视频 {source}"
    try:
        capture = cv2.VideoCapture(source if is_camera else str(source))
        if not capture.isOpened():
            raise RuntimeError(f"无法打开{description}，请检查输入或设备占用。")

        # 每次播放只解析一次真实识别接口，逐帧复用已初始化的引擎。
        try:
            from cnn_face.recognition import recognize
        except ImportError as error:
            raise RuntimeError(f"无法加载识别模块：{error}") from error

        frame_count = 0 if is_camera else capture.get(cv2.CAP_PROP_FRAME_COUNT)
        window = tk.Tk()
        window.title("CNN 人脸识别")
        families = {name.casefold(): name for name in tkfont.families(window)}
        font_family = next((
            families[name.casefold()]
            for name in (
                "SimSun", "宋体", "Microsoft YaHei", "微软雅黑",
                "Noto Sans CJK SC", "WenQuanYi Zen Hei", "PingFang SC",
            )
            if name.casefold() in families
        ), None)
        if font_family is None:
            raise RuntimeError("未找到支持中文的字体，请安装宋体或微软雅黑。")
        font = (font_family, 12)
        preview = tk.Label(window)
        preview.pack()
        status = tk.Label(window, text="待识别", font=font)
        status.pack()
        tk.Label(window, text="按 Esc 或 Q 退出", font=font).pack()

        stopped = False

        def stop(event=None) -> None:
            """只标记退出，统一在清理阶段关闭窗口与视频源。"""
            nonlocal stopped
            stopped = True

        window.protocol("WM_DELETE_WINDOW", stop)
        for key in ("<Escape>", "<KeyPress-q>", "<KeyPress-Q>"):
            window.bind(key, stop)

        frames_read = 0
        while not stopped:
            window.update_idletasks()
            window.update()
            if stopped:
                break
            ok, frame_bgr = capture.read()
            if not ok:
                if is_camera:
                    raise RuntimeError(f"{description}读取失败，请检查设备连接。")
                if (
                    frames_read == 0
                    or (
                        math.isfinite(frame_count)
                        and frames_read < frame_count
                    )
                ):
                    raise RuntimeError(f"{description}读取失败，未正常读到结尾。")
                break
            if frame_bgr is None or frame_bgr.size == 0:
                raise RuntimeError(f"{description}读取失败，返回空画面。")
            frames_read += 1

            # ponytail: 同步推理期间不能处理退出事件；延迟过高时再引入工作线程。
            try:
                result = recognize(frame_bgr, gallery, threshold=threshold)
            except Exception as error:
                raise RuntimeError(f"当前帧识别失败：{error}") from error

            messages = {
                "matched": f"匹配：{result.identity}",
                "unknown": "未知",
                "no_face": "未检测到人脸",
                "multiple_faces": "检测到多张人脸，暂不判定身份",
            }
            color = "green" if result.status == "matched" else "red"
            # 推理始终使用原始 BGR；仅对展示副本转换颜色并绘制原坐标框。
            image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            if result.bbox is not None:
                x, y, width, height = result.bbox
                right = min(x + width - 1, image.width - 1)
                bottom = min(y + height - 1, image.height - 1)
                if x <= right and y <= bottom:
                    ImageDraw.Draw(image).rectangle(
                        (x, y, right, bottom), outline=color, width=2,
                    )
            photo = ImageTk.PhotoImage(image, master=window)
            preview.configure(image=photo)
            preview.image = photo
            status.configure(text=messages[result.status], foreground=color)

    except cv2.error as error:
        raise RuntimeError(f"{description}读取或画面处理失败：{error}") from error
    except tk.TclError as error:
        raise RuntimeError(f"视频窗口运行失败：{error}") from error
    finally:
        try:
            if capture is not None:
                capture.release()
        finally:
            if window is not None:
                window.destroy()
