"""确定性截图标注：模型提供位置，Pillow 只负责绘制，不重绘页面内容。"""

from io import BytesIO
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def prepare_screenshot(image_bytes):
    """模型识别和本地绘制共用同一张规范化图片，避免缩放/方向造成偏移。"""
    if not image_bytes or len(image_bytes) > 15 * 1024 * 1024:
        raise ValueError("请上传小于 15 MB 的截图。")
    with Image.open(BytesIO(image_bytes)) as source:
        if source.width * source.height > 20_000_000:
            raise ValueError("截图尺寸过大，请裁剪后重新上传。")
        normalized = ImageOps.exif_transpose(source).convert("RGB")
        normalized.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
        output = BytesIO()
        normalized.save(output, format="PNG")
        return output.getvalue()


def validate_bbox(bbox):
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError("目标坐标必须包含四个数字。")
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in bbox):
        raise ValueError("目标坐标必须是有限数字。")
    left, top, right, bottom = bbox
    if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
        raise ValueError("目标坐标超出截图范围或顺序错误。")
    if (right - left) * (bottom - top) > 0.5:
        raise ValueError("目标范围过大，无法精确指出按钮。")
    return tuple(bbox)


def _font(size):
    for path in (
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ):
        if Path(path).is_file():
            try:
                return ImageFont.truetype(path, size), True
            except OSError:
                continue
    # 无中文字体时图片内显示英文，界面 caption 仍完整显示中文。
    return ImageFont.load_default(), False


def annotate_screenshot(image_bytes, bbox):
    """返回带红框、箭头和顶部提示条的 PNG 字节。输入应为 prepare_screenshot 的结果。"""
    left, top, right, bottom = validate_bbox(bbox)
    with Image.open(BytesIO(image_bytes)) as source:
        screenshot = source.convert("RGB")
    width, height = screenshot.size
    if width < 80 or height < 80:
        raise ValueError("截图太小，请上传完整手机截图。")
    line_width = max(3, round(width / 180))
    font_size = max(14, min(36, round(width / 24)))
    banner_height = max(44, font_size * 2)
    canvas = Image.new("RGB", (width, height + banner_height), "#fff4f4")
    canvas.paste(screenshot, (0, banner_height))
    draw = ImageDraw.Draw(canvas)
    font, has_chinese = _font(font_size)
    label = "请点击这里 ↓" if has_chinese else "Click the red box"
    draw.text((max(4, width // 40), banner_height // 5), label, fill="#d71920", font=font)

    x0 = round(left * (width - 1))
    x1 = round(right * (width - 1))
    y0 = banner_height + round(top * (height - 1))
    y1 = banner_height + round(bottom * (height - 1))
    if x1 - x0 < 4 or y1 - y0 < 4:
        raise ValueError("目标太小，无法可靠标注。")
    draw.rectangle((x0, y0, x1, y1), outline="#e02020", width=line_width)

    # 从目标外侧留白最多的方向画箭头，尖端指向红框边缘，不遮住按钮文字。
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    candidates = [
        (y0 - banner_height, (cx, y0), (0, -1)),
        (height + banner_height - 1 - y1, (cx, y1), (0, 1)),
        (x0, (x0, cy), (-1, 0)),
        (width - 1 - x1, (x1, cy), (1, 0)),
    ]
    space, tip, direction = max(candidates, key=lambda item: item[0])
    length = min(max(28, width * 0.16), space * 0.85)
    if length >= 12:
        dx, dy = direction
        tail = (tip[0] + dx * length, tip[1] + dy * length)
        draw.line((tail, tip), fill="#e02020", width=line_width)
        head = min(length * 0.4, max(10, line_width * 3))
        base = (tip[0] + dx * head, tip[1] + dy * head)
        draw.polygon([
            tip,
            (base[0] - dy * head / 2, base[1] + dx * head / 2),
            (base[0] + dy * head / 2, base[1] - dx * head / 2),
        ], fill="#e02020")
    output = BytesIO()
    canvas.save(output, format="PNG")
    return output.getvalue()
