"""适配新版 RapidOCR：根据文字真实位置定位目标。"""

from functools import lru_cache
from io import BytesIO
import math
import re
from threading import RLock
import unicodedata

from PIL import Image

from services.image_annotation import validate_bbox


_OCR_LOCK = RLock()


class OCRUnavailable(RuntimeError):
    """OCR 初始化或识别失败。"""


@lru_cache(maxsize=1)
def get_ocr_engine():
    try:
        from rapidocr import RapidOCR

    except ImportError as exc:
        raise OCRUnavailable(
            f"OCR 依赖加载失败：{exc}"
        ) from exc

    try:
        return RapidOCR()

    except Exception as exc:
        raise OCRUnavailable(
            f"OCR 初始化失败：{exc}"
        ) from exc


def read_screen_text(image_bytes):
    """
    识别截图中的文字和位置。

    新版返回对象：
        result.boxes
        result.txts
        result.scores

    不再使用旧版的 result, elapsed 解包方式。
    """
    with Image.open(BytesIO(image_bytes)) as image:
        width, height = image.size

    try:
        with _OCR_LOCK:
            engine = get_ocr_engine()

            result = engine(
                image_bytes,
                use_det=True,
                use_cls=True,
                use_rec=True,
                text_score=0.80,
            )

    except OCRUnavailable:
        raise

    except Exception as exc:
        raise OCRUnavailable(
            "文字识别失败，请上传清晰的原始截图重试。"
        ) from exc

    boxes = getattr(result, "boxes", None)
    labels = getattr(result, "txts", None)
    scores = getattr(result, "scores", None)

    # 没有识别到文字时，新版可能返回字段为 None 的对象。
    if boxes is None or labels is None or scores is None:
        return []

    entries = []

    for points, label, score in zip(boxes, labels, scores):
        try:
            score = float(score)

            if (
                not isinstance(label, str)
                or not label.strip()
                or not math.isfinite(score)
                or not 0.80 <= score <= 1
            ):
                continue

            xs = [float(point[0]) for point in points]
            ys = [float(point[1]) for point in points]

            if len(xs) != 4:
                continue

            if not all(math.isfinite(value) for value in xs + ys):
                continue

            bbox = validate_bbox([
                max(0, min(xs) / width),
                max(0, min(ys) / height),
                min(1, max(xs) / width),
                min(1, max(ys) / height),
            ])

            entries.append({
                "label": label.strip(),
                "bbox": bbox,
                "score": score,
            })

        except (TypeError, ValueError, IndexError):
            continue

    return entries


def normalize_label(text):
    """统一空格、大小写及明确的名称别名。"""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", "", text).casefold()

    aliases = {
        "壁纸": "墙纸",
        "wallpaper": "墙纸",
        "bluetooth": "蓝牙",
    }

    return aliases.get(text, text)


def locate_target(label, entries, image_size):
    """精确查找唯一的文字位置，不猜测或模糊匹配。"""
    if not isinstance(label, str) or not label.strip():
        raise ValueError("缺少目标名称。")

    target_name = normalize_label(label)

    matches = [
        entry
        for entry in entries
        if normalize_label(entry["label"]) == target_name
    ]

    if len(matches) != 1:
        raise ValueError("没有找到唯一匹配的目标文字。")

    entry = matches[0]

    left, top, right, bottom = validate_bbox(entry["bbox"])

    width, height = image_size
    text_height = (bottom - top) * height

    # 紧贴目标文字，避免扩大到相邻菜单。
    padding_x = max(2, text_height * 0.30) / width
    padding_y = max(1, text_height * 0.12) / height

    bbox = validate_bbox([
        max(0, left - padding_x),
        max(0, top - padding_y),
        min(1, right + padding_x),
        min(1, bottom + padding_y),
    ])

    return {
        "label": entry["label"],
        "bbox": bbox,
    }
