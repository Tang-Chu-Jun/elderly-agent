"""回答朗读：生成 MP3，由手机或电脑浏览器播放。"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import html
import re


class TTSUnavailable(RuntimeError):
    """语音服务不可用。"""


def speech_text(markdown_text):
    """清理 Markdown 标记，保留适合朗读的文字。"""
    text = str(markdown_text or "")

    text = re.sub(
        r"```[\s\S]*?```",
        "代码内容请查看屏幕。",
        text,
    )

    text = re.sub(
        r"!\[[^\]]*\]\([^)]*\)",
        "",
        text,
    )

    text = re.sub(
        r"\[([^\]]+)\]\([^)]*\)",
        r"\1",
        text,
    )

    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"https?://\S+", "链接请查看屏幕", text)

    text = re.sub(
        r"(?m)^\s{0,3}(?:#{1,6}|>|[-*])\s+",
        "",
        text,
    )

    text = (
        text.replace("**", "")
        .replace("__", "")
        .replace("`", "")
    )

    # 去除表情符号，避免逐个朗读图标名称。
    text = re.sub(
        r"[\U0001F000-\U0001FAFF\u2600-\u27BF\u200d\ufe0f]",
        "",
        text,
    )

    return re.sub(r"\s+", " ", text).strip()


async def _collect_audio(text):
    try:
        import edge_tts
    except ImportError as exc:
        raise TTSUnavailable(
            "请先安装 edge-tts，文字回答仍可正常使用。"
        ) from exc

    communicate = edge_tts.Communicate(
        text=text,
        voice="zh-CN-XiaoxiaoNeural",
        rate="-10%",
        connect_timeout=10,
        receive_timeout=15,
    )

    audio = bytearray()

    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio.extend(chunk["data"])

    if not audio:
        raise TTSUnavailable(
            "没有生成语音，您可以先阅读文字回答。"
        )

    return bytes(audio)


async def _with_timeout(text):
    return await asyncio.wait_for(
        _collect_audio(text),
        timeout=40,
    )


def synthesize_speech(markdown_text):
    """同步调用入口：返回 MP3 字节，不创建临时文件。"""
    text = speech_text(markdown_text)

    if not text:
        return None

    if len(text) > 6000:
        raise TTSUnavailable(
            "这次回答过长，暂不朗读，请查看完整文字。"
        )

    try:
        # 独立事件循环，避免与其他异步组件发生冲突。
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                lambda: asyncio.run(_with_timeout(text))
            )
            return future.result()

    except TTSUnavailable:
        raise

    except Exception as exc:
        raise TTSUnavailable(
            "语音服务暂时不可用，文字回答已保留。"
        ) from exc