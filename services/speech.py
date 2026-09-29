import os
import tempfile

import streamlit as st
from faster_whisper import WhisperModel

from services.llm import chat_with_llm


# =========================
# Whisper 配置
# =========================

# 中文识别不要再用 base
WHISPER_MODEL_SIZE = "small"


# =========================
# 懒加载模型
# =========================

@st.cache_resource
def get_whisper_model():
    """
    第一次真正识别语音时才加载。
    Streamlit 后续 rerun 直接复用。
    """

    return WhisperModel(
        WHISPER_MODEL_SIZE,
        device="cpu",
        compute_type="int8"
    )


# =========================
# 保存临时音频
# =========================

def save_audio_temp(audio_file) -> str:

    if audio_file is None:
        return ""

    audio_bytes = audio_file.getvalue()

    if not audio_bytes:
        return ""

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    )

    temp_file.write(audio_bytes)
    temp_file.close()

    return temp_file.name


# =========================
# Whisper 原始识别
# =========================

def whisper_transcribe(audio_path: str) -> str:

    model = get_whisper_model()

    # 给 Whisper 一些项目领域词汇
    initial_prompt = """
这是一个面向老年人的数字生活智能助手。

常见内容包括：
医院、挂号、看病、膝盖、骨科、医保、
诈骗、短信、银行卡、验证码、转账、
微信、手机、设置、蓝牙、WiFi、字体、
购物、买菜、公交、地铁、出行、路线。

请准确识别普通话。
"""

    segments, info = model.transcribe(
        audio_path,

        language="zh",

        vad_filter=True,

        beam_size=5,

        temperature=0.0,

        initial_prompt=initial_prompt,

        condition_on_previous_text=False
    )

    texts = []

    for segment in segments:

        text = segment.text.strip()

        if text:
            texts.append(text)

    return "".join(texts).strip()


# =========================
# LLM 轻量纠错
# =========================

def correct_transcript(
    raw_text: str
) -> str:
    """
    只修正明显的语音识别错误。
    不允许模型凭空添加用户没有说过的信息。
    """

    if not raw_text:
        return ""

    prompt = f"""
你正在处理一个老年人数字生活助手的语音识别结果。

下面是语音识别模型输出的原始文本：

【原始识别】
{raw_text}

请只修正非常明显的：

- 同音字错误
- 错别字
- 断句错误
- 明显不通顺的语音识别错误

常见场景包括：

医院、挂号、医保、骨科、膝盖、
诈骗、验证码、银行卡、转账、
手机设置、蓝牙、WiFi、
购物、买菜、公交、地铁和出行。

非常重要：

1. 不要改变用户原意。
2. 不要添加原文没有的信息。
3. 如果无法确定某个词，保留原文。
4. 只输出修正后的句子。
5. 不要解释。

原始文本：

{raw_text}
"""

    corrected = chat_with_llm(
        prompt
    )

    if not corrected:
        return raw_text

    # 避免LLM接口异常文本污染结果
    if corrected.startswith("模型调用失败"):
        return raw_text

    return corrected.strip()


# =========================
# 对 app.py 暴露
# =========================

def transcribe_audio(audio_file) -> str:
    """
    完整语音识别流程：

    音频
    → Whisper
    → LLM轻量纠错
    → 最终文字
    """

    if audio_file is None:
        return ""

    audio_path = save_audio_temp(
        audio_file
    )

    if not audio_path:
        return ""

    try:

        # Whisper
        raw_text = whisper_transcribe(
            audio_path
        )

        if not raw_text:
            return ""

        # 大模型纠错
        final_text = correct_transcript(
            raw_text
        )

        return final_text

    finally:

        try:

            if (
                audio_path
                and os.path.exists(audio_path)
            ):
                os.remove(audio_path)

        except Exception:
            pass