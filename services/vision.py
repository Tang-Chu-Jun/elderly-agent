import base64

from openai import OpenAI

from config import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    MODEL_NAME
)


client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)


def image_to_base64(uploaded_file) -> str:
    """
    将 Streamlit 上传的图片转换为 Base64
    """

    image_bytes = uploaded_file.getvalue()

    image_base64 = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    return image_base64


def analyze_fraud_image(uploaded_file) -> str:
    """
    读取短信截图、微信聊天截图等图片内容，
    并整理出与诈骗风险有关的信息。
    """

    # 图片转 base64
    image_base64 = image_to_base64(
        uploaded_file
    )

    # 获取图片 MIME 类型
    file_type = uploaded_file.type

    if not file_type:
        file_type = "image/png"

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是“银龄智办”的截图内容识别模块。"
                    "你的任务是读取短信截图、微信聊天截图、"
                    "网页截图等图片中的信息。"
                    "请重点提取与诈骗风险有关的内容，"
                    "包括人物身份、金额、链接、银行卡、"
                    "验证码、转账要求、退款要求、"
                    "中奖信息、账户异常、紧急要求等。"
                    "这一阶段只负责理解图片内容，"
                    "不要直接判断是不是诈骗。"
                )
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "请仔细阅读这张截图。"
                            "提取主要文字内容，"
                            "并整理其中可能与风险判断有关的信息。"
                            "不要遗漏金额、链接、验证码、"
                            "银行卡、转账等重要内容。"
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                f"data:{file_type};base64,"
                                f"{image_base64}"
                            )
                        }
                    }
                ]
            }
        ]
    )

    result = (
        response
        .choices[0]
        .message
        .content
    )

    if result is None:
        return "未能识别截图内容。"

    return result.strip()