"""公共模型调用：区分正常回答、调用失败和输出格式错误。"""

import json

from openai import OpenAI

from config import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    MODEL_NAME,
)


class LLMError(RuntimeError):
    """模型调用失败。"""


class ModelOutputError(ValueError):
    """模型输出不符合要求。"""


client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    timeout=120.0,
    max_retries=0,
)


def chat_with_llm(
    message: str,
    *,
    strict: bool = False,
) -> str:
    """
    调用公共文字模型。

    strict=True：
        调用失败时抛出 LLMError，供 Agent 返回 success=False。

    strict=False：
        保持原有字符串接口，兼容现有普通聊天、Router、语音纠错调用。
        错误提示不包含密钥、请求地址或底层异常详情。
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是一名面向老年人的数字生活智能助手。"
                        "回答应简洁、清楚、耐心。"
                        "不能声称已经执行实际上没有执行的操作。"
                    ),
                },
                {
                    "role": "user",
                    "content": message,
                },
            ],
        )

        if not response.choices:
            raise LLMError("模型没有返回回答，请稍后重试。")

        choice = response.choices[0]
        content = choice.message.content

        if choice.finish_reason == "length":
            raise LLMError("模型回答未完整生成，请重试。")

        if not isinstance(content, str) or not content.strip():
            raise LLMError("模型没有返回有效内容，请稍后重试。")

        return content.strip()

    except Exception as exc:
        message = (
            str(exc)
            if isinstance(exc, LLMError)
            else "暂时无法连接智能服务，请稍后重试。"
        )

        if strict:
            raise LLMError(message) from None

        return f"模型调用失败：{message}"


def parse_json_object(raw_response: str) -> dict:
    """
    解析模型返回的 JSON 对象。

    接受普通 JSON 和 Markdown 代码块。
    拒绝数组、null、数字、字符串及错误提示。
    """
    if not isinstance(raw_response, str):
        raise ModelOutputError("模型返回内容不是文字。")

    text = raw_response.strip()

    if not text:
        raise ModelOutputError("模型返回内容为空。")

    if text.startswith("模型调用失败："):
        raise LLMError("本次模型调用失败，请重试。")

    if text.startswith("```"):
        lines = text.splitlines()

        if (
            len(lines) < 3
            or lines[0].strip().lower() not in {"```", "```json"}
            or lines[-1].strip() != "```"
        ):
            raise ModelOutputError("模型返回的代码块不完整。")

        text = "\n".join(lines[1:-1]).strip()

    def reject_nonstandard_number(value):
        raise ModelOutputError("JSON 包含无效数字。")

    try:
        data = json.loads(
            text,
            parse_constant=reject_nonstandard_number,
        )
    except json.JSONDecodeError as exc:
        raise ModelOutputError("模型返回的 JSON 格式不正确。") from exc

    if not isinstance(data, dict):
        raise ModelOutputError("模型必须返回 JSON 对象。")

    return data