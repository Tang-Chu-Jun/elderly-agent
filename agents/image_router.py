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
    Streamlit 上传图片转 Base64
    """

    image_bytes = uploaded_file.getvalue()

    return base64.b64encode(
        image_bytes
    ).decode("utf-8")


def route_image_task(
    uploaded_file,
    user_text: str = ""
) -> str:
    """
    图文联合任务路由。

    返回：
    fraud
    screenshot
    planner
    general
    """

    image_base64 = image_to_base64(
        uploaded_file
    )

    file_type = (
        uploaded_file.type
        or "image/png"
    )

    user_text = (
        user_text.strip()
        if user_text
        else ""
    )

    prompt = f"""
你是“银龄智办”的多模态任务路由智能体。

你的任务不是简单判断截图里面出现了什么，
而是结合：

1. 用户输入的文字
2. 用户上传的截图

判断：

【用户现在真正想让系统帮他完成什么事情】

用户文字：

{user_text if user_text else "用户没有输入文字说明"}

--------------------------------------------------

fraud：诈骗与风险识别

只有当用户的主要目标是：

- 判断短信、聊天、链接是否可信
- 判断是不是诈骗
- 判断转账、中奖、退款、客服消息是否有风险
- 处理验证码、银行卡、陌生链接等风险信息

或者：

截图本身存在非常明显的诈骗风险，
例如要求转账、验证码、银行卡、
陌生链接、中奖领取等。

才返回：

fraud


--------------------------------------------------

screenshot：手机截图操作指引

当用户的主要目标是：

- 不知道当前页面应该点哪里
- 不知道下一步怎么操作
- 想打开或关闭某个手机功能
- 想设置 WiFi、蓝牙、字体、亮度等
- 想在某个 App 页面完成具体点击操作

返回：

screenshot

特别注意：

截图里出现“医院”“挂号”“微信”“支付宝”等内容，
并不代表一定属于 planner。

例如：

用户上传医院 App 截图并问：

“我应该点哪里挂号？”

必须返回：

screenshot


--------------------------------------------------

planner：数字生活事务规划

只有当用户希望系统帮助规划或者完成完整事务时返回 planner。

例如：

- 帮我安排明天下午去医院看病
- 帮我看看应该挂什么科
- 帮我规划怎么去医院
- 帮我买菜
- 帮我安排购物
- 帮我规划公交、地铁路线

重点：

planner 是“完成一件生活事务”。

screenshot 是“告诉我当前这个页面怎么操作”。


--------------------------------------------------

general：

明显不属于以上三类。


--------------------------------------------------

判断优先级：

第一优先：
理解用户文字表达的真实目的。

第二优先：
结合截图判断当前场景。

第三优先：
如果用户没有输入文字：

1. 明显诈骗短信/聊天 → fraud
2. 普通手机/App页面 → screenshot
3. 其他情况 → general

只有截图里出现医院、购物、公交等元素，
不能单独作为 planner 的判断依据。


--------------------------------------------------

只能输出下面四个单词之一：

fraud
screenshot
planner
general

不要解释。
不要输出标点。
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                f"data:{file_type};"
                                f"base64,{image_base64}"
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
        .strip()
        .lower()
    )

    valid_types = {
        "fraud",
        "screenshot",
        "planner",
        "general"
    }

    if result in valid_types:
        return result

    return "general"