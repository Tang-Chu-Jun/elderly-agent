from services.llm import chat_with_llm


def rule_route(text: str):
    """
    规则优先判断。
    如果无法明确判断，返回 None。
    """

    if not text:
        return "general"

    text = text.lower().strip()

    fraud_keywords = [
        "诈骗", "骗子", "骗钱", "被骗", "可疑",
        "转账", "汇款", "打钱", "垫钱", "垫付",
        "借钱", "付款", "银行卡", "验证码",
        "支付密码", "银行卡号", "身份证",
        "陌生链接", "二维码", "扫码",
        "下载软件", "共享屏幕",
        "中奖", "刷单", "返利", "退款",
        "投资", "理财", "高收益",
        "账户冻结", "医保异常", "公检法",
        "冒充", "侄子", "侄女", "孙子", "孙女",
        "儿子", "女儿", "客服",
        "警察", "公安", "法院", "检察院"
    ]

    screenshot_keywords = [
        "怎么点", "点哪里", "点哪个",
        "下一步", "这个页面", "这个界面",
        "截图", "怎么设置", "在哪里设置",
        "找不到", "怎么打开", "怎么关闭",
        "wifi", "wi-fi", "无线网",
        "蓝牙", "字体", "亮度",
        "声音", "音量"
    ]

    planner_keywords = [
        "医院", "挂号", "就医", "医生",
        "科室", "预约医生", "看病", "复诊",
        "买菜", "购物", "买东西", "商品",
        "公交", "地铁", "路线",
        "怎么去", "怎么走", "出行",
        "坐车", "打车", "到达"
    ]

    if any(word in text for word in fraud_keywords):
        return "fraud"

    if any(word in text for word in screenshot_keywords):
        return "screenshot"

    if any(word in text for word in planner_keywords):
        return "planner"

    return None


def ai_route(text: str) -> str:
    """
    使用大模型进行意图分类。
    """

    prompt = f"""
你是“银龄智办”的任务路由器。

请判断下面用户需求属于哪一类：

fraud：
诈骗识别、风险信息判断、陌生短信、
陌生聊天、转账风险、验证码风险、
冒充亲友、客服、公检法等。

screenshot：
手机操作指导、看不懂页面、
不知道点哪里、设置操作、
需要根据截图一步一步指导。

planner：
数字生活事务规划，包括：
医疗、挂号、就医、
购物、买菜、
公交、地铁、路线、出行。

general：
不属于以上三类的普通问题。

用户输入：
{text}

只能返回下面四个单词之一：

fraud
screenshot
planner
general
"""

    result = chat_with_llm(prompt).strip().lower()

    valid_types = {
        "fraud",
        "screenshot",
        "planner",
        "general"
    }

    if result in valid_types:
        return result

    return "general"


def route_task(text: str) -> str:
    """
    总路由：
    规则优先，AI兜底
    """

    result = rule_route(text)

    if result is not None:
        return result

    return ai_route(text)