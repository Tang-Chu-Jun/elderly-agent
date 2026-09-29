from datetime import datetime, timedelta

import streamlit as st


# =========================
# 配置
# =========================

MAX_HISTORY_ROUNDS = 10
TASK_TIMEOUT_MINUTES = 30


# =========================
# 初始化
# =========================

def init_memory():
    """
    初始化整个会话记忆。
    """

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "task_state" not in st.session_state:
        st.session_state.task_state = {
            "task_type": None,
            "sub_task": None,
            "goal": None,
            "current_step": None,
            "status": "idle",
            "data": {},
            "started_at": None,
            "updated_at": None
        }

    if "history_summary" not in st.session_state:
        st.session_state.history_summary = ""

    if "last_active_time" not in st.session_state:
        st.session_state.last_active_time = datetime.now()


# =========================
# 活跃时间
# =========================

def update_last_active_time():
    """
    更新最后操作时间。
    """

    st.session_state.last_active_time = datetime.now()


def is_task_expired() -> bool:
    """
    判断当前任务是否超过30分钟无操作。
    """

    last_time = st.session_state.get(
        "last_active_time"
    )

    if last_time is None:
        return False

    timeout = timedelta(
        minutes=TASK_TIMEOUT_MINUTES
    )

    return datetime.now() - last_time > timeout


# =========================
# 对话历史
# =========================

def add_message(
    role: str,
    content: str,
    image_bytes: bytes | None = None,
    image_caption: str = "",
):
    """
    添加一条消息。

    role:
    user
    assistant
    """

    if not content and not image_bytes:
        return

    st.session_state.chat_history.append({
        "role": role,
        "content": content,
        "image_bytes": image_bytes,
        "image_caption": image_caption,
        "time": datetime.now().isoformat()
    })

    trim_history()

    update_last_active_time()


def trim_history():
    """
    只保留最近10轮。

    一轮 = 用户 + assistant
    因此最多保留20条消息。
    """

    max_messages = (
        MAX_HISTORY_ROUNDS * 2
    )

    history = st.session_state.chat_history

    if len(history) > max_messages:
        st.session_state.chat_history = (
            history[-max_messages:]
        )


def get_chat_history() -> list:
    """
    获取当前最近聊天历史。
    """

    return st.session_state.chat_history


def get_chat_history_text() -> str:
    """
    将聊天历史转换成适合塞进 Prompt 的文字。
    """

    history = get_chat_history()

    if not history:
        return "暂无历史对话"

    lines = []

    for item in history:

        role = item.get("role")

        content = item.get(
            "content",
            ""
        )

        if role == "user":
            speaker = "用户"

        else:
            speaker = "助手"

        lines.append(
            f"{speaker}：{content}"
        )

    return "\n".join(lines)


# =========================
# 当前任务
# =========================

def start_task(
    task_type: str,
    goal: str = None,
    sub_task: str = None
):
    """
    创建新的任务状态。
    """

    now = datetime.now()

    st.session_state.task_state = {
        "task_type": task_type,
        "sub_task": sub_task,
        "goal": goal,
        "current_step": None,
        "status": "in_progress",
        "data": {},
        "started_at": now.isoformat(),
        "updated_at": now.isoformat()
    }

    update_last_active_time()


def update_task_state(
    goal=None,
    current_step=None,
    status=None,
    sub_task=None,
    data: dict = None
):
    """
    更新当前任务。

    data 用来存储额外关键字段。
    """

    task = st.session_state.task_state

    if goal is not None:
        task["goal"] = goal

    if current_step is not None:
        task["current_step"] = current_step

    if status is not None:
        task["status"] = status

    if sub_task is not None:
        task["sub_task"] = sub_task

    if data:

        task["data"].update(
            data
        )

    task["updated_at"] = (
        datetime.now().isoformat()
    )

    st.session_state.task_state = task

    update_last_active_time()


def get_task_state() -> dict:
    """
    获取当前任务。

    如果任务超时，自动标记 expired。
    """

    if is_task_expired():

        task = st.session_state.task_state

        if task["status"] == "in_progress":

            task["status"] = "expired"

            task["updated_at"] = (
                datetime.now().isoformat()
            )

            st.session_state.task_state = task

    return st.session_state.task_state


def finish_task():
    """
    标记任务完成。
    """

    update_task_state(
        status="completed"
    )


def cancel_task():
    """
    标记任务取消。
    """

    update_task_state(
        status="cancelled"
    )


def clear_task():
    """
    清空当前任务。
    """

    st.session_state.task_state = {
        "task_type": None,
        "sub_task": None,
        "goal": None,
        "current_step": None,
        "status": "idle",
        "data": {},
        "started_at": None,
        "updated_at": None
    }


# =========================
# 判断是否切换任务
# =========================

def should_start_new_task(
    new_task_type: str
) -> bool:
    """
    判断是否需要创建新任务。

    情况：
    1. 当前没有任务
    2. 任务已完成/取消/超时
    3. 用户切换到新的 task_type
    """

    task = get_task_state()

    current_type = task.get(
        "task_type"
    )

    status = task.get(
        "status"
    )

    if current_type is None:
        return True

    if status in [
        "completed",
        "cancelled",
        "expired",
        "idle"
    ]:
        return True

    if (
        new_task_type
        and new_task_type != current_type
    ):
        return True

    return False


# =========================
# Prompt上下文
# =========================

def get_memory_context() -> str:
    """
    生成给 Agent 使用的完整记忆上下文。
    """

    history_text = (
        get_chat_history_text()
    )

    task = get_task_state()

    context = f"""
【最近对话】

{history_text}


【当前任务状态】

任务类型：
{task.get("task_type")}

子任务：
{task.get("sub_task")}

用户目标：
{task.get("goal")}

当前步骤：
{task.get("current_step")}

任务状态：
{task.get("status")}

关键数据：
{task.get("data")}
"""

    return context

# =========================
# 当前活跃任务
# =========================

def get_active_task_type():
    """
    获取当前正在进行的任务类型。

    如果没有正在进行的任务，
    返回 None。
    """

    task = get_task_state()

    if (
        task.get("status") == "in_progress"
        and task.get("task_type")
    ):
        return task["task_type"]

    return None


# =========================
# 任务连续性判断
# =========================

def resolve_task_type(
    candidate_type: str,
    user_text: str = "",
    has_image: bool = False
) -> str:
    """
    根据当前任务状态判断：
    是继续原任务，
    还是切换到新任务。

    candidate_type：
    Router 本轮识别出的候选任务。

    返回最终 task_type。
    """

    current_type = get_active_task_type()

    # 当前没有任务
    if current_type is None:
        return candidate_type

    user_text = (
        user_text.strip()
        if user_text
        else ""
    )

    # -------------------------
    # Router 与当前任务一致
    # -------------------------

    if candidate_type == current_type:
        return current_type

    # -------------------------
    # 本轮没有文字，但上传了图片
    # -------------------------
    #
    # 例如：
    # Screenshot Agent 正在连续指导，
    # 用户只是继续上传下一张截图。
    #
    # 这种情况下优先延续当前任务。
    # -------------------------

    if (
        not user_text
        and has_image
    ):
        return current_type

    # -------------------------
    # Router 判断为 general
    # -------------------------
    #
    # 多轮对话中很多短回答：
    #
    # "可以"
    # "然后呢"
    # "第二个"
    # "我已经点了"
    #
    # 单独判断可能是 general，
    # 但实际上是在继续当前任务。
    # -------------------------

    if candidate_type == "general":
        return current_type

    # -------------------------
    # 明确识别为其他任务
    # -------------------------
    #
    # 例如：
    # screenshot -> fraud
    # fraud -> planner
    #
    # 认为用户切换了任务。
    # -------------------------

    return candidate_type
