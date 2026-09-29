"""银龄智办：手机页面、截图指导与可选语音播报。"""

from pathlib import Path
from datetime import datetime
from types import SimpleNamespace
import base64
import html
import inspect
import json

import streamlit as st

from agents.router import route_task, rule_route
from agents.image_router import route_image_task
from agents.fraud import run_fraud_agent
from agents.screenshot import run_screenshot_agent
from agents.planner import run_planner_agent

from services.llm import chat_with_llm
from services.speech import transcribe_audio
from services.tts import synthesize_speech, TTSUnavailable

from services.memory import (
    init_memory,
    add_message,
    get_chat_history,
    start_task,
    should_start_new_task,
    resolve_task_type,
    get_chat_history_text,
    is_task_expired,
    MAX_HISTORY_ROUNDS,
    TASK_TIMEOUT_MINUTES,
)


from config import MODEL_NAME

# ==================================================
# Router 埋点：仅重新绑定 app.py 的模块全局名，
# 不改动 process_submission 的任何字节，也不改 agents/router.py
# ==================================================

_original_route_task = route_task
_original_route_image_task = route_image_task


def _record_router_trace(candidate, text, via):
    """只写 UI 自己的字段，绝不触碰 chat_history / task_state。"""
    st.session_state.router_trace = {
        "candidate": candidate,
        "text": (text or "").replace("\n", " ")[:40],
        "via": via,
        "at": datetime.now().strftime("%H:%M:%S"),
    }


def route_task(text):
    """包装版：行为与原函数完全一致，只多记一条 UI 埋点。"""
    result = _original_route_task(text)
    rule_hit = bool(text) and (rule_route(text) is not None)
    _record_router_trace(result, text, "规则命中" if rule_hit else "模型判定")
    return result


def route_image_task(image, text):
    """包装版：同理。"""
    result = _original_route_image_task(image, text)
    _record_router_trace(result, text, "图像路由")
    return result

# ==================================================
# 页面与记忆初始化
# ==================================================

st.set_page_config(
    page_title="银龄智办",
    page_icon="👵",
    layout="centered",
)

init_memory()

# ==================================================
# 界面状态初始化
# ==================================================

if "page" not in st.session_state:
    st.session_state.page = "home"

if "font_scale_name" not in st.session_state:
    st.session_state.font_scale_name = "标准"

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

if "tts_enabled" not in st.session_state:
    st.session_state.tts_enabled = False

# 导航会提前 rerun；显式保留同一个 UI key，避免尚未渲染的开关被清理。
st.session_state.ws_detail = st.session_state.get("ws_detail", False)

BASE_DIR = Path(__file__).resolve().parent
LOGO_PATH = BASE_DIR / "assets" / "team_logo.png"


# ==================================================
# 【新增】界面常量
# ==================================================

FONT_SCALES = {"标准": 1.00, "大": 1.15, "特大": 1.30}
FONT_ORDER  = ["标准", "大", "特大"]

# 首页四个服务卡片
# ⚠️ 话术不要改！这四句话都刻意包含了规则路由关键词，
#    用来命中 agents/router.py 的 rule_route，保证任务类型 100% 确定。
SERVICE_CARDS = [
    {"key": "svc_doctor",   "title": "看医生", "sub": "挂号、就医、复诊",
     "ask": "我明天下午想去医院看医生，帮我安排一下。"},
    {"key": "svc_shopping", "title": "买东西", "sub": "买菜、买药",
     "ask": "帮我买菜。"},
    {"key": "svc_screen",   "title": "看屏幕", "sub": "教您一步步操作",
     "ask": "我不知道页面操作下一步点哪里，请一步一步教我。"},
    {"key": "svc_fraud",    "title": "防诈骗", "sub": "辨别可疑信息",
     "ask": "帮我看看是不是诈骗。"},
]

TASK_TYPE_CN = {
    "fraud": "防诈骗核查",
    "screenshot": "手机操作指导",
    "planner": "事务规划",
    "general": "智能问答",
}

TASK_STATUS_BADGE = {
    "in_progress": ("● 进行中", "badge-run"),
    "completed":   ("✅ 已完成", "badge-done"),
    "cancelled":   ("⛔ 已取消", "badge-cancel"),
    "expired":     ("⏰ 已超时", "badge-expire"),
    "idle":        ("未开始",   "badge-idle"),
}


# ==================================================
# 【第二阶段】Agent Workspace 常量
# ==================================================

WS_SUB_TASK_CN = {
    "medical": "就医", "shopping": "购物",
    "transport": "出行", "general": "通用",
}

WS_AGENT_TABLE = [
    ("fraud",      "反诈骗 Agent",   "agents/fraud.py",      "run_fraud_agent",      "风险判定 · 停止操作提示"),
    ("screenshot", "视觉指导 Agent", "agents/screenshot.py", "run_screenshot_agent", "目标定位 · 红框箭头绘制"),
    ("planner",    "事务规划 Agent", "agents/planner.py",    "run_planner_agent",    "就医 / 购物 / 出行规划"),
    ("general",    "通用问答",       "services/llm.py",      "chat_with_llm",        "无工具直答"),
]

WS_MODULES = [
    ("agents/router.py",             "意图路由 Router", "每轮"),
    ("agents/image_router.py",       "图像路由",        "有图时"),
    ("services/vision.py",           "视觉 / OCR",      "有图时"),
    ("services/screen_grounding.py", "目标定位",        "screenshot"),
    ("services/image_annotation.py", "红框箭头绘制",    "screenshot"),
    ("services/llm.py",              "文字模型",        "每轮"),
    ("services/speech.py",           "语音识别 ASR",    "录音时"),
    ("services/tts.py",              "语音合成 TTS",    "朗读开启"),
    ("services/memory.py",           "会话记忆 Memory", "每轮"),
]

# ==================================================
# 展示聊天记录
# ==================================================

def render_history(slot):
    """消息区域只显示消息，不创建输入框。"""
    with slot.container():
        history = get_chat_history()

        if not history:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(
                    "您好 👋\n\n"
                    "您可以直接告诉我遇到了什么问题。\n\n"
                    "- 有人让我转账，我该怎么办？\n"
                    "- 这个页面下一步点哪里？\n"
                    "- 我明天下午想去医院看膝盖。"
                )

        for item in history:
            content = str(item.get("content", ""))

            if item.get("role") == "user":
                safe_content = html.escape(content).replace(
                    "\n", "<br>"
                )

                st.markdown(
                    '<div class="user-message-row">'
                    '<div class="user-message-bubble">'
                    + safe_content
                    + '</div>'
                    '<div class="user-avatar">👤</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )

            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(content)

                    if item.get("image_bytes"):
                        st.image(
                            item["image_bytes"],
                            caption=(
                                item.get("image_caption")
                                or "请按照红框和箭头操作"
                            ),
                            width="stretch",
                        )


# ==================================================
# 诈骗风险结果展示
# ==================================================

def format_fraud_result(result):
    if not result["success"]:
        return result["error"]

    data = result["data"]

    risk_icon = {
        "high": "🛑",
        "medium": "⚠️",
    }.get(data["risk_level"], "✅")

    message = (
        f"## {risk_icon} {data['risk_label']}\n\n"
        f"**判断：** {data['summary']}\n\n"
        f"**可能类型：** {data['fraud_type']}"
    )

    if data["risk_points"]:
        message += "\n\n**发现的风险：**"

        for point in data["risk_points"]:
            message += f"\n- {point}"

    if data["actions"]:
        message += "\n\n**现在建议您：**"

        for index, action in enumerate(data["actions"], 1):
            message += f"\n{index}. {action}"

    if data["need_stop"]:
        message += (
            "\n\n🛑 **请先停止当前操作，"
            "不要继续付款、转账或提供验证码。**"
        )

    if data["need_family_help"]:
        message += (
            "\n\n👨‍👩‍👧 **建议请可信家人帮助确认。**"
        )

    return message


# ==================================================
# 调用对应 Agent
# ==================================================

def run_agent(task_type, user_text, uploaded_image):
    if task_type == "fraud":
        try:
            with st.spinner("正在分析风险..."):
                result = run_fraud_agent(
                    text=user_text,
                    image=uploaded_image,
                )

            return format_fraud_result(result)

        except Exception as exc:
            return f"诈骗风险分析失败：{exc}"

    if task_type == "screenshot":
        try:
            with st.spinner("正在处理您的操作问题..."):
                result = run_screenshot_agent(
                    text=user_text,
                    image=uploaded_image,
                )

            if not result["success"]:
                return result["error"]

            return {
                "content": result["result"],
                "image_bytes": result.get("image_bytes"),
                "image_caption": result.get(
                    "image_caption", ""
                ),
            }

        except Exception as exc:
            return f"截图识别失败：{exc}"

    if task_type == "planner":
        try:
            with st.spinner("正在为您安排..."):
                result = run_planner_agent(user_text)

            if result["success"]:
                return result["result"]

            return result["error"]

        except Exception as exc:
            return f"事务规划失败：{exc}"

    try:
        with st.spinner("正在思考..."):
            return chat_with_llm(user_text)

    except Exception as exc:
        return f"智能助手暂时无法回答：{exc}"


# ==================================================
# 展示最新回答的语音播放器
# ==================================================

def render_tts_player(slot, autoplay=False):
    enabled = st.session_state.get("tts_enabled", False)
    audio = st.session_state.get("last_tts_audio")

    if not enabled or not audio:
        slot.empty()
        return

    with slot.container():
        st.caption("最新回答朗读 · 没有自动播放时，请点 ▶")

        st.audio(
            audio,
            format="audio/mpeg",
            autoplay=autoplay,
        )


# ==================================================
# 处理本次提交
# ==================================================

def process_submission(submission, history_slot, tts_slot):
    # 新消息开始处理时，移除上一条音频。
    st.session_state.pop("last_tts_audio", None)
    tts_slot.empty()

    user_text = (submission.text or "").strip()

    uploaded_image = (
        submission.files[0]
        if submission.files
        else None
    )

    audio_value = submission.audio

    # ---------- 语音输入转文字 ----------

    if audio_value is not None:
        try:
            with st.spinner("正在识别您的语音..."):
                speech_text = (
                    transcribe_audio(audio_value) or ""
                ).strip()

            if speech_text:
                if user_text:
                    user_text += "\n\n语音补充：" + speech_text
                else:
                    user_text = speech_text

            else:
                st.warning(
                    "没有识别到清晰的语音，"
                    "请重新录音或输入文字。"
                )

        except Exception as exc:
            st.error(f"语音识别失败：{exc}")

    if not user_text and uploaded_image is None:
        if audio_value is None:
            st.warning(
                "请输入文字、进行语音输入或上传截图。"
            )
        return

    # ---------- Router 与任务连续性 ----------

    try:
        with st.spinner("正在理解您的需求..."):
            if uploaded_image is not None:
                try:
                    candidate_type = route_image_task(
                        uploaded_image,
                        user_text,
                    )

                except Exception:
                    candidate_type = (
                        route_task(user_text)
                        if user_text
                        else "general"
                    )

            else:
                candidate_type = route_task(user_text)

            task_type = resolve_task_type(
                candidate_type=candidate_type,
                user_text=user_text,
                has_image=uploaded_image is not None,
            )

            if should_start_new_task(task_type):
                start_task(
                    task_type=task_type,
                    goal=(
                        user_text
                        or "根据当前截图完成用户任务"
                    ),
                )

    except Exception as exc:
        st.error(
            f"暂时无法判断任务类型，请重新发送：{exc}"
        )
        return

    # ---------- 保存用户消息 ----------

    user_message = user_text or "请帮我看看这张截图。"

    if uploaded_image is not None:
        user_message += "\n\n🖼️ 已附带一张截图"

    add_message(
        role="user",
        content=user_message,
    )

    render_history(history_slot)

    # ---------- 获取并保存助手回复 ----------

    assistant_message = run_agent(
        task_type,
        user_text,
        uploaded_image,
    )

    if isinstance(assistant_message, dict):
        reply_text = assistant_message["content"]

        add_message(
            role="assistant",
            content=reply_text,
            image_bytes=assistant_message.get(
                "image_bytes"
            ),
            image_caption=assistant_message.get(
                "image_caption", ""
            ),
        )

    else:
        reply_text = assistant_message

        add_message(
            role="assistant",
            content=reply_text,
        )

    # 先显示文字，语音失败也不影响文字回答。
    render_history(history_slot)

    # ---------- 可选的回答朗读 ----------

    if st.session_state.get("tts_enabled", False):
        try:
            with st.spinner("正在准备朗读…"):
                audio = synthesize_speech(reply_text)

            if (
                st.session_state.get("tts_enabled", False)
                and audio
            ):
                st.session_state.last_tts_audio = audio

                render_tts_player(
                    tts_slot,
                    autoplay=True,
                )

        except TTSUnavailable as exc:
            st.warning(str(exc))

    # 不主动 rerun，避免重复处理本轮消息。


# ==================================================
# 界面样式：设计文档第 7 章
# ==================================================

def inject_base_style():
    """静态主题先注入；尺寸遵循第 3.6 节，内容溢出时可滚动。"""
    st.markdown(
        """<style>
:root, .stApp {
  /* ===== 字体缩放（由 Python 动态覆盖） ===== */
  --fs: 1;

  /* ===== 品牌色 ===== */
  --c-primary:      #E8590C;
  --c-primary-deep: #C2410C;
  --c-primary-soft: #FFF1E6;
  --c-primary-line: #FFD8BE;

  /* ===== 中性色 ===== */
  --c-bg:      #FFF9F2;
  --c-surface: #FFFFFF;
  --c-text:    #2B2B2B;
  --c-text-2:  #6B6B6B;
  --c-text-3:  #9A9088;
  --c-line:    #F0E4D8;

  /* ===== 功能色 ===== */
  --c-green:       #2E7D52;
  --c-green-soft:  #E8F5EE;
  --c-danger:      #D32F2F;
  --c-danger-soft: #FDECEC;
  --c-warn:        #B26A00;
  --c-warn-soft:   #FFF4E0;

  /* ===== 字号阶梯 ===== */
  --f-2xl: calc(26px * var(--fs));
  --f-xl:  calc(21px * var(--fs));
  --f-lg:  calc(18px * var(--fs));
  --f-md:  calc(16px * var(--fs));
  --f-sm:  calc(14px * var(--fs));
  --f-xs:  calc(12px * var(--fs));
  --f-nav: clamp(11px, calc(12px * var(--fs)), 15px);
  --f-icon-lg: calc(34px * var(--fs));
  --f-icon-nav: clamp(19px, calc(21px * var(--fs)), 26px);

  /* ===== 圆角 ===== */
  --r-xl: 24px;  --r-lg: 18px;  --r-md: 14px;
  --r-sm: 10px;  --r-pill: 999px;

  /* ===== 阴影 ===== */
  --sh-card:   0 2px 10px rgba(176, 118, 63, 0.07);
  --sh-raised: 0 6px 20px rgba(176, 118, 63, 0.12);
  --sh-nav:    0 -4px 18px rgba(176, 118, 63, 0.08);
  --sh-press:  inset 0 2px 6px rgba(176, 118, 63, 0.14);

  /* ===== 间距 ===== */
  --sp-1: 4px; --sp-2: 8px; --sp-3: 12px;
  --sp-4: 16px; --sp-5: 20px; --sp-6: 24px;

  /* ===== 骨架尺寸 ===== */
  --topbar-h: 64px;
  --navbar-h: 72px;
  --gap: 12px;
  --gap-sec: 20px;   /* 【P4-1 新增】区块之间的分隔：问候卡 → 服务卡片网格 */
  --font-cn: "PingFang SC", "HarmonyOS Sans SC", "Microsoft YaHei",
             "Hiragino Sans GB", "Source Han Sans CN",
             "Noto Sans CJK SC", system-ui, -apple-system, "Segoe UI",
             Roboto, sans-serif;
}

/* ---------- 页面高度（必须在 100dvh 之前写 100vh 作为回退） ---------- */
:root {
  --page-h: calc(100vh  - var(--topbar-h) - var(--navbar-h) - var(--gap) * 2);
  --page-h: calc(100dvh - var(--topbar-h) - var(--navbar-h) - var(--gap) * 2
                     - env(safe-area-inset-bottom, 0px));
}

/* ---------- 全局 ---------- */
html, body, .stApp {
  background: var(--c-bg) !important;
  font-family: var(--font-cn) !important;
  color: var(--c-text);
  -webkit-text-size-adjust: 100%;
}

/* 隐藏 Streamlit 自带的顶部工具条 */
[data-testid="stHeader"] { display: none !important; }

/* 屏蔽"运行中"转圈与旧内容变灰（切页零闪烁的关键） */
[data-testid="stStatusWidget"] { display: none !important; }
[data-stale="true"] {
  opacity: 1 !important;
  filter: none !important;
  transition: none !important;
}

/* 去掉 Streamlit 默认的面包屑、锚点、底部 footer */
[data-testid="stToolbar"]      { display: none !important; }
[data-testid="stDecoration"]   { display: none !important; }
footer, #MainMenu              { display: none !important; }

/* 全局按钮兜底：圆角、字体、最小触控尺寸 */
.stButton button,
.stDownloadButton button {
  font-family: var(--font-cn) !important;
  font-size: var(--f-lg) !important;
  border-radius: var(--r-md) !important;
  min-height: 48px !important;
  transition: transform .12s ease, box-shadow .12s ease, background .12s ease !important;
}
.stButton button:active { transform: scale(.97) !important; }
.stButton button p { margin: 0 !important; }

/* 主内容容器：手机宽度居中，给底栏预留空间 */
.block-container {
  max-width: 460px !important;
  margin: 0 auto !important;
  padding: 8px 16px
           calc(var(--navbar-h) + var(--gap) + env(safe-area-inset-bottom, 0px))
           16px !important;
}

/* 消除 Streamlit 默认的纵向块间距，改由我们自己控制 */
.stApp [data-testid="stVerticalBlock"] { gap: 0 !important; }

/* 通用纵向间距：用一个空的间隔容器控制，见第 8 章 $gap() */

.st-key-topbar {
  height: var(--topbar-h);
  display: flex;
  align-items: center;
  margin-bottom: var(--gap);
}
.st-key-topbar [data-testid="stHorizontalBlock"] {
  align-items: center !important;
  gap: 8px !important;
  width: 100% !important;
}
.st-key-topbar [data-testid="stColumn"] { padding: 0 !important; }

/* 品牌区 */
.app-brand {
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 0;
}
.app-brand-logo {
  width: 40px; height: 40px;
  border-radius: 12px;
  object-fit: contain;
  background: var(--c-surface);
  box-shadow: var(--sh-card);
  flex-shrink: 0;
}
.app-brand-title {
  font-size: var(--f-xl);
  font-weight: 700;
  line-height: 1.15;
  color: var(--c-text);
}
.app-brand-sub {
  font-size: var(--f-xs);
  line-height: 1.2;
  color: var(--c-text-2);
  margin-top: 2px;
  white-space: nowrap;
  overflow: visible;
  text-overflow: ellipsis;
}

/* 字号快捷按钮 A- / A+ —— 热区锁定 48px（3.6 节硬底线，不要改小） */
.st-key-font_dec button,
.st-key-font_inc button {
  width: 48px !important;
  min-width: 48px !important;
  height: 48px !important;
  min-height: 48px !important;
  padding: 0 !important;
  border-radius: var(--r-pill) !important;
  background: var(--c-surface) !important;
  border: 1.5px solid var(--c-line) !important;
  color: var(--c-primary-deep) !important;
  font-size: var(--f-md) !important;
  font-weight: 700 !important;
  box-shadow: var(--sh-card) !important;
}
.st-key-font_dec button:hover,
.st-key-font_inc button:hover {
  border-color: var(--c-primary) !important;
  background: var(--c-primary-soft) !important;
}

.st-key-bottom_nav {
  position: fixed !important;
  left: 50% !important;
  transform: translateX(-50%);
  bottom: 0 !important;
  width: 100% !important;
  max-width: 460px !important;
  z-index: 900 !important;
  background: rgba(255, 255, 255, .96);
  -webkit-backdrop-filter: blur(10px);
  backdrop-filter: blur(10px);
  border-top: 1px solid var(--c-line);
  border-radius: var(--r-lg) var(--r-lg) 0 0;
  box-shadow: var(--sh-nav);
  padding: 6px 8px calc(6px + env(safe-area-inset-bottom, 0px)) !important;
}

.st-key-bottom_nav [data-testid="stHorizontalBlock"] {
  gap: 0 !important;
  align-items: stretch !important;
}
.st-key-bottom_nav [data-testid="stColumn"] {
  padding: 0 3px !important;
  min-width: 0 !important;
}
.st-key-bottom_nav .stButton,
.st-key-bottom_nav .stButton button { width: 100% !important; }

.st-key-bottom_nav .stButton button {
  height: 56px !important;
  min-height: 56px !important;
  padding: 4px 0 !important;
  border: none !important;
  border-radius: var(--r-md) !important;
  background: transparent !important;
  box-shadow: none !important;
  display: flex !important;
  flex-direction: column !important;
  align-items: center !important;
  justify-content: center !important;
  gap: 2px !important;
  color: var(--c-text-2) !important;
  font-size: var(--f-nav) !important;
  font-weight: 600 !important;
  line-height: 1.1 !important;
}
.st-key-bottom_nav .stButton button p {
  margin: 0 !important;
  font-size: var(--f-nav) !important;
  line-height: 1.1 !important;
}

/* 图标（伪元素注入，实现"图标在上、文字在下"） */
.st-key-bottom_nav .stButton button::before {
  font-size: var(--f-icon-nav);
  line-height: 1;
  display: block;
  margin-bottom: 1px;
  filter: grayscale(1) opacity(.62);
  transition: filter .15s ease;
}
.st-key-nav_home    .stButton button::before { content: "🏠"; }
.st-key-nav_chat    .stButton button::before { content: "💬"; }
.st-key-nav_tasks   .stButton button::before { content: "📋"; }
.st-key-nav_profile .stButton button::before { content: "👤"; }

/* 选中态：primary 按钮 → 浅橙底 + 橙字 + 彩色图标 */
.st-key-bottom_nav .stButton button[kind="primary"],
.st-key-bottom_nav [data-testid="stBaseButton-primary"] {
  background: var(--c-primary-soft) !important;
  color: var(--c-primary) !important;
  box-shadow: none !important;
}
.st-key-bottom_nav .stButton button[kind="primary"]::before,
.st-key-bottom_nav [data-testid="stBaseButton-primary"]::before {
  filter: none;
}

/* 按压反馈 */
.st-key-bottom_nav .stButton button:active {
  transform: scale(.94) !important;
}

.st-key-page_home {
  height: var(--page-h);
  overflow-y: auto;
  overflow-x: hidden;
  display: flex;
  flex-direction: column;
}

/* ---------- 问候卡 ---------- */
/* 【P4-1b】抵消问候卡外层容器的 -16px 负下边距（Streamlit 容器链自带，非本项目 CSS） */
div:has(> .greet-card) { margin-bottom: 0 !important; }
.greet-card {
  position: relative;
  overflow: visible;
  border-radius: var(--r-xl);
  background: linear-gradient(135deg, #FFF1E6 0%, #FFE7D4 100%);
  padding: 18px;
  margin-bottom: 0;      /* 【P4-1】间距改由 .st-key-svc_row1 的 margin-top 控制 */
  flex-shrink: 0;
}
.greet-card::after {
  content: "🌼";
  position: absolute;
  right: 10px; top: -6px;
  font-size: 76px;
  line-height: 1;
  opacity: .16;
  pointer-events: none;
}
.greet-title {
  font-size: var(--f-2xl);
  font-weight: 700;
  line-height: 1.25;
  color: var(--c-primary-deep);
}
.greet-sub {
  margin-top: 6px;
  font-size: var(--f-md);
  line-height: 1.5;
  color: var(--c-text-2);
}

/* ---------- 服务卡片网格 ---------- */
.st-key-svc_row1, .st-key-svc_row2 { flex-shrink: 0; }
.st-key-svc_row1 { margin-top: var(--gap-sec); }   /* 【P4-1】区块分隔 20px */
.st-key-svc_row2 { margin-top: var(--gap); }
.st-key-svc_row1 [data-testid="stHorizontalBlock"],
.st-key-svc_row2 [data-testid="stHorizontalBlock"] {
  gap: var(--gap) !important;
  align-items: stretch !important;
}
.st-key-svc_row1 [data-testid="stColumn"],
.st-key-svc_row2 [data-testid="stColumn"] { padding: 0 !important; }

.st-key-svc_row1 .stButton,
.st-key-svc_row2 .stButton { height: 100% !important; }

.st-key-svc_row1 .stButton button,
.st-key-svc_row2 .stButton button {
  width: 100% !important;
  min-height: 108px !important;
  height: 100% !important;
  padding: 14px 10px !important;
  display: flex !important;
  flex-direction: column !important;
  align-items: center !important;
  justify-content: center !important;
  gap: 4px !important;
  background: var(--c-surface) !important;
  border: none !important;
  border-radius: var(--r-xl) !important;
  box-shadow: var(--sh-card) !important;
  color: var(--c-text) !important;
}
.st-key-svc_row1 .stButton button:hover,
.st-key-svc_row2 .stButton button:hover {
  box-shadow: var(--sh-raised) !important;
}
.st-key-svc_row1 .stButton button:active,
.st-key-svc_row2 .stButton button:active {
  transform: scale(.97) !important;
  box-shadow: var(--sh-press) !important;
}

/* 卡片图标 */
.st-key-svc_row1 .stButton button::before,
.st-key-svc_row2 .stButton button::before {
  font-size: var(--f-icon-lg);
  line-height: 1;
  display: block;
  margin-bottom: 2px;
}
.st-key-svc_doctor button::before { content: "🏥"; }
.st-key-svc_shopping button::before { content: "🛒"; }
.st-key-svc_screen button::before { content: "📱"; }
.st-key-svc_fraud button::before { content: "🛡️"; }

/* 卡片标题 / 副标题 */
.st-key-svc_row1 .stButton button p,
.st-key-svc_row2 .stButton button p {
  margin: 0 !important;
  text-align: center !important;
  line-height: 1.25 !important;
}
.st-key-svc_row1 .stButton button p:first-child,
.st-key-svc_row2 .stButton button p:first-child {
  font-size: var(--f-xl) !important;
  font-weight: 700 !important;
  color: var(--c-text) !important;
}
.st-key-svc_row1 .stButton button p:last-child,
.st-key-svc_row2 .stButton button p:last-child {
  font-size: var(--f-xs) !important;
  font-weight: 400 !important;
  color: var(--c-text-2) !important;
  margin-top: 3px !important;
}

/* ---------- 任务状态条 ---------- */
.st-key-task_banner { margin-top: var(--gap); flex-shrink: 0; }
.st-key-task_banner .stButton button {
  position: relative !important;
  width: 100% !important;
  min-height: 52px !important;
  height: 52px !important;
  padding: 0 54px 0 40px !important;
  justify-content: flex-start !important;
  background: var(--c-surface) !important;
  border: 2px solid var(--c-primary-line) !important;
  border-radius: var(--r-md) !important;
  box-shadow: none !important;
  font-size: var(--f-md) !important;
  font-weight: 600 !important;
  color: var(--c-text) !important;
  text-align: left !important;
}
.st-key-task_banner .stButton button p {
  text-align: left !important;
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
}
/* 左侧呼吸圆点 */
.st-key-task_banner .stButton button::before {
  content: "";
  position: absolute;
  left: 16px; top: 50%;
  margin-top: -4px;
  width: 8px; height: 8px;
  border-radius: 50%;
  background: var(--c-primary);
  animation: pulse-dot 1.6s ease-in-out infinite;
}
/* 右侧"查看 >" */
.st-key-task_banner .stButton button::after {
  content: "查看 ›";
  position: absolute;
  right: 14px; top: 50%;
  transform: translateY(-50%);
  font-size: var(--f-sm);
  font-weight: 400;
  color: var(--c-text-2);
}
@keyframes pulse-dot {
  0%, 100% { opacity: 1;   transform: scale(1);   }
  50%      { opacity: .35; transform: scale(.7);  }
}

/* ---------- 求助按钮 ---------- */
.st-key-help_btn { margin-top: var(--gap); flex-shrink: 0; }
.st-key-help_btn .stButton button {
  width: 100% !important;
  height: 56px !important;
  min-height: 56px !important;
  background: var(--c-primary-soft) !important;
  border: 1.5px solid var(--c-primary-line) !important;
  border-radius: var(--r-md) !important;
  box-shadow: none !important;
  color: var(--c-primary-deep) !important;
  font-size: var(--f-lg) !important;
  font-weight: 600 !important;
}
.st-key-help_btn .stButton button:hover {
  background: #FFE7D4 !important;
  border-color: var(--c-primary) !important;
}

.st-key-page_chat {
  height: var(--page-h);
  display: flex;
  flex-direction: column;
  overflow: visible;
}

/* ---------- 消息区 ---------- */
.st-key-conversation_panel {
  height: calc(var(--page-h) - 88px) !important;
  min-height: 200px !important;
  overflow-y: auto !important;
  background: transparent !important;
  border: none !important;
  border-radius: var(--r-lg) !important;
  padding: 4px 2px 12px 2px !important;
  flex: 1 1 auto;
}
.st-key-conversation_panel [data-testid="stVerticalBlockBorderWrapper"] {
  border: none !important;
  background: transparent !important;
}

/* 细滚动条 */
.st-key-conversation_panel::-webkit-scrollbar { width: 6px; }
.st-key-conversation_panel::-webkit-scrollbar-thumb {
  background: var(--c-line);
  border-radius: 3px;
}

/* ---------- 助手消息卡片 ---------- */
.st-key-conversation_panel [data-testid="stChatMessage"] {
  position: relative;
  overflow: visible;
  background: var(--c-surface) !important;
  border: 1px solid var(--c-line) !important;
  border-radius: var(--r-lg) !important;
  padding: 14px 16px 14px 20px !important;
  margin: 10px 0 !important;
  box-shadow: var(--sh-card);
  gap: 10px !important;
  min-width: 0 !important;
}
/* 左侧语气竖条（统一用主色，不用动态判断语气，见 6.10） */
.st-key-conversation_panel [data-testid="stChatMessage"]::before {
  content: "";
  position: absolute;
  left: 0; top: 0; bottom: 0;
  width: 4px;
  background: var(--c-primary);
}

/* 助手消息正文 */
.st-key-conversation_panel [data-testid="stMarkdownContainer"] p,
.st-key-conversation_panel [data-testid="stMarkdownContainer"] li {
  font-size: var(--f-md) !important;
  line-height: 1.7 !important;
  overflow-wrap: anywhere;
  color: var(--c-text);
}
.st-key-conversation_panel [data-testid="stMarkdownContainer"] h2 {
  font-size: var(--f-xl) !important;
  font-weight: 700 !important;
  margin: 2px 0 8px !important;
  color: var(--c-text) !important;
}
.st-key-conversation_panel [data-testid="stMarkdownContainer"] strong {
  font-weight: 700 !important;
  color: var(--c-primary-deep) !important;
}
.st-key-conversation_panel [data-testid="stMarkdownContainer"] ol,
.st-key-conversation_panel [data-testid="stMarkdownContainer"] ul {
  padding-left: 20px !important;
  margin: 6px 0 !important;
}

/* ---------- 用户消息气泡（沿用原 class，只改样式） ---------- */
.user-message-row {
  display: flex;
  justify-content: flex-end;
  align-items: flex-start;
  gap: 8px;
  margin: 12px 0;
  width: 100%;
  box-sizing: border-box;
}
.user-message-bubble {
  max-width: 82%;
  padding: 11px 14px;
  border-radius: var(--r-lg) var(--r-lg) 6px var(--r-lg);
  background: var(--c-primary-soft);
  border: 1px solid var(--c-primary-line);
  color: var(--c-text);
  font-size: var(--f-lg);
  line-height: 1.7;
  overflow-wrap: anywhere;
}
.user-avatar {
  width: 32px; height: 32px;
  min-width: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  border: 1px solid var(--c-primary-line);
  border-radius: 50%;
  background: var(--c-primary-soft);
  font-size: calc(17px * var(--fs));
  box-sizing: border-box;
  flex-shrink: 0;
}

/* ---------- 红框截图 ---------- */
.st-key-conversation_panel [data-testid="stImage"] img {
  max-width: 100% !important;
  height: auto !important;
  border-radius: var(--r-md) !important;
  border: 1px solid var(--c-line) !important;
}
.st-key-conversation_panel [data-testid="stImage"] figcaption,
.st-key-conversation_panel [data-testid="stCaptionContainer"] p {
  font-size: calc(15px * var(--fs)) !important;
  font-weight: 600 !important;
  color: var(--c-danger) !important;
  text-align: center !important;
  line-height: 1.5 !important;
}

/* ---------- 输入框 ---------- */
.st-key-composer_inline {
  flex-shrink: 0;
  margin-top: var(--gap);
}
.st-key-composer_inline [data-testid="stChatInput"] {
  border-radius: var(--r-lg) !important;
  border: 1.5px solid var(--c-line) !important;
  background: var(--c-surface) !important;
  box-shadow: var(--sh-card) !important;
  min-height: 60px !important;
  padding: 4px 6px !important;
}
.st-key-composer_inline [data-testid="stChatInput"]:focus-within {
  border-color: var(--c-primary) !important;
}
.st-key-composer_inline [data-testid="stChatInput"] textarea {
  font-size: var(--f-lg) !important;
  line-height: 1.5 !important;
  padding: 10px 4px !important;
  color: var(--c-text) !important;
}
.st-key-composer_inline [data-testid="stChatInput"] textarea::placeholder {
  color: var(--c-text-3) !important;
}
/* 附件 / 麦克风 / 发送 按钮放大到 48px（3.6 节硬底线，不要改小） */
.st-key-composer_inline [data-testid="stChatInput"] button {
  min-width: 48px !important;
  min-height: 48px !important;
  border-radius: var(--r-pill) !important;
  color: var(--c-text-2) !important;
}
.st-key-composer_inline [data-testid="stChatInput"] button:hover {
  background: var(--c-primary-soft) !important;
  color: var(--c-primary) !important;
}
/* 已选附件缩略图放大 */
.st-key-composer_inline [data-testid="stChatInput"] img {
  width: 64px !important;
  height: 64px !important;
  object-fit: cover !important;
  border-radius: var(--r-sm) !important;
  border: 2px solid var(--c-primary-line) !important;
}

/* ---------- 朗读播放器 ---------- */
.st-key-conversation_panel + div audio,
.stApp audio {
  width: 100% !important;
  height: 48px !important;
}

.st-key-page_tasks {
  height: var(--page-h);
  overflow-y: auto;
  overflow-x: hidden;
}

/* 大卡片通用外观 */
.task-card, .record-card, .empty-card {
  background: var(--c-surface);
  border: 1px solid var(--c-line);
  border-radius: var(--r-xl);
  box-shadow: var(--sh-card);
  padding: 16px;
  margin-bottom: var(--gap);
}
.task-card-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--c-line);
  margin-bottom: 12px;
}
.task-card-title {
  font-size: var(--f-lg);
  font-weight: 700;
  color: var(--c-text);
}
.badge {
  font-size: var(--f-xs);
  font-weight: 700;
  padding: 5px 10px;
  border-radius: var(--r-pill);
  color: #fff;
  white-space: nowrap;
}
.badge-run    { background: var(--c-primary-deep); }
.badge-done   { background: var(--c-green); }
.badge-cancel { background: var(--c-text-2); }
.badge-expire { background: var(--c-warn); }
.badge-idle   { background: var(--c-line); color: var(--c-text-2); }

.task-row {
  display: flex;
  gap: 10px;
  padding: 7px 0;
  align-items: flex-start;
}
.task-row-key {
  flex: 0 0 76px;
  font-size: var(--f-sm);
  color: var(--c-text-2);
  line-height: 1.6;
}
.task-row-val {
  flex: 1;
  font-size: var(--f-md);
  color: var(--c-text);
  line-height: 1.6;
  overflow-wrap: anywhere;
}

/* 记录行 */
.record-item {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  padding: 9px 0;
  border-bottom: 1px dashed var(--c-line);
}
.record-item:last-child { border-bottom: none; }
.record-role {
  flex-shrink: 0;
  font-size: var(--f-xs);
  font-weight: 700;
  padding: 3px 8px;
  border-radius: var(--r-sm);
  white-space: nowrap;
}
.record-role-user { background: var(--c-primary-soft); color: var(--c-primary-deep); }
.record-role-bot  { background: var(--c-green-soft);  color: var(--c-green); }
.record-text {
  flex: 1;
  font-size: var(--f-sm);
  color: var(--c-text);
  line-height: 1.6;
  overflow-wrap: anywhere;
}
.record-time {
  flex-shrink: 0;
  font-size: var(--f-xs);
  color: var(--c-text-3);
  padding-top: 2px;
}

/* 空状态 */
.empty-card { text-align: center; padding: 40px 20px; }
.empty-icon { font-size: calc(48px * var(--fs)); line-height: 1; }
.empty-title {
  margin-top: 14px;
  font-size: var(--f-lg);
  font-weight: 700;
  color: var(--c-text);
}
.empty-sub {
  margin-top: 6px;
  font-size: var(--f-sm);
  color: var(--c-text-2);
  line-height: 1.6;
}

/* 区块小标题 */
.section-title {
  font-size: var(--f-md);
  font-weight: 700;
  color: var(--c-text-2);
  margin: 4px 0 10px 2px;
}

.st-key-page_profile {
  height: var(--page-h);
  overflow-y: auto;
  overflow-x: hidden;
}
.settings-card {
  background: var(--c-surface);
  border: 1px solid var(--c-line);
  border-radius: var(--r-xl);
  box-shadow: var(--sh-card);
  padding: 16px;
  margin-bottom: var(--gap);
}
.settings-label {
  font-size: var(--f-lg);
  font-weight: 700;
  color: var(--c-text);
  margin-bottom: 10px;
}
.settings-hint {
  font-size: var(--f-sm);
  color: var(--c-text-2);
  line-height: 1.6;
  margin-top: 8px;
}

/* 字号三档按钮 */
.st-key-fs_row [data-testid="stHorizontalBlock"] {
  gap: 10px !important;
}
.st-key-fs_row .stButton button {
  width: 100% !important;
  height: 52px !important;
  min-height: 52px !important;
  border-radius: var(--r-md) !important;
  background: var(--c-surface) !important;
  border: 1.5px solid var(--c-line) !important;
  color: var(--c-text-2) !important;
  font-size: var(--f-md) !important;
  font-weight: 600 !important;
  box-shadow: none !important;
}
.st-key-fs_row .stButton button[kind="primary"],
.st-key-fs_row [data-testid="stBaseButton-primary"] {
  background: var(--c-primary-soft) !important;
  border: 2px solid var(--c-primary) !important;
  color: var(--c-primary) !important;
}

/* 朗读开关 */
.st-key-tts_toggle_widget { margin-top: 4px; }
.st-key-tts_toggle_widget [data-testid="stWidgetLabel"] p {
  font-size: var(--f-lg) !important;
  font-weight: 600 !important;
  color: var(--c-text) !important;
}
/* 放大开关本体 */
.st-key-tts_toggle_widget [role="switch"],
.st-key-tts_toggle_widget [data-baseweb="checkbox"] > div:first-child {
  transform: scale(1.3);
  transform-origin: left center;
}

/* 行式按钮（清空 / 说明 / 关于） */
.st-key-clear_chat .stButton button,
.st-key-help_doc  .stButton button,
.st-key-about_app .stButton button {
  width: 100% !important;
  height: 58px !important;
  min-height: 58px !important;
  justify-content: flex-start !important;
  padding: 0 16px !important;
  border-radius: var(--r-lg) !important;
  background: var(--c-surface) !important;
  border: 1px solid var(--c-line) !important;
  color: var(--c-text) !important;
  font-size: var(--f-md) !important;
  font-weight: 600 !important;
  box-shadow: var(--sh-card) !important;
  margin-bottom: var(--gap);
}
.st-key-clear_chat .stButton button p,
.st-key-help_doc  .stButton button p,
.st-key-about_app .stButton button p {
  text-align: left !important;
  font-size: var(--f-md) !important;
}
/* 行式按钮右侧箭头 */
.st-key-clear_chat .stButton button::after,
.st-key-help_doc  .stButton button::after,
.st-key-about_app .stButton button::after {
  content: "›";
  position: absolute;
  right: 18px;
  font-size: 22px;
  opacity: .45;
}
.st-key-clear_chat .stButton button,
.st-key-help_doc  .stButton button,
.st-key-about_app .stButton button {
  position: relative !important;
}

/* 提示条 */
.stAlert {
  border-radius: var(--r-md) !important;
  border: none !important;
  padding: 12px 14px !important;
  position: relative;
  overflow: visible;
}
.stAlert::before {
  content: "";
  position: absolute;
  left: 0; top: 0; bottom: 0;
  width: 4px;
}
.stAlert p {
  font-size: var(--f-md) !important;
  line-height: 1.6 !important;
}
.stAlert[data-baseweb="notification"] { background: var(--c-warn-soft) !important; }

/* 加载提示卡片化 */
.stSpinner > div {
  gap: 10px !important;
}
.stSpinner [data-testid="stMarkdownContainer"] p {
  font-size: var(--f-lg) !important;
  font-weight: 600 !important;
  color: var(--c-text) !important;
}
.stSpinner svg { color: var(--c-primary) !important; }

/* 弹层（求助 / 清空确认） */
[data-testid="stDialog"] > div {
  border-radius: var(--r-xl) !important;
  border: 1px solid var(--c-line) !important;
  background: var(--c-surface) !important;
}
[data-testid="stDialog"] h2 {
  font-size: var(--f-xl) !important;
  font-weight: 700 !important;
  color: var(--c-text) !important;
}
[data-testid="stDialog"] p {
  font-size: var(--f-md) !important;
  line-height: 1.7 !important;
}
/* 弹层内的"求助文字"代码块：改成正常字体，方便老人阅读与复制 */
[data-testid="stDialog"] code,
[data-testid="stDialog"] pre {
  font-family: var(--font-cn) !important;
  font-size: var(--f-sm) !important;
  line-height: 1.7 !important;
  white-space: pre-wrap !important;
  word-break: break-all !important;
  background: var(--c-primary-soft) !important;
  color: var(--c-text) !important;
  border-radius: var(--r-md) !important;
}

/* 文本域（若使用） */
.stTextArea textarea {
  font-size: var(--f-md) !important;
  border-radius: var(--r-md) !important;
}

@media (max-width: 400px) {
  :root, .stApp {
    --topbar-h: 58px;
    --navbar-h: 68px;
    --f-2xl: calc(23px * var(--fs));
  }
  .block-container {
    padding-left: 12px !important;
    padding-right: 12px !important;
  }
  .app-brand-logo { width: 36px; height: 36px; }
  .app-brand-sub   { display: none; }   /* 小屏隐藏副标题，给主标题让位 */
  .greet-card      { padding: 12px 15px; }
  .greet-card .greet-sub { display: none; }   /* 先压非交互元素，换取服务卡片的空间 */
  /* 服务卡片高度【不降】，守住 3.6 节的 100px 底线 */
  .st-key-svc_row1 .stButton button,
  .st-key-svc_row2 .stButton button { min-height: 100px !important; }
}

/* 矮屏（老机型如 iPhone SE / 横屏 / 地址栏占位大）：进入紧凑档
   压缩顺序严格遵循 5.1.3 节：装饰 > 次要文字 > 图标 > 滚动条，交互尺寸永不牺牲 */
@media (max-height: 640px) {
  :root, .stApp {
    --topbar-h: 56px;
    --navbar-h: 68px;   /* 6 + 6 内边距 + 56 按钮 */
    --gap: 8px;
    --gap-sec: 8px;   /* 【P4-1】紧凑档净增仅 4px，不挤破一屏 */
  }
  .app-brand-sub         { display: none; }   /* 先压装饰：顶栏副标题 */
  .greet-card            { padding: 10px 14px; min-height: 64px; }
  .greet-card .greet-sub { display: none; }   /* 先压装饰：问候卡副标题 */
  /* 任务状态条 52 → 48（仍远高于 48px 底线，此处是高度压缩不是触控压缩） */
  .st-key-task_banner .stButton button {
    height: 48px !important;
    min-height: 48px !important;
  }
  /* 服务卡片与按钮触控尺寸【一律不降】 */
  .st-key-svc_row1 .stButton button,
  .st-key-svc_row2 .stButton button { min-height: 100px !important; }
}

/* 桌面端：仍按手机宽度居中，两侧露出暖色背景 */
@media (min-width: 461px) {
  .stApp { background: var(--c-bg) !important; }
  .block-container { box-shadow: 0 0 0 1px rgba(176,118,63,.06); }
}

@keyframes fade-up {
  from { opacity: 0; transform: translateY(6px); }
  to   { opacity: 1; transform: translateY(0); }
}
.st-key-conversation_panel [data-testid="stChatMessage"] {
  animation: fade-up .18s ease-out;
}

/* 尊重系统的"减弱动态效果"设置 */
@media (prefers-reduced-motion: reduce) {
  * { animation: none !important; transition: none !important; }
}

/* 任务页 / 我的页的卡片容器（替换 7.7 / 7.8 中对应 class 的用法） */
.st-key-settings_card,
.st-key-empty_go_wrap {
  background: var(--c-surface);
  border: 1px solid var(--c-line);
  border-radius: var(--r-xl);
  box-shadow: var(--sh-card);
  padding: 16px;
  margin-bottom: var(--gap);
}

/* 任务页的"去首页""查看完整对话"按钮 */
.st-key-empty_go  { margin-top: var(--gap); }
.st-key-task_to_chat { margin-top: var(--gap); }
.st-key-empty_go .stButton button,
.st-key-task_to_chat .stButton button {
  width: 100% !important;
  height: 56px !important;
  min-height: 56px !important;
  background: var(--c-primary) !important;
  border: none !important;
  border-radius: var(--r-md) !important;
  color: #fff !important;
  font-size: var(--f-lg) !important;
  font-weight: 600 !important;
}

/* Streamlit 1.63 DOM fallback: help tooltips insert wrappers around buttons.
   Keep selectors scoped to stable widget/container keys, never generated classes. */
html { box-sizing: border-box; }
*, *::before, *::after { box-sizing: inherit; }
.block-container { padding-top: 0 !important; transform: none; }
.st-key-topbar { height: var(--topbar-h) !important; min-height: var(--topbar-h) !important; }
.st-key-topbar > [data-testid="stLayoutWrapper"] { width: 100% !important; }
.st-key-topbar [data-testid="stHorizontalBlock"] {
  display: grid !important; grid-template-columns: minmax(0, 1fr) 48px 48px;
  gap: 12px !important;
}
.st-key-topbar [data-testid="stColumn"] { width: 100% !important; min-width: 0 !important; }
.app-brand-title { font-size: clamp(18px, calc(21px * var(--fs)), 25px); white-space: nowrap; }
.app-brand-sub { white-space: normal; overflow-wrap: anywhere; }
.st-key-font_dec button:disabled, .st-key-font_inc button:disabled { opacity: .4; }
.st-key-page_home, .st-key-page_tasks, .st-key-page_profile, .st-key-page_chat {
  height: var(--page-h) !important; min-height: 0 !important;
  overflow-y: auto !important; overflow-x: hidden !important;
  flex: 0 0 auto !important; width: 100% !important; padding: 2px 2px 8px;
}
.st-key-page_chat { display: flex !important; flex-direction: column !important; }
.st-key-page_home > *, .st-key-page_profile > *, .st-key-page_tasks > * { flex-shrink: 0 !important; }
.st-key-page_chat > [data-testid="stLayoutWrapper"]:has(.st-key-conversation_panel) {
  flex: 1 1 200px !important; min-height: 200px !important; height: auto !important;
}
.st-key-conversation_panel { height: 100% !important; min-height: 200px !important; }
.st-key-page_chat > :not([data-testid="stLayoutWrapper"]:has(.st-key-conversation_panel)) { flex-shrink: 0 !important; }
.st-key-svc_row1 [data-testid="stHorizontalBlock"],
.st-key-svc_row2 [data-testid="stHorizontalBlock"] {
  display: grid !important; grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px !important;
}
.st-key-svc_row1 [data-testid="stColumn"], .st-key-svc_row2 [data-testid="stColumn"],
.st-key-fs_row [data-testid="stColumn"], .st-key-bottom_nav [data-testid="stColumn"] {
  width: 100% !important; min-width: 0 !important;
}
.st-key-svc_row1 .stElementContainer, .st-key-svc_row2 .stElementContainer,
.st-key-fs_row .stElementContainer, .st-key-bottom_nav .stElementContainer,
.st-key-task_banner .stElementContainer, .st-key-help_btn .stElementContainer,
.st-key-clear_chat .stElementContainer, .st-key-help_doc .stElementContainer,
.st-key-about_app .stElementContainer, .st-key-empty_go .stElementContainer,
.st-key-task_to_chat .stElementContainer { width: 100% !important; }
.st-key-svc_row1 [data-testid="stMarkdownContainer"],
.st-key-svc_row2 [data-testid="stMarkdownContainer"] {
  display: flex !important; flex-direction: column !important; align-items: center;
}
.st-key-svc_row1 button, .st-key-svc_row2 button { min-height: 112px !important; padding: 10px 6px !important; }
.st-key-svc_row1 button strong, .st-key-svc_row2 button strong { font-size: var(--f-xl); line-height: 1.25; }
.st-key-bottom_nav [data-testid="stHorizontalBlock"] {
  display: grid !important; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px !important;
}
.st-key-bottom_nav [data-testid="stColumn"] { padding: 0 !important; }
.st-key-bottom_nav { height: calc(var(--navbar-h) + env(safe-area-inset-bottom, 0px)) !important; }
.st-key-bottom_nav button[kind="primary"] { color: var(--c-primary-deep) !important; }
.st-key-fs_row [data-testid="stHorizontalBlock"] {
  display: grid !important; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px !important;
}
.stButton button { min-width: 48px !important; overflow-wrap: anywhere; }
.stButton button p { font-size: inherit; font-family: var(--font-cn); }
.stButton button[kind="secondary"] { color: var(--c-text); border-color: var(--c-line); }
.stButton button[kind="primary"] { background: var(--c-primary-deep); border-color: var(--c-primary-deep); color: var(--c-surface); }
.st-key-fs_row button[kind="primary"] { color: var(--c-primary-deep) !important; }
.st-key-task_banner button { height: auto !important; min-height: 52px !important; padding-top: 8px !important; padding-bottom: 8px !important; }
.st-key-task_banner button p { white-space: normal !important; overflow: visible !important; padding-right: 4px; }
.task-card-head { flex-wrap: wrap; }
.task-row-val, .record-text { min-width: 0; }
.record-time { color: var(--c-text-2); }
.st-key-tts_toggle_widget label { min-height: 48px; padding: 8px 0; }
.st-key-tts_toggle_widget [data-baseweb="checkbox"] { gap: 12px; }
.st-key-tts_toggle_widget input:checked + div { background: var(--c-primary-deep); }
.st-key-composer_inline [data-testid="stChatInput"] button { width: 48px !important; height: 48px !important; }
.st-key-composer_inline [data-testid="stChatInputSubmitButton"] {
  background: var(--c-primary-deep) !important; color: var(--c-surface) !important;
  border-radius: var(--r-md) !important;
}
.st-key-composer_inline [data-testid="stChatInput"] { gap: 12px; }
.st-key-composer_inline textarea { min-width: 0; }
.stApp [data-testid="stCaptionContainer"] { color: var(--c-text-2); }
.stApp [data-testid="stMarkdownContainer"] p { overflow-wrap: anywhere; }
.st-key-page_profile p, .st-key-page_tasks p { font-size: var(--f-md); }
.stSpinner { padding: 8px 12px; background: var(--c-surface); border-radius: var(--r-lg); }
[role="dialog"] { max-height: calc(100dvh - 24px) !important; overflow-y: auto !important; }
[role="dialog"] button, [data-testid="stCodeCopyButton"] { min-width: 48px !important; min-height: 48px !important; }
[role="dialog"] pre, [role="dialog"] code { white-space: pre-wrap !important; overflow-wrap: anywhere; }
@media (max-width: 400px) {
  .app-brand { gap: 6px; }
  .app-brand-title { font-size: clamp(18px, calc(21px * var(--fs)), 23px); }
  .st-key-svc_row1 button, .st-key-svc_row2 button { min-height: 100px !important; }
}
@media (max-height: 640px) {
  .st-key-topbar [data-testid="stHorizontalBlock"], .st-key-bottom_nav [data-testid="stHorizontalBlock"],
  .st-key-fs_row [data-testid="stHorizontalBlock"] { gap: 8px !important; }
  .greet-title { font-size: clamp(16px, calc(18px * var(--fs)), 23px); }
  .greet-card { min-height: 48px; padding: 6px 12px; }
  .st-key-svc_row1 button, .st-key-svc_row2 button { min-height: 100px !important; padding: 6px !important; }
  .st-key-svc_row1 button::before, .st-key-svc_row2 button::before { font-size: 28px; }
  .st-key-task_banner .stButton button { min-height: 48px !important; }
}


/* Native control wrappers: preserve 48px targets and visible two-line text. */
.st-key-tts_toggle_widget button { width: 48px !important; height: 48px !important; min-width: 48px !important; min-height: 48px !important; }
.st-key-tts_toggle_widget label:has(input:checked) > div:first-of-type { background: var(--c-primary-deep) !important; }
.st-key-composer_inline [data-testid="stChatInput"] > div { background: var(--c-surface) !important; border-radius: var(--r-lg); }
.st-key-composer_inline [data-testid="stChatInputTextArea"] {
  min-height: calc(var(--f-lg) * 3 + 20px) !important;
  background: var(--c-surface) !important; overflow-y: auto !important;
}
.st-key-composer_inline [data-testid="stChatInputSubmitButton"] { border-radius: var(--r-md) !important; }
.st-key-composer_inline [data-testid="stChatInputSubmitButton"]:disabled { opacity: .4; }
.st-key-help_btn button p { font-size: var(--f-lg) !important; font-weight: 600 !important; }
.st-key-clear_chat button, .st-key-help_doc button, .st-key-about_app button { margin-bottom: 0 !important; }
.st-key-clear_chat, .st-key-help_doc, .st-key-about_app { margin-bottom: var(--gap) !important; }

/* Toast close controls share the same touch-target floor as all other buttons. */
[data-testid="stToast"] button, [data-testid="stToastContainer"] button,
button[aria-label="Close"] { min-width: 48px !important; min-height: 48px !important; }

/* Attachment thumbnails: native flex shrink must not reduce the 64px preview. */
.st-key-composer_inline [data-testid="stChatInput"] img {
  min-width: 64px !important; max-width: 64px !important;
  min-height: 64px !important; flex: 0 0 64px !important;
}

.st-key-composer_inline div:has(> [data-testid="stFileChipImagePreview"]) {
  width: 64px !important; min-width: 64px !important; height: 64px !important; flex: 0 0 64px !important;
}

.st-key-composer_inline div:has(> [data-testid="stChatInputSubmitButton"]) { gap: var(--gap) !important; }

/* ============================================================
   【第二阶段】Agent Workspace —— 电脑演示端
   默认隐藏；仅 ≥1280px 显示。手机端完全不受影响。
   ============================================================ */

:root {
  /* ⚠️ 不要写死 400px：1280px 视口下会与手机列重叠 6px。
     连续公式保证 ≥1280px 时与手机列的水平间隙恒 ≥ 20px（见裁决 7） */
  --ws-w: min(400px, calc(50vw - 266px));
  --ws-bg:     #1E1E22;
  --ws-panel:  #26262C;
  --ws-code:   #14141A;
  --ws-line:   #3A3A44;
  --ws-text:   #ECECF1;
  --ws-text-2: #A8A8B6;
  --ws-accent: #7BD88F;
  --ws-warn:   #FFB454;
  --ws-danger: #FF6B6B;
}

/* 默认隐藏（窄屏 / 手机） */
.st-key-workspace { display: none !important; }

@media (min-width: 1280px) {

  /* 【P4-8】手机列与右侧面板"严格等高对齐"：
     ① 两者共用同一个高度上限 --ws-h = min(100dvh-32px, 70dvh)；
     ② --page-h 由「视口高」改为「--ws-h」，页面区随之收窄，手机列不再铺满整屏；
     ③ 手机列外框顶部从 16px 开始，与面板 top:16px 齐平；
     ④ 底栏由贴视口底改为贴手机列底边，与面板底边落在同一条水平线上。
     全部只在 ≥1280px 生效，手机端一个像素都不受影响。 */
  :root, .stApp {
    --ws-h: min(calc(100dvh - 32px), 70dvh);
    --page-h: calc(var(--ws-h) - var(--topbar-h) - var(--navbar-h) - var(--gap) * 2
                   - env(safe-area-inset-bottom, 0px));
  }
  .block-container { margin-top: 16px !important; }
  .st-key-bottom_nav { bottom: calc(100dvh - 16px - var(--ws-h)) !important; }

  .st-key-workspace {
    display: block !important;
    position: fixed !important;
    top: 16px !important;
    /* 【P4-2】跟随居中的手机列，固定留出 20px 水平间隔。 */
    right: auto !important;
    left: calc(50% + 250px) !important;
    /* 【P4-3】按内容定高；超出视口时只滚动面板内部。 */
    bottom: auto !important;
    height: auto !important;
    max-height: min(calc(100dvh - 32px), 70dvh) !important;
    width: var(--ws-w) !important;
    max-width: var(--ws-w) !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    z-index: 800 !important;
    background: var(--ws-bg);
    border: 1px solid var(--ws-line);
    border-radius: 12px;
    padding: 14px;
    box-shadow: 0 12px 40px rgba(0, 0, 0, .28);
    /* 终端风：固定字号，不引用 --fs */
    font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
    font-size: 12px;
    line-height: 1.6;
    color: var(--ws-text);
  }

  /* 压掉面板内 Streamlit 原生元素的外观 */
  .st-key-workspace [data-testid="stVerticalBlock"] { gap: 10px !important; }
  .st-key-workspace p,
  .st-key-workspace li,
  .st-key-workspace span,
  .st-key-workspace div[data-testid="stMarkdownContainer"] {
    color: var(--ws-text) !important;
    font-family: inherit !important;
    font-size: 12px !important;
  }
  .st-key-workspace h1, .st-key-workspace h2,
  .st-key-workspace h3, .st-key-workspace h4 {
    color: var(--ws-text) !important;
    font-size: 13px !important;
    margin: 4px 0 !important;
  }
  .st-key-workspace hr {
    border: none !important;
    border-top: 1px solid var(--ws-line) !important;
    margin: 10px 0 !important;
  }
  .st-key-workspace code,
  .st-key-workspace pre {
    background: var(--ws-code) !important;
    color: var(--ws-accent) !important;
    font-size: 11px !important;
    white-space: pre-wrap !important;
    overflow-wrap: anywhere !important;
  }
  /* 折叠开关：key="ws_detail" → Streamlit 生成的容器类名是 .st-key-ws_detail */
  .st-key-ws_detail label { min-height: 48px !important; padding: 8px 0 !important; }
  .st-key-ws_detail p { font-size: 12px !important; }
  .st-key-ws_detail [data-baseweb="checkbox"] { gap: 10px; }
  .st-key-ws_detail input:checked + div { background: var(--ws-accent) !important; }
}

/* ---------- 面板内自定义类（用自己的 markdown 渲染） ---------- */

.ws-hd {
  display: flex; align-items: baseline; justify-content: space-between;
  font-size: 12px; letter-spacing: .06em; color: var(--ws-accent);
  font-weight: 700; margin-bottom: 8px;
}
.ws-hd-sub { font-size: 11px; color: var(--ws-text-2); font-weight: 400; letter-spacing: 0; }

.ws-card {
  background: var(--ws-panel);
  border: 1px solid var(--ws-line);
  border-radius: 8px;
  padding: 10px 12px;
}
.ws-title {
  font-size: 11px; letter-spacing: .08em; text-transform: uppercase;
  color: var(--ws-text-2); margin-bottom: 6px;
}
.ws-row {
  display: flex; justify-content: space-between; align-items: baseline; gap: 10px;
  padding: 3px 0; border-bottom: 1px dashed rgba(255, 255, 255, .06);
}
.ws-row:last-child { border-bottom: none; }
.ws-k { color: var(--ws-text-2); flex: 0 0 auto; }
.ws-v {
  color: var(--ws-text); text-align: right;
  min-width: 0; overflow-wrap: anywhere;
}
.ws-badge {
  display: inline-block; padding: 1px 8px; border-radius: 999px;
  font-size: 11px; border: 1px solid currentColor; white-space: nowrap;
}
.ws-run  { color: var(--ws-warn); }
.ws-done { color: var(--ws-accent); }
.ws-idle { color: var(--ws-text-2); }
.ws-danger { color: var(--ws-danger); }

.ws-dot {
  display: inline-block; width: 6px; height: 6px; border-radius: 50%;
  background: var(--ws-accent); margin-right: 5px; vertical-align: middle;
  animation: wsPulse 1.4s ease-in-out infinite;
}
@keyframes wsPulse { 0%, 100% { opacity: .25; } 50% { opacity: 1; } }

.ws-agent-row {
  display: flex; justify-content: space-between; align-items: baseline; gap: 8px;
  padding: 4px 6px; border-radius: 6px; color: var(--ws-text-2);
}
.ws-agent-on {
  background: rgba(123, 216, 143, .12);
  color: var(--ws-accent); font-weight: 700;
}
.ws-agent-mod { font-size: 11px; opacity: .8; }

.ws-bar {
  height: 4px; border-radius: 999px; background: rgba(255, 255, 255, .12);
  overflow: hidden; margin-top: 5px;
}
.ws-bar > i { display: block; height: 100%; background: var(--ws-accent); }

.ws-muted { color: var(--ws-text-2); }
.ws-tag-off { color: var(--ws-text-2); opacity: .5; }

/* Streamlit 包装层不占手机列空间；所有覆写仅限 Workspace。 */
[data-testid="stLayoutWrapper"]:has(> .st-key-workspace) { display: contents !important; }
@media (min-width: 1280px) {
  .st-key-workspace { height: auto !important; min-height: 0 !important; box-sizing: border-box; display: flex !important; flex-direction: column !important; gap: 10px !important; }
  .st-key-workspace > * { min-width: 0; flex-shrink: 0 !important; }
  .st-key-workspace[data-testid="stVerticalBlock"] { gap: 10px !important; }
  .st-key-workspace [data-testid="stMarkdownContainer"] { margin-bottom: 0 !important; }
  .st-key-workspace .ws-hd { color: var(--ws-accent) !important; font-size: 13px !important; }
  .st-key-workspace .ws-title, .st-key-workspace .ws-k,
  .st-key-workspace .ws-muted, .st-key-workspace .ws-hd-sub,
  .st-key-workspace .ws-agent-mod, .st-key-workspace .ws-agent-duty,
  .st-key-workspace [data-testid="stCaptionContainer"] p {
    color: var(--ws-text-2) !important; font-size: 11px !important;
  }
  .st-key-workspace .ws-k { max-width: 55%; white-space: normal; }
  .st-key-workspace .ws-row, .st-key-workspace .ws-agent-row { min-width: 0; overflow-wrap: anywhere; }
  .st-key-workspace .ws-agent-row { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 2px 8px; }
  .st-key-workspace .ws-agent-row > span { min-width: 0; }
  .st-key-workspace .ws-agent-mod { text-align: right; opacity: 1; }
  .st-key-workspace .ws-agent-duty { grid-column: 1 / -1; }
  .st-key-workspace .ws-agent-on > span, .st-key-workspace .ws-done { color: var(--ws-accent) !important; }
  .st-key-workspace .ws-run { color: var(--ws-warn) !important; }
  .st-key-workspace .ws-idle { color: var(--ws-text-2) !important; }
  .st-key-workspace .ws-tag-off { opacity: 1; }
  .st-key-workspace .ws-bar { overflow: visible; }
  .st-key-workspace button { min-width: 48px !important; min-height: 48px !important; color: var(--ws-text) !important; }
  .st-key-workspace [data-testid="stCodeCopyButton"],
  .st-key-workspace button[kind="elementToolbar"] { background: var(--ws-panel) !important; border: 1px solid var(--ws-line) !important; }
  .st-key-workspace .st-key-ws_detail label:has(input:checked) > div:first-of-type { background: var(--ws-accent) !important; }
  .st-key-workspace [data-testid="stCode"] { min-width: 0; max-width: 100%; }
  .st-key-workspace pre, .st-key-workspace code, .st-key-workspace code span {
    font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace !important;
    font-size: 11px !important; color: var(--ws-accent) !important;
    white-space: pre-wrap !important; overflow-wrap: anywhere !important;
  }
}

        </style>""",
        unsafe_allow_html=True,
    )


def inject_font_scale():
    """在静态主题之后注入当前字号。"""
    name = st.session_state.get("font_scale_name", "标准")
    scale = FONT_SCALES.get(name, 1.0)
    st.markdown(
        f"<style>:root, .stApp {{ --fs: {scale}; }}</style>",
        unsafe_allow_html=True,
    )


# ==================================================
# 【新增】顶栏
# ==================================================

def render_topbar():
    logo_html = ""
    if LOGO_PATH.exists():
        logo_base64 = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
        logo_html = (
            f'<img class="app-brand-logo" '
            f'src="data:image/png;base64,{logo_base64}" alt="团队标志">'
        )

    with st.container(key="topbar"):
        c_brand, c_minus, c_plus = st.columns([1, 0.14, 0.14])

        with c_brand:
            st.markdown(
                '<div class="app-brand">'
                + logo_html
                + '<div>'
                '<div class="app-brand-title">银龄智办</div>'
                '<div class="app-brand-sub">老年人数字生活智能助手</div>'
                '</div>'
                '</div>',
                unsafe_allow_html=True,
            )

        order = FONT_ORDER
        index = order.index(st.session_state.font_scale_name)

        with c_minus:
            if st.button(
                "A−", key="font_dec",
                disabled=(index == 0),
                help="把字调小一点",
            ):
                st.session_state.font_scale_name = order[index - 1]
                st.session_state.font_notice = order[index - 1]
                st.rerun()

        with c_plus:
            if st.button(
                "A+", key="font_inc",
                disabled=(index == len(order) - 1),
                help="把字调大一点",
            ):
                st.session_state.font_scale_name = order[index + 1]
                st.session_state.font_notice = order[index + 1]
                st.rerun()

    if st.session_state.get("font_notice"):
        st.toast("字号：" + st.session_state.pop("font_notice"))


# ==================================================
# 【新增】首页
# ==================================================

def render_home():
    hour = datetime.now().hour
    if hour < 11:
        greet, icon = "早上好", "🌞"
    elif hour < 13:
        greet, icon = "中午好", "☀️"
    elif hour < 18:
        greet, icon = "下午好", "🌤️"
    else:
        greet, icon = "晚上好", "🌙"

    st.markdown(
        '<div class="greet-card">'
        f'<div class="greet-title">{greet}，欢迎使用银龄智办 {icon}</div>'
        '<div class="greet-sub">您想办什么事？点一下就行。</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ---------- 四个服务卡片（2 × 2） ----------
    for row_key, row_cards in (
        ("svc_row1", SERVICE_CARDS[0:2]),
        ("svc_row2", SERVICE_CARDS[2:4]),
    ):
        with st.container(key=row_key):
            cols = st.columns(2)
            for col, card in zip(cols, row_cards):
                with col:
                    clicked = st.button(
                        f"**{card['title']}**\n\n{card['sub']}",
                        key=card["key"],
                        help=f"点击后自动帮您处理「{card['title']}」",
                    )
                    if clicked:
                        st.session_state.pending_question = card["ask"]
                        st.session_state.page = "chat"
                        st.rerun()

    # ---------- 任务状态条（有进行中任务时才出现） ----------
    task = st.session_state.get("task_state", {}) or {}
    if task.get("status") == "in_progress" and task.get("task_type"):
        type_cn = TASK_TYPE_CN.get(task["task_type"], "智能问答")
        with st.container(key="task_banner"):
            if st.button(
                f"📋 正在进行：{type_cn}",
                key="banner_go",
                help="点一下看任务详情",
            ):
                st.session_state.page = "tasks"
                st.rerun()

    # ---------- 求助家人 ----------
    with st.container(key="help_btn"):
        if st.button(
            "🤝  需要家人帮忙",
            key="help_open",
            help="生成一段话，复制后发给家人",
        ):
            show_help_dialog()


# ==================================================
# 【新增】对话页（承载原有全部业务逻辑）
# ==================================================

def render_chat():
    # 加载提示放在消息区上方
    status_slot = st.empty()

    # ---------- 消息区：调用形式与原代码保持一致 ----------
    with st.container(
        height=520,
        border=True,
        key="conversation_panel",
        autoscroll=True,
    ):
        history_slot = st.empty()

        # 【P4-7】本轮若由首页卡片带问题进来、且当前无历史，先不在这里渲染：
        # 否则同一个 empty 槽会被 render_history 写两次（这一次 + process_submission 内部那次），
        # 前端会把首页已过期的卡片节点（data-stale）复用进对话面板，约 2 秒后消失。
        _skip_first_history = bool(
            st.session_state.get("pending_question")
        ) and not get_chat_history()

        if not _skip_first_history:
            render_history(history_slot)

    # ---------- 朗读播放器 ----------
    tts_slot = st.empty()
    render_tts_player(tts_slot, autoplay=False)

    # ---------- 输入框：参数与原代码完全一致 ----------
    with st.container(key="composer_inline"):
        submission = st.chat_input(
            "请输入您的问题…",
            key="main_chat_input",
            accept_file=True,
            file_type=["png", "jpg", "jpeg"],
            accept_audio=True,
            audio_sample_rate=16000,
        )

    # ---------- 本轮处理 ----------
    if submission is not None:
        with status_slot.container():
            process_submission(submission, history_slot, tts_slot)

    # ---------- 首页服务卡片带过来的问题 ----------
    elif st.session_state.get("pending_question"):
        question = st.session_state.pending_question
        st.session_state.pending_question = None      # ⚠️ 立刻清空，防止重复发送
        with status_slot.container():
            process_submission(
                make_manual_submission(question),
                history_slot,
                tts_slot,
            )


# ==================================================
# 【新增】模拟用户提交（让服务卡片能触发原有流程）
# ==================================================

def make_manual_submission(text):
    """构造一个与 st.chat_input 返回值同形的对象。

    这样 process_submission() 一个字都不用改就能复用。
    """
    return SimpleNamespace(text=text, files=[], audio=None)


# ==================================================
# 【新增】任务页（只读展示）
# ==================================================

def render_tasks():
    task = st.session_state.get("task_state", {}) or {}
    history = st.session_state.get("chat_history", []) or []

    if not task.get("task_type") and not history:
        # ---------- 空状态 ----------
        st.markdown(
            '<div class="empty-card">'
            '<div class="empty-icon">📋</div>'
            '<div class="empty-title">还没有进行中的任务</div>'
            '<div class="empty-sub">去首页点一个服务卡片，'
            '或者到对话页和助手聊聊吧。</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        with st.container(key="empty_go"):
            if st.button("去首页看看", key="empty_go_home"):
                st.session_state.page = "home"
                st.rerun()
        return

    # ---------- 当前任务卡 ----------
    if task.get("task_type"):
        _render_task_card(task)

    # ---------- 本轮对话记录 ----------
    st.markdown('<div class="section-title">本轮对话</div>',
                unsafe_allow_html=True)

    if not history:
        st.markdown(
            '<div class="record-card">'
            '<div class="record-text">这一轮还没有对话记录。</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        rows = []
        for item in list(reversed(history))[:10]:
            role = item.get("role")
            role_cls = "record-role-user" if role == "user" else "record-role-bot"
            role_txt = "您" if role == "user" else "助手"
            content = str(item.get("content", ""))
            if item.get("image_bytes"):
                content = "🖼️ " + content
            content = content.replace("\n", " ")
            if len(content) > 30:
                content = content[:30] + "…"
            ts = str(item.get("time", ""))
            ts = ts[11:16] if len(ts) >= 16 else ""
            rows.append(
                f'<div class="record-item">'
                f'<span class="record-role {role_cls}">{role_txt}</span>'
                f'<span class="record-text">{html.escape(content)}</span>'
                f'<span class="record-time">{ts}</span>'
                f'</div>'
            )
        st.markdown('<div class="record-card">' + "".join(rows) + "</div>",
                    unsafe_allow_html=True)

    with st.container(key="task_to_chat"):
        if st.button("查看完整对话", key="task_go_chat"):
            st.session_state.page = "chat"
            st.rerun()


def _render_task_card(task):
    status = task.get("status", "idle")
    badge_txt, badge_cls = TASK_STATUS_BADGE.get(status, ("未开始", "badge-idle"))
    type_cn = TASK_TYPE_CN.get(task.get("task_type"), "智能问答")

    def clip(value, n):
        value = str(value or "").replace("\n", " ")
        return value[:n] + "…" if len(value) > n else (value or "—")

    started = str(task.get("started_at") or "")
    started = started[11:16] if len(started) >= 16 else "—"
    updated = str(task.get("updated_at") or "")
    updated = updated[11:16] if len(updated) >= 16 else "—"

    st.markdown(
        '<div class="task-card">'
        '<div class="task-card-head">'
        '<span class="task-card-title">📋  当前任务</span>'
        f'<span class="badge {badge_cls}">{badge_txt}</span>'
        '</div>'
        f'<div class="task-row"><span class="task-row-key">任务类型</span>'
        f'<span class="task-row-val">{type_cn}</span></div>'
        f'<div class="task-row"><span class="task-row-key">您的目标</span>'
        f'<span class="task-row-val">{html.escape(clip(task.get("goal"), 40))}</span></div>'
        f'<div class="task-row"><span class="task-row-key">当前步骤</span>'
        f'<span class="task-row-val">{html.escape(clip(task.get("current_step"), 60))}</span></div>'
        f'<div class="task-row"><span class="task-row-key">开始时间</span>'
        f'<span class="task-row-val">{started}　·　最后更新 {updated}</span></div>'
        '</div>',
        unsafe_allow_html=True,
    )


# ==================================================
# 【新增】我的页
# ==================================================

def render_profile():
    # ---------- 显示与听感 ----------
    with st.container(key="settings_card"):
        st.markdown('<div class="settings-label">字体大小</div>',
                    unsafe_allow_html=True)

        with st.container(key="fs_row"):
            cols = st.columns(3)
            for col, name in zip(cols, FONT_ORDER):
                with col:
                    active = st.session_state.font_scale_name == name
                    if st.button(
                        name,
                        key=f"fs_{name}",
                        type="primary" if active else "secondary",
                        help=f"把界面字体设为「{name}」",
                    ):
                        st.session_state.font_scale_name = name
                        st.rerun()

        px = round(16 * FONT_SCALES[st.session_state.font_scale_name])
        st.markdown(
            f'<div class="settings-hint">当前：{st.session_state.font_scale_name}'
            f'（正文 {px} 号字）。也可以点右上角的 A+ / A− 快速调整。</div>',
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        st.toggle(
            "🔊 朗读回答",
            value=st.session_state.tts_enabled,
            key="tts_toggle_widget",
            on_change=lambda: st.session_state.update(
                tts_enabled=st.session_state.tts_toggle_widget
            ),
            help=(
                "开启后朗读新回答。"
                "回答文字会发送给在线语音服务；"
                "关闭后移除播放器并停止播报。"
            ),
        )
        st.markdown(
            '<div class="settings-hint">开启后，每次回答都会读给您听。</div>',
            unsafe_allow_html=True,
        )

    # ---------- 关键：这段逻辑必须保留（与原代码一致） ----------
    if not st.session_state.tts_enabled:
        st.session_state.pop("last_tts_audio", None)

    # ---------- 清空对话记录 ----------
    with st.container(key="clear_chat"):
        if st.button("🗑️  清空对话记录", key="clear_chat_btn",
                     help="删除本轮的聊天内容和任务进度"):
            confirm_clear_dialog()

    # ---------- 使用说明 ----------
    with st.container(key="help_doc"):
        if st.button("📖  使用说明", key="help_doc_btn", help="教您怎么用"):
            show_help_doc_dialog()

    # ---------- 关于 ----------
    with st.container(key="about_app"):
        if st.button("ℹ️  关于银龄智办", key="about_btn", help="关于本项目"):
            show_about_dialog()


# ==================================================
# 【新增】底部导航
# ==================================================

NAV_ITEMS = [
    ("home",    "首页", "回到首页"),
    ("chat",    "对话", "和智能助手聊天"),
    ("tasks",   "任务", "看看任务进行到哪一步"),
    ("profile", "我的", "字号、朗读和其它设置"),
]

def render_bottom_nav():
    current = st.session_state.page
    with st.container(key="bottom_nav"):
        cols = st.columns(4)
        for col, (pid, label, tip) in zip(cols, NAV_ITEMS):
            with col:
                if st.button(
                    label,
                    key=f"nav_{pid}",
                    type="primary" if current == pid else "secondary",
                    help=tip,
                ):
                    st.session_state.page = pid
                    st.rerun()


# ==================================================
# 【新增】弹层
# ==================================================

def build_help_text():
    """生成求助文案。纯字符串拼接，不调用任何 AI。"""
    task = st.session_state.get("task_state", {}) or {}
    history = st.session_state.get("chat_history", []) or []

    type_cn = TASK_TYPE_CN.get(task.get("task_type"), "用手机")
    goal = task.get("goal") or "（还没有说明）"
    step = task.get("current_step") or "（还没有开始）"

    last_user = ""
    for item in reversed(history):
        if item.get("role") == "user":
            last_user = str(item.get("content", "")).split("\n")[0][:40]
            break

    return (
        "【银龄智办 · 求助】\n"
        "我在用手机时遇到点困难，麻烦帮我看看：\n\n"
        f"· 我正在做的事：{type_cn}\n"
        f"· 我的目标：{goal}\n"
        f"· 我现在的进度：{step}\n"
        f"· 我最后说的是：{last_user or '（还没说话）'}\n\n"
        "谢谢你！"
    )


@st.dialog("需要家人帮忙")
def show_help_dialog():
    st.markdown("把下面这段话复制给家人，他们就能知道您卡在哪一步了。")
    st.code(build_help_text(), language=None)
    if st.button("知道了", key="help_close", type="primary"):
        st.rerun()


@st.dialog("确认清空")
def confirm_clear_dialog():
    st.markdown("确定要清空这一轮的对话记录和任务进度吗？清空后不能恢复。")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("取消", key="clear_cancel"):
            st.rerun()
    with c2:
        if st.button("确定清空", key="clear_ok", type="primary"):
            st.session_state.chat_history = []
            st.session_state.task_state = {
                "task_type": None, "sub_task": None, "goal": None,
                "current_step": None, "status": "idle", "data": {},
                "started_at": None, "updated_at": None,
            }
            st.session_state.pop("last_tts_audio", None)
            st.session_state.pending_question = None
            st.session_state.page = "home"
            st.rerun()


@st.dialog("使用说明")
def show_help_doc_dialog():
    st.markdown(
        "1. 打字提问：在「对话」页输入您的问题，按发送。\n"
        "2. 语音提问：点输入框里的麦克风，说完再发送。\n"
        "3. 截图指导：点输入框的「+」，选一张手机截图，助手会用红框告诉您点哪里。\n"
        "4. 朗读回答：到「我的」页打开「朗读回答」。\n"
        "5. 看不清字：点右上角的 A+ 放大，或到「我的」页选字号。"
    )
    st.info(
        "用手机录音需要在安全的网址下使用；如果麦克风点了没反应，"
        "可以直接用输入法的语音转文字，再说一遍。"
    )
    if st.button("知道了", key="help_doc_close", type="primary"):
        st.rerun()


@st.dialog("关于银龄智办")
def show_about_dialog():
    st.markdown("**银龄智办**\n\n老年人数字生活智能助手。\n\n界面版本：1.0")
    if st.button("知道了", key="about_close", type="primary"):
        st.rerun()


def _ws_build_context() -> str:
    """只读复刻面板上下文，不写入任何会话状态。"""
    history_text = get_chat_history_text()
    task = st.session_state.get("task_state") or {}
    return f"""
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


def _ws_hhmmss(iso_text):
    if not iso_text:
        return "—"
    try:
        return datetime.fromisoformat(str(iso_text)).strftime("%H:%M:%S")
    except (ValueError, TypeError):
        return "—"


def _ws_elapsed(iso_text):
    if not iso_text:
        return "—"
    try:
        seconds = max(0, int((datetime.now() - datetime.fromisoformat(str(iso_text))).total_seconds()))
        return f"{seconds} 秒" if seconds < 60 else f"{seconds // 60} 分 {seconds % 60} 秒"
    except (ValueError, TypeError):
        return "—"


def _ws_row(key, value_html):
    return (
        f'<div class="ws-row"><span class="ws-k">{html.escape(str(key))}</span>'
        f'<span class="ws-v">{value_html}</span></div>'
    )


def _ws_card(title, rows):
    return (
        f'<div class="ws-card"><div class="ws-title">{html.escape(title)}</div>'
        f'{"".join(rows)}</div>'
    )


def _ws_badge(status):
    text = TASK_STATUS_BADGE.get(status, ("未开始", ""))[0]
    cls = {"in_progress": "ws-run", "completed": "ws-done"}.get(status, "ws-idle")
    dot = '<span class="ws-dot"></span>' if status == "in_progress" else ""
    return f'<span class="ws-badge {cls}">{dot}{html.escape(text)}</span>'


def render_workspace():
    """Workspace 仅展示已有状态；不发请求，不推进或修改任务。"""
    with st.container(key="workspace"):
        task = st.session_state.get("task_state") or {}
        history = st.session_state.get("chat_history") or []
        trace = st.session_state.get("router_trace") or {}
        task_type = task.get("task_type")
        status = task.get("status") or "idle"
        context = _ws_build_context()
        data = task.get("data") or {}

        st.markdown(
            '<div class="ws-hd">Agent Workspace'
            '<span class="ws-hd-sub">实时系统状态</span></div>',
            unsafe_allow_html=True,
        )
        st.caption("面板严格只读 · 不调用底层写操作函数（渲染期不会改写任务状态）")

        rows = [
            _ws_row("任务类型", html.escape(TASK_TYPE_CN.get(task_type, "—"))),
            _ws_row("子任务", html.escape(WS_SUB_TASK_CN.get(task.get("sub_task"), "—"))),
            _ws_row("状态", _ws_badge(status)),
            _ws_row("已运行", html.escape(_ws_elapsed(task.get("started_at")))),
            _ws_row("最近更新", html.escape(_ws_hhmmss(task.get("updated_at")))),
        ]
        if task.get("goal"):
            rows.append(_ws_row("用户目标", html.escape(str(task["goal"]))))
        if task.get("current_step"):
            rows.append(_ws_row("当前步骤", html.escape(str(task["current_step"]))))
        st.markdown(_ws_card("① 当前任务", rows), unsafe_allow_html=True)

        rows = []
        for key, name, module, func, duty in WS_AGENT_TABLE:
            cls = "ws-agent-row ws-agent-on" if key == task_type and status != "idle" else "ws-agent-row"
            rows.append(
                f'<div class="{cls}"><span>{html.escape(name)}</span>'
                f'<span class="ws-agent-mod">{html.escape(module)}<br>{html.escape(func)}</span>'
                f'<span class="ws-agent-duty">{html.escape(duty)}</span></div>'
            )
        st.markdown(_ws_card("② 执行 Agent", rows), unsafe_allow_html=True)

        phase = {"idle": "待机", "in_progress": "执行中", "completed": "已完成",
                 "cancelled": "已取消", "expired": "已超时"}.get(status, "待机")
        rows = [
            _ws_row("阶段", html.escape(phase)),
            _ws_row("最近动作", html.escape(_ws_hhmmss(history[-1].get("time")) if history else "—")),
            _ws_row("任务已超时", "是" if is_task_expired() else "否"),
        ]
        if status == "in_progress" and task.get("current_step"):
            rows.append(_ws_row("当前步骤", html.escape(str(task["current_step"]))))
        if status == "completed":
            rows.append(_ws_row("结束时间", html.escape(_ws_hhmmss(task.get("updated_at")))))
        st.markdown(_ws_card("③ 运行阶段", rows), unsafe_allow_html=True)

        if trace:
            st.markdown(_ws_card("④ Router 状态", [
                _ws_row("用户输入", html.escape(str(trace.get("text", "—")))),
                _ws_row("路由通道", html.escape(str(trace.get("via", "—")))),
                _ws_row("本轮判定", html.escape(TASK_TYPE_CN.get(trace.get("candidate"), "—"))),
                _ws_row("最终类型", html.escape(TASK_TYPE_CN.get(task_type, "—"))
                        + ' <span class="ws-muted">（任务连续性优先）</span>'),
                _ws_row("时间", html.escape(str(trace.get("at", "—")))),
            ]), unsafe_allow_html=True)
        else:
            st.markdown(_ws_card("④ Router 状态", [_ws_row("状态", "尚未路由")]), unsafe_allow_html=True)

        limit = max(1, MAX_HISTORY_ROUNDS * 2)
        used = len(history)
        percent = min(100, round(used / limit * 100))
        known = data.get("known_info") or {}
        missing = data.get("missing_info") or []
        st.markdown(_ws_card("⑤ Memory 状态", [
            _ws_row("对话条数", f'{used} / {limit}<div class="ws-bar"><i style="width:{percent}%"></i></div>'),
            _ws_row("已存信息", f'{len(known)} 项' + (" · " + html.escape("、".join(map(str, known))) if known else "")),
            _ws_row("待补信息", f'{len(missing)} 项' + (" · " + html.escape("、".join(map(str, missing))) if missing else "")),
            _ws_row("最后活跃", html.escape(_ws_hhmmss(st.session_state.get("last_active_time")))),
            _ws_row("任务超时", f"{TASK_TIMEOUT_MINUTES} 分钟"),
            _ws_row("上下文长度（UI 复刻）", f"{len(context)} 字符"),
        ]), unsafe_allow_html=True)

        # 图像通道记录本轮输入；绘图证据仅取最近一条助手消息，不沿用历史图片。
        has_image = trace.get("via") == "图像路由"
        latest_reply = next((item for item in reversed(history) if item.get("role") == "assistant"), {})
        annotated = task_type == "screenshot" and bool(latest_reply.get("image_bytes"))
        audio_enabled = bool(st.session_state.get("tts_enabled", False))
        module_state = {
            "agents/router.py": str(trace.get("at") or "尚未路由"),
            "agents/image_router.py": "本轮有图" if has_image else "本轮无图",
            "services/vision.py": "本轮有图" if has_image else "本轮无图",
            "services/screen_grounding.py": "最近回复有标注图" if annotated else "最近回复无标注图",
            "services/image_annotation.py": "最近回复有标注图" if annotated else "最近回复无标注图",
            "services/llm.py": MODEL_NAME,
            "services/speech.py": "录音能力可用" if "accept_audio" in inspect.signature(st.chat_input).parameters else "录音能力不可用",
            "services/tts.py": ("开启 · 已有音频" if st.session_state.get("last_tts_audio") else "开启 · 暂无音频") if audio_enabled else "关闭",
            "services/memory.py": f"{used} 条对话",
        }
        rows = []
        for module, name, when in WS_MODULES:
            rows.append(
                f'<div class="ws-agent-row"><span>{html.escape(name)}</span>'
                f'<span class="ws-agent-mod">{html.escape(str(module_state[module]))}</span>'
                f'<span class="ws-agent-duty">{html.escape(module)} · {html.escape(when)}</span></div>'
            )
        rows.append('<div class="ws-agent-row ws-tag-off"><span>数据库（预留）</span>'
                    '<span>未接入</span><span class="ws-agent-duty">services/database.py</span></div>')
        st.markdown(_ws_card("⑥ 系统协同模块", rows), unsafe_allow_html=True)

        st.divider()
        if st.toggle("更多技术细节", key="ws_detail"):
            st.caption("task_state.data")
            st.code(json.dumps(data, ensure_ascii=False, indent=2), language=None)
            st.caption("router_trace")
            st.code(json.dumps(trace, ensure_ascii=False, indent=2), language=None)
            st.caption("_ws_build_context()（UI 只读复刻）")
            st.code(context, language=None)


# ==================================================
# 主流程：静态样式 → 字号 → 能力检查 → 顶栏 → 当前页 → 底栏
# ==================================================

inject_base_style()
inject_font_scale()

if (
    "accept_audio"
    not in inspect.signature(st.chat_input).parameters
):
    st.error(
        "当前 Streamlit 版本不支持输入框录音，"
        "请更新后重启。"
    )
    st.stop()

if (
    "autoscroll"
    not in inspect.signature(st.container).parameters
):
    st.error(
        "当前 Streamlit 版本不支持对话框自动滚动，"
        "请更新后重启。"
    )
    st.stop()


render_topbar()

_page = st.session_state.page
if _page == "home":
    with st.container(key="page_home"):
        render_home()
elif _page == "chat":
    with st.container(key="page_chat"):
        render_chat()
elif _page == "tasks":
    with st.container(key="page_tasks"):
        render_tasks()
else:
    with st.container(key="page_profile"):
        render_profile()

render_bottom_nav()
render_workspace()
