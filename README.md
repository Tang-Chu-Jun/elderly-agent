# 银龄智办 · 使用说明与运行手册

**银龄智办**是面向老年人数字生活的任务执行与安全守护智能体。老人用说话、打字或发一张截图提出需求，系统自动判断该交给哪个智能体处理，把“要做什么”拆成“下一步点哪里”，并在遇到可疑信息时提前提醒。界面为手机竖屏设计，电脑浏览器按窄屏居中展示。

目录：[一、这个项目是什么](#一这个项目是什么) · [二、准备项目](#二准备项目) · [三、首次安装](#三首次安装) · [四、启动程序](#四启动程序) · [五、四个页面怎么用](#五四个页面怎么用) · [六、四大功能怎么用](#六四大功能怎么用) · [七、适老化设计](#七适老化设计) · [八、在手机上打开网页](#八在手机上打开网页) · [九、手机网页能打开但不能录音](#九手机网页能打开但不能录音怎么办) · [十、常见启动问题](#十常见启动问题) · [十一、最常用命令](#十一最常用命令)。

## 一、这个项目是什么

系统按需求选择四类处理方式。它会给出文字、语音或截图红框指引；实际手机操作仍由使用者完成。

|任务类型|界面显示|能做什么|
|---|---|---|
|fraud|防诈骗核查|识别可疑短信、链接和话术；遇到高风险信息提示先停下来|
|screenshot|手机操作指导|上传截图后结合文字识别与画面理解，用红框指示下一步该点哪里|
|planner|事务规划|把看医生、买菜等目标拆成可执行步骤|
|general|智能问答|回答不属于上述任务的普通问题|

### 两个启动入口，请先选一个

|文件|界面|用在哪|
|---|---|---|
|`app.py`|手机 App 界面，四个页面|**默认入口**，日常使用与交付|
|`app_with_Agent_workspace.py`|手机界面 + 右侧深色 Agent 运行面板|**演示/评审**：让评委看见智能体的决策过程|

两者的业务处理相同。技术演示版多一个只读的 Agent Workspace，显示当前任务、路由结果、各智能体状态和运行时间，以及本轮对话上下文。

> ⚠️ 右侧面板由样式控制，**浏览器窗口宽度小于 1280px 时会自动隐藏**。看不到面板不是故障；演示时把窗口拉宽到至少 1280px（建议最大化）。用 `app.py` 启动的产品版本来就没有这块面板。

## 二、准备项目

解压到一个项目文件夹，例如 `D:\elderly_agent`。**下文示例路径请替换为你实际解压的目录。**打开该文件夹后，应直接看到 `app.py` 和 `app_with_Agent_workspace.py`，而不是另一层同名文件夹。文件名是 **requirements.txt（带 s）**。

主要文件结构：

```text
<项目根目录>/
├── app.py                        # 产品版入口（默认，无 Agent 面板）
├── app_with_Agent_workspace.py   # 技术演示版入口（右侧带 Agent 运行面板）
├── config.py                     # 读取 .env 的三个变量
├── requirements.txt              # 依赖清单（8 个直接依赖）
├── README.md
├── .env                          # 本机配置，含真实密钥，不公开上传
├── agents/                       # 路由器与各类 Agent 逻辑（不要改）
│   ├── router.py                 # 文字任务路由（规则优先）
│   ├── image_router.py           # 带图任务路由
│   ├── fraud.py                  # 防诈骗核查
│   ├── screenshot.py             # 截图操作指导
│   └── planner.py                # 事务规划
├── services/                     # 底层能力（不要改）
│   ├── llm.py  vision.py  memory.py
│   ├── speech.py                # 本地 Whisper 语音识别
│   ├── tts.py                   # edge-tts 朗读
│   ├── screen_grounding.py      # OCR 文本定位 → 红框坐标
│   └── image_annotation.py      # 画红框和箭头
├── prompts/                      # 提示词（不要改）
├── data/                         # 预留目录：fraud_cases / knowledge / opeartion_guides
├── assets/team_logo.png          # 顶栏 Logo（缺失时顶栏降级，不影响启动）
├── test_pictures/                # 6 张测试图片，供演示/自测用
└── test_config.py  test_llm.py  test_router.py  # 三个连通性自检脚本
```

新机器建议使用 **64 位 Python 3.12**。本清单已在 Windows + Python 3.12.4 环境完成安装与启动验收；其它操作系统或未列出的 Python 版本未做完整验证。不要把另一台机器的 `.venv` 当作可直接搬运的运行环境。

`requirements.txt` 锁定下面 8 个直接依赖，不是包含全部间接依赖的完整锁文件：

```text
streamlit==1.63.0
python-dotenv==1.2.3
openai==3.8.0
Pillow==12.3.0
rapidocr==3.9.2
onnxruntime==1.30.0
faster-whisper==1.2.1
edge-tts==7.2.8
```

`streamlit==1.63.0` **不能降级**：代码使用 `st.chat_input` 的 `accept_audio` 和容器的 `autoscroll` 参数，旧版本不具备这些能力时会提示错误并停止。

## 三、首次安装

### 1. 打开PowerShell，进入项目文件夹

```powershell
cd D:\elderly_agent
```

实际解压位置不同，就将路径换成你自己的。

### 2. 创建虚拟环境（仅新机器或没有可用环境时执行）

先确认安装了Python 3.12，再执行：

```powershell
py -3.12 -m venv .venv
```

如果没有 `py` 命令，可先检查：

```powershell
python --version
```

确认是计划使用的Python版本后执行：

```powershell
python -m venv .venv
```

已有能正常运行的 `.venv` 时，跳过创建步骤。不要直接覆盖仍在使用的环境。

### 3. 安装依赖

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

看到 `No broken requirements found.` 表示pip未发现依赖声明冲突，不代表所有功能都已验收。

这些命令直接指定虚拟环境Python，无需先运行Activate.ps1，也不需要修改PowerShell执行策略。

项目使用新版 `rapidocr`，不要改装旧的 `rapidocr-onnxruntime`。当前OCR代码读取 `result.boxes`、`result.txts` 和 `result.scores`。

### 4. 配置API

如果项目原来的 `.env` 已经能用，保留原有配置。

新机器在 `app.py` 同级新建 `.env` 文件，用自己实际可用的配置填写：

```dotenv
OPENAI_API_KEY=你的真实API密钥
OPENAI_BASE_URL=与你的密钥匹配的API基础地址
MODEL_NAME=该服务实际支持的模型名称
```

注意：

- 文件名应为 `.env`，不是 `.env.txt`。
- 三项必须来自同一个兼容的模型服务配置；不要把不同服务的密钥和地址混用。
- 截图分析需要模型支持图像输入，只有文字能力的模型无法完成该功能。
- `config.py` 虽有默认地址和模型名，也不代表你的账号一定能使用；优先沿用已经验证可用的三项配置。
- 向他人公开源代码时，用占位配置说明，不公开真实密钥；运行者需在自己的环境中配置密钥。

## 四、启动程序

在 PowerShell 中进入项目目录，然后按用途选择一个入口。建议两个版本使用不同端口：

```powershell
cd D:\elderly_agent
# 产品版：老人使用，也是默认入口
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port=8501
# 技术演示版：右侧带 Agent 运行面板
.\.venv\Scripts\python.exe -m streamlit run app_with_Agent_workspace.py --server.port=8502
```

程序启动后通常自动打开浏览器。没有自动打开时，产品版在电脑浏览器输入 `http://localhost:8501`，技术演示版输入 `http://localhost:8502`。如果终端显示其他端口，以终端输出为准。保持对应 PowerShell 窗口运行；关闭窗口或按 `Ctrl+C` 会停止服务，电脑睡眠也会影响访问。以后再次使用，只需进入项目目录并执行启动命令，不必每次重新安装依赖。

## 五、四个页面怎么用

顶栏显示 Logo、品牌名和字号快捷按钮 **A− / A+**。底部导航常驻，点「首页 / 对话 / 任务 / 我的」可一步切换页面。

### 首页

上方是随时间变化的问候卡，中间是 2×2 排列的四张服务卡。有进行中的任务时会出现任务状态条，页面下方有「🤝  需要家人帮忙」按钮，可生成一段求助文字。**点一下服务卡，系统会跳到对话页并替您把对应的话发出去**，不用打字。

### 对话

这是**唯一有输入框的页面**。可打字、点输入框内的麦克风说话，或点「+」上传截图；回答下方有朗读播放器。

### 任务

只读展示当前任务的类型、目标、状态、当前步骤和开始时间，以及本轮对话摘要；也可从这里跳回完整对话。任务内容在这一页不能直接修改。

### 我的

可选「标准 / 大 / 特大」三档字号，顶栏 **A− / A+** 也能快捷调整；「🔊 朗读回答」控制新回答的语音播放。「🗑️  清空对话记录」会先请您确认，再清除本轮聊天与任务进度。这里还有「📖  使用说明」和「ℹ️  关于银龄智办」。

## 六、四大功能怎么用

首页四张服务卡的文案和自动发送的话如下。这四句话含任务路由关键词，说明时请逐字照抄：

|卡片|副标题|点击后自动发出|
|---|---|---|
|看医生|挂号、就医、复诊|我明天下午想去医院看医生，帮我安排一下。|
|买东西|买菜、买药|帮我买菜。|
|看屏幕|教您一步步操作|我不知道页面操作下一步点哪里，请一步一步教我。|
|防诈骗|辨别可疑信息|帮我看看是不是诈骗。|

### 事务规划与普通问答

点「看医生」或「买东西」，或到「对话」页直接描述要办的事，助手会规划步骤并按追问继续。其他问题可直接在「对话」页提问，由智能问答处理。

### 截图指导

在「对话」页点输入框的「+」，选择 PNG、JPG 或 JPEG 截图，再描述目标，例如“我要进入蓝牙设置”。根据红框和文字提示，在手机实际设置页面操作；操作后返回网页上传新截图继续。如果目标已经完成，助手可能直接确认完成，而不再画红框。

这是上传截图后的指导，不会自动控制手机，也不会自动读取手机当前页面。

### 防诈骗核查

点「防诈骗」，或在「对话」页发送可疑短信、链接、话术或相关截图。系统给出风险判断与文字提醒；遇到高风险信息，请先停止付款、转账或提供个人资料，再通过可信渠道核实。

### 语音输入与回答朗读

在允许麦克风的访问环境中，到「对话」页点输入框内的麦克风，允许录音，说出需求并结束录音，然后按输入框提示发送。首次使用需要加载本地 Whisper small 模型，可能等待较久。

Whisper 按模型名称加载时可能首次下载模型；模型文件不包含在 `requirements.txt` 中。首次演示前应联网预热语音识别。当前代码用 CPU 运行，不需要为此安装 CUDA。[faster-whisper 官方说明](https://github.com/SYSTRAN/faster-whisper)。

要朗读新回答，先到「我的」页打开「🔊 朗读回答」，再发送消息。生成音频后可以播放；浏览器阻止自动播放时，手动点击播放器的播放按钮。朗读使用在线 edge-tts 服务，需要联网。它与麦克风权限是不同功能；网络或音频生成失败时，优先阅读文字回答。

## 七、适老化设计

界面使用暖米白与暖橙配色；字号有标准 1.00、大 1.15、特大 1.30 三档。按钮触控热区以不小于 48px 为底线，主要操作通常不小于 52px，首页服务卡在小屏上也不低于 100px。页面会先压缩装饰和留白；内容放不下时允许纵向滚动，不裁掉内容。

## 八、在手机上打开网页

手机通过浏览器使用电脑运行的程序，不需要安装 Python 或 Streamlit。

### 1. 手机和电脑连接同一个可互相访问的网络

例如都连接家里的同一个Wi-Fi。校园网、公司网、访客Wi-Fi即使名称相同，也可能隔离设备，导致无法互相访问。

### 2. 电脑允许局域网访问

如果程序已经运行，先在原PowerShell窗口按 `Ctrl+C`，再执行：

```powershell
cd D:\elderly_agent
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address=0.0.0.0 --server.port=8501
```

`0.0.0.0` 是服务监听设置，不是手机浏览器要输入的网址。电脑本机仍可访问 `http://localhost:8501`。

若Windows弹出网络访问提示，在自己信任的专用网络上允许该Python程序访问。无需关闭防火墙或禁用Streamlit的跨站请求保护。

### 3. 查询电脑当前IPv4地址

另开一个PowerShell窗口执行：

```powershell
ipconfig
```

找到正在连接网络的适配器：

- 无线连接通常看“无线局域网适配器 WLAN”。
- 网线连接看实际使用的“以太网适配器”。
- 不要选VirtualBox、VMware、未连接的适配器。

假设当前IPv4地址为 `192.168.1.23`。

### 4. 手机浏览器输入网址

```text
http://192.168.1.23:8501
```

把示例IP换成刚刚查到的电脑IPv4。换Wi-Fi或重连网络后，IP可能变化，需要重新查询。

手机上不要输入 `localhost` 或 `127.0.0.1`，那指向手机自己。也不要输入 `0.0.0.0`。

### 5. 手机打不开时

按顺序检查：

1. 电脑的 `http://localhost:8501` 能否打开，终端是否还在运行。
2. 启动命令是否带 `--server.address=0.0.0.0`。
3. 当前电脑IP、手机网址和端口是否一致。
4. 手机是否实际使用同一Wi-Fi，是否切换到了移动数据或其他网络。
5. Windows防火墙是否允许当前Python程序在对应可信网络上通信。
6. 若是校园网或访客网络，改用允许设备互访的家庭网络；也可尝试电脑连接手机热点，再查询电脑的新IP。热点能否互访取决于设备设置。

局域网网址只能供可访问该网络的设备使用，不是评委在外网能打开的公网链接。跨网络访问需要另外部署。

## 九、手机网页能打开，但不能录音怎么办

**页面访问和麦克风录音是两回事。**

- 使用 `http://电脑IP:8501` 访问时，可以测试文字输入、图片上传和页面展示。
- 浏览器麦克风通常要求安全上下文。电脑上的 `http://localhost` 可作为本机例外；手机访问电脑的普通HTTP IP地址通常不满足要求。
- 若要在手机上完整使用网页录音，需要通过配置正确、浏览器信任的HTTPS地址访问，并允许网站使用麦克风。
- 只把网址中的 `http` 手动改成 `https` 不会生效；服务端或部署平台必须实际配置HTTPS。
- 在普通HTTP局域网测试时，可以先使用手机输入法自带的语音输入转成文字，再发送。这只能替代文字输入，不算验证本项目的Whisper语音识别。

上述录音限制来自浏览器安全要求，不是缺少某个pip包。[MDN麦克风访问说明](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia)、[Streamlit浏览器权限说明](https://docs.streamlit.io/knowledge-base/using-streamlit/enable-camera)。

## 十、常见启动问题

|问题|处理方法|
|---|---|
|找不到app.py|进入实际包含app.py的目录后启动|
|No module named faster_whisper / rapidocr|使用同一个虚拟环境Python安装完整requirements.txt，再用它启动|
|页面提示Streamlit版本过旧|按本清单安装；此版使用accept_audio和autoscroll，不能任意换成旧版本|
|缺少API密钥或模型不可用|检查.env路径、三个变量及账号权限；修改后重启服务|
|第一次语音很慢|等待模型首次下载和加载，提前做一次语音预热|
|截图识别提示OCR不可用|检查rapidocr、onnxruntime安装及终端模型加载信息；首次模型准备可能需要网络|
|端口8501已占用|停止自己之前启动的实例，或将启动命令端口改成8502；访问网址也改成8502|
|虚拟环境提示Unable to create process|通常需用本机已安装的Python重新创建环境；先保留旧环境，勿直接删除项目|
|手机只能打开页面不能录音|查看上面的HTTPS与麦克风说明|
|电脑上看不到右侧 Agent 面板|窗口宽度 <1280px 时样式自动隐藏，把窗口拉宽或最大化；用 `app.py` 则本来就没有面板|
|点首页服务卡后自己跳到对话页并开始回答|这是设计行为：点卡片会替您把需求发出去，不是误触|
|手机上右侧面板/电脑布局错乱|手机竖屏是设计目标形态；面板只在电脑宽屏出现，手机不会显示|

Streamlit相关参数可参考：[聊天输入](https://docs.streamlit.io/1.63.0/develop/api-reference/chat/st.chat_input)、[服务配置](https://docs.streamlit.io/develop/api-reference/configuration/config.toml)。

## 十一、最常用命令

电脑自己使用产品版：

```powershell
cd D:\elderly_agent
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port=8501
```

电脑宽屏展示 Agent 运行面板：

```powershell
cd D:\elderly_agent
.\.venv\Scripts\python.exe -m streamlit run app_with_Agent_workspace.py --server.port=8502
```

电脑和同一网络的手机一起使用产品版：

```powershell
cd D:\elderly_agent
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address=0.0.0.0 --server.port=8501
```

停止：回到运行程序的 PowerShell 窗口，按 `Ctrl+C`。
