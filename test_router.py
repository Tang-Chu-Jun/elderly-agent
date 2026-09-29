from agents.router import route_task


tests = [
    "我收到一个短信让我转账",
    "这个页面下一步点哪里",
    "我明天下午想去医院",
    "今天天气怎么样"
]


for text in tests:
    print(text, "->", route_task(text))