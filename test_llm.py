from services.llm import chat_with_llm

result = chat_with_llm(
    "我收到一条短信，说医保账户异常，让我点击一个陌生链接，我该怎么办？"
)

print(result)

