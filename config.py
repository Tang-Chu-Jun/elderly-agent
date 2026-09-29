import os
from dotenv import load_dotenv

# 读取 .env 文件
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

OPENAI_BASE_URL = os.getenv(
    "OPENAI_BASE_URL",
    "https://api.deepseek.com/v1"
)

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "deepseek-v4-flash-vision-exp"
)

