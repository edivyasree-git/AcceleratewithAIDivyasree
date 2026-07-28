import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

m = ChatOpenAI(
    api_key=os.getenv("GITHUB_TOKEN"),
    base_url="https://models.github.ai/inference",
    model="openai/gpt-4.1-mini",
)

print(m.invoke("Say OK").content)