import os

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

SYSTEM_PROMPT = "You are a helpful assistant."


def with_system(prompt: str) -> list[BaseMessage]:
    """给提示加上 system 消息。不带它，同样的提示在"仅向量"路径上得分系统性低约 2 个点（实测）"""
    return [SystemMessage(SYSTEM_PROMPT), HumanMessage(prompt)]


def get_llm(provider: str | None = None) -> BaseChatModel:
    """按 .env 里的 LLM_PROVIDER 返回 LangChain 聊天模型, openai 与 deepseek 一行切换"""
    provider = (provider or os.getenv("LLM_PROVIDER", "openai")).lower()
    if provider not in ("openai", "deepseek"):
        raise ValueError(f"LLM_PROVIDER 只支持 openai 或 deepseek, 当前是: {provider}")
    key_name = f"{provider.upper()}_API_KEY"
    if not os.getenv(key_name):
        raise RuntimeError(f"没有找到 {key_name}, 请在 .env 中填写")

    common = dict(temperature=0, max_tokens=1024, max_retries=3)
    if provider == "openai":
        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), **common)
    return ChatDeepSeek(model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
                        api_base=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"), **common)


def provider_of(llm: BaseChatModel) -> str:
    return "deepseek" if isinstance(llm, ChatDeepSeek) else "openai"


if __name__ == "__main__":
    llm = get_llm()
    print(f"provider={provider_of(llm)} model={llm.model_name}")
    print(llm.invoke("用一句话介绍一下你自己。").text)
