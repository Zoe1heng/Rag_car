import os
import time
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))


class ChatLLM(object):
    """通过 OpenAI 兼容接口调用大模型, 接口与原来的本地 vLLM 版本一致: infer(prompts) -> answers"""

    def __init__(self, provider=None, max_workers=4):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "openai")).lower()
        if self.provider == "openai":
            api_key, base_url = os.getenv("OPENAI_API_KEY"), None
            self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        elif self.provider == "deepseek":
            api_key = os.getenv("DEEPSEEK_API_KEY")
            base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
            self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        else:
            raise ValueError(f"LLM_PROVIDER 只支持 openai 或 deepseek, 当前是: {self.provider}")
        if not api_key:
            raise RuntimeError(f"没有找到 {self.provider} 的 API key, 请在 .env 中填写")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.max_workers = max_workers

    def _chat(self, prompt):
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=1024,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)

    # 批量推理: 一个 batch 的 prompt 并发请求, 返回等长的答案列表
    def infer(self, prompts):
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return list(pool.map(self._chat, prompts))


if __name__ == "__main__":
    llm = ChatLLM()
    print(f"provider={llm.provider} model={llm.model}")
    print(llm.infer(["用一句话介绍一下你自己。"]))
