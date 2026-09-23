# -*- coding: utf-8 -*-
# File: model_zoo.py 

"""
该文件是vllm_model.py用vllm batch调用其他本地模型的版本，2卡分布式部署，包括chatglm3, baichuan2, 可替换现有的qwen-7b模型，对于其他大模型，同理新增即可。
"""

import os
import torch
from config import *
from vllm import LLM, SamplingParams
import time
from transformers import AutoTokenizer
from transformers import GenerationConfig
from qwen_generation_utils import make_context, decode_tokens, get_stop_words_ids


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


DEVICE = LLM_DEVICE
IMEND = "<|im_end|>"
ENDOFTEXT = "<|endoftext|>"


# 释放gpu显存
def torch_gc():
    if torch.cuda.is_available():
        for device in os.environ["CUDA_VISIBLE_DEVICES"].split(","):
            with torch.cuda.device(f"{DEVICE}:{DEVICE_ID}"):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()


class ChatLLM(object):

    def __init__(self, model_path):
        self.model_path = model_path
        if "qwen" in self.model_path or "Qwen" in self.model_path:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                pad_token="<|extra_0|>",
                eos_token="<|endoftext|>",
                padding_side="left",
                trust_remote_code=True,
            )
            self.generation_config = GenerationConfig.from_pretrained(
                model_path, pad_token_id=self.tokenizer.pad_token_id
            )
            self.tokenizer.eos_token_id = self.generation_config.eos_token_id
            self.stop_words_ids = []

            # 加载vLLM大模型
            self.model = LLM(
                model=model_path,
                tokenizer=model_path,
                tensor_parallel_size=2,  # 如果是多卡，可以自己把这个并行度设置为卡数N，默认会采用张量并行来分布式部署
                trust_remote_code=True,
                gpu_memory_utilization=0.6,  # 可以根据gpu的显存利用自己调整这个比例, 假设GPU是24G显存，0.6就代表vllm模型用0.6*24G=14.4G，两张卡就用28.8G
                dtype="bfloat16",
            )
            for stop_id in get_stop_words_ids(
                self.generation_config.chat_format, self.tokenizer
            ):
                self.stop_words_ids.extend(stop_id)
            self.stop_words_ids.extend([self.generation_config.eos_token_id])

            # LLM的采样参数
            sampling_kwargs = {
                "stop_token_ids": self.stop_words_ids,
                "early_stopping": False,
                "top_p": 1.0,
                "top_k": (
                    -1
                    if self.generation_config.top_k == 0
                    else self.generation_config.top_k
                ),
                "temperature": 0.0,
                "max_tokens": 2000,
                "repetition_penalty": self.generation_config.repetition_penalty,
                "n": 1,
                "best_of": 2,
                "use_beam_search": True,
            }
            self.sampling_params = SamplingParams(**sampling_kwargs)
        elif "chatglm" in self.model_path or "ChatGLM" in self.model_path:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            self._model = LLM(
                model=self.model_path,
                tensor_parallel_size=2,
                trust_remote_code=True,
                max_model_len=1024,
                gpu_memory_utilization=0.6,
                enforce_eager=True,
                dtype="bfloat16",
            )
            stop_token_ids = [151329, 151336, 151338]
            self._sampling_params = SamplingParams(
                temperature=0.95, max_tokens=1024, stop_token_ids=stop_token_ids
            )
        elif "baichuan" in self.model_path or "Baichuan" in self.model_path:
            self._model = LLM(
                model=self.model_path,
                tensor_parallel_size=2,
                trust_remote_code=True,
                gpu_memory_utilization=0.6,
                dtype="bfloat16",
            )
        else:
            raise NotImplementedError(f"Unknown model {self.model_path!r}")

    def infer(self, prompts):
        if "qwen" in self.model_path or "Qwen" in self.model_path:
            return self.infer_qwen(prompts)
        elif "chatglm" in self.model_path or "ChatGLM" in self.model_path:
            return self.infer_chatglm(prompts)
        elif "baichuan" in self.model_path or "Baichuan" in self.model_path:
            return self.infer_baichuan(prompts)
        else:
            raise NotImplementedError(f"Unknown model {self.model_path!r}")

    # 批量推理，输入一个batch，返回一个batch的答案
    def infer_qwen(self, prompts):
        batch_text = []
        for q in prompts:
            raw_text, _ = make_context(
                self.tokenizer,
                q,
                system="You are a helpful assistant.",
                max_window_size=self.generation_config.max_window_size,
                chat_format=self.generation_config.chat_format,
            )
            batch_text.append(raw_text)
        outputs = self.model.generate(batch_text, sampling_params=self.sampling_params)
        batch_response = []
        for output in outputs:
            output_str = output.outputs[0].text
            if IMEND in output_str:
                output_str = output_str[: -len(IMEND)]
            if ENDOFTEXT in output_str:
                output_str = output_str[: -len(ENDOFTEXT)]
            batch_response.append(output_str)
        torch_gc()
        return batch_response

    def infer_baichuan(self, prompts):
        output = self._model.generate(
            prompts, sampling_params=SamplingParams(temperature=0.95, max_tokens=1024)
        )
        torch_gc()
        return [item.outputs[0].text for item in output]

    def infer_chatglm(self, prompts):
        output = self._model.generate(
            prompts=prompts, sampling_params=self._sampling_params
        )
        torch_gc()
        print(output)
        return [item.outputs[0].text for item in output]


if __name__ == "__main__":
    base = "."
    qwen_7b = base + "/model/Qwen-7B-Chat"
    chatglm3_6b = base + "/model/chatglm3-6b"
    baichuan2_7b = base + "/model/Baichuan2-7B-Chat"
    test = ["吉利汽车座椅按摩", "吉利汽车语音组手唤醒", "自动驾驶功能介绍"]

    start = time.time()
    llm_qwen_7b = ChatLLM(qwen_7b)
    qwen_7b_generated_text = llm_qwen_7b.infer(test)
    print(qwen_7b_generated_text)
    end = time.time()
    print("qwen7b cost time: " + str((end - start) / 60))

    start = time.time()
    chatglm3_6b = "/root/autodl-tmp/TinyRAG/model/chatglm3-6b"
    llm_chatglm3_6b = ChatLLM(chatglm3_6b)
    chatglm3_6b_generated_text = llm_chatglm3_6b.infer(test)
    print(chatglm3_6b_generated_text)
    end = time.time()
    print("chatglm3_6b cost time: " + str((end - start) / 60))

    start = time.time()
    llm_baichuan2_7b = ChatLLM(baichuan2_7b)
    baichuan2_7b_generated_text = llm_baichuan2_7b.infer(test)
    print(baichuan2_7b_generated_text)
    end = time.time()
    print("baichuan2_7b cost time: " + str((end - start) / 60))
