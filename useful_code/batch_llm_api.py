# -*- coding: utf-8 -*-
# File: call_qwen_plus.py 

"""
该文件是vllm_model.py直接批量调用商业大模型api的版本，这里采用的是阿里的qwen-plus，对于豆包，deepseek的api同理。
"""
 

import os
import torch
import time
from dashscope import Generation
from datetime import datetime
import random
import json
import requests
import dashscope
from concurrent.futures import ThreadPoolExecutor


dashscope.api_key = "sk-xxx" # 去通义官网注册一个qwen-plus的api-key
executor = ThreadPoolExecutor(max_workers=10)    # 表示在这个线程池中同时运行的线程有10个线程


# 封装模型响应函数
def get_response(messages):
    response = Generation.call(
        model='qwen-plus',
        messages=messages,
        seed=random.randint(1, 10000),
        result_format='message'
    )
    return response

 
def call_with_messages(query):
    messages = []
    messages.append(
            {
                "content": query, 
                "role": "user"
            }
    )
    
    first_response = get_response(messages)
    assistant_output = first_response.output.choices[0].message["content"]
    return assistant_output


class ChatLLM():

    # 并发批量推理，输入一个batch，返回一个batch的答案
    def infer(self, prompts):
       #batch_response = []
       tasks = [executor.submit(call_with_messages, (prompt)) for prompt in prompts]
       batch_response = [task.result() for task in tasks]
       #for q in prompts:
       #    out = call_with_messages(q) 
       #    batch_response.append(out)
       return batch_response

if __name__ == "__main__":
    start = time.time()
    llm = ChatLLM()
    test = ["吉利汽车座椅按摩","吉利汽车语音组手唤醒","自动驾驶功能介绍"]
    generated_text = llm.infer(test)
    print(generated_text)
    end = time.time()
    print("cost time: " + str((end-start)/60))
