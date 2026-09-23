# -*- coding: utf-8 -*-
# File: ragas.py 

"""
该文件是RAG评估代码的扩展，利用Ragas框架来对问答系统输出的结果做评估。输入是query，生成的答案，参考答案，以及召回的上下文信息。
评估采用了精确率和召回率两个指标
"""
 

import json
from langchain_openai import ChatOpenAI
from ragas.metrics import LLMContextRecall, LLMContextPrecisionWithReference
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas import EvaluationDataset

# api_key是智增增的密钥，在这里注册：https://zhizengzeng.com/#/home

llm = ChatOpenAI(model="gpt-4o", api_key="sk-xxxx", base_url="https://api.zhizengzeng.com/")

gold = json.load(open("./data/gold.json"))
result = json.load(open("./data/result.json"))

dataset = []
for g, r in zip(gold, result):
    query = g["question"] # 输入问题
    reference = g["answer"] # 参考答案
    response = r["answer_5"] #生成的答案
    context = [r["answer_6"]] # 上下文
    dataset.append(
        {
            "user_input":query,
            "retrieved_contexts": context,
            "response":response,
            "reference":reference
        }
    )

evaluation_dataset = EvaluationDataset.from_list(dataset)
evaluator_llm = LangchainLLMWrapper(llm)

result = evaluate(dataset=evaluation_dataset,metrics=[LLMContextRecall(), LLMContextPrecisionWithReference()],llm=evaluator_llm)
print("评估结果：", result)
