#!/usr/bin/env python
# coding: utf-8


from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from pdf_parse import DataProcess
from config import EMBEDDING_DEVICE
import hashlib
import os
import torch


class FaissRetriever(object):
    # 初始化文档块索引，然后插入faiss库
    def __init__(self, model_path, data, cache_dir="./faiss_cache"):
        self.embeddings  = HuggingFaceEmbeddings(
                               model_name = model_path,
                               model_kwargs = {"device": EMBEDDING_DEVICE},
                               encode_kwargs = {"batch_size": 16}  # Mac统一内存: 批量64会吃满内存导致卡死
                           )
        docs = []
        for idx, line in enumerate(data):
            line = line.strip("\n").strip()
            words = line.split("\t")
            docs.append(Document(page_content=words[0], metadata={"id": idx}))
        # 生成embeddings的时间较长，按文档块内容的哈希缓存索引，分块结果不变就直接读盘
        key = hashlib.md5("\n".join(data).encode("utf-8")).hexdigest()[:12]
        index_path = os.path.join(cache_dir, key)
        if os.path.exists(index_path):
            self.vector_store = FAISS.load_local(index_path, self.embeddings, allow_dangerous_deserialization=True)
        else:
            self.vector_store = FAISS.from_documents(docs, self.embeddings)
            self.vector_store.save_local(index_path)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # 获取top-K分数最高的文档块
    def GetTopK(self, query, k):
       context = self.vector_store.similarity_search_with_score(query, k=k)
       return context

    # 返回faiss向量检索对象
    def GetvectorStore(self):
        return self.vector_store

if __name__ == "__main__":
    base = "."
    model_name = base + "/pre_train_model/m3e-large" #text2vec-large-chinese
    dp =  DataProcess(pdf_path = base + "/data/train_a.pdf")
    dp.ParseBlock(max_seq = 1024)
    dp.ParseBlock(max_seq = 512)
    print(len(dp.data))
    dp.ParseAllPage(max_seq = 256)
    dp.ParseAllPage(max_seq = 512)
    print(len(dp.data))
    dp.ParseOnePageWithRule(max_seq = 256)
    dp.ParseOnePageWithRule(max_seq = 512)
    print(len(dp.data))
    data = dp.data

    faissretriever = FaissRetriever(model_name, data)
    faiss_ans = faissretriever.GetTopK("如何预防新冠肺炎", 6)
    print(faiss_ans)
    faiss_ans = faissretriever.GetTopK("交通事故如何处理", 6)
    print(faiss_ans)
    faiss_ans = faissretriever.GetTopK("吉利集团的董事长是谁", 6)
    print(faiss_ans)
    faiss_ans = faissretriever.GetTopK("吉利汽车语音组手叫什么", 6)
    print(faiss_ans)
