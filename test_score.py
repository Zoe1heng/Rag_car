# coding=utf-8
import json
import sys
import re
import numpy as np
from text2vec import SentenceModel, semantic_search, Similarity
from config import EMBEDDING_DEVICE


simModel_path = './pre_train_model/text2vec-base-chinese'  # 相似度模型路径
simModel = SentenceModel(model_name_or_path=simModel_path, device=EMBEDDING_DEVICE)

def calc_jaccard(list_a, list_b, threshold=0.3):
    size_a, size_b = len(list_a), len(list_b)
    list_c = [i for i in list_a if i in list_b]
    size_c = len(list_c)
    score = size_c / (size_b + 1e-6)
    if score > threshold:
        return 1
    else:
        return 0


def report_score(gold_path, predict_path, field="answer_4", verbose=True):
    gold_info = json.load(open(gold_path))
    pred_info = json.load(open(predict_path))

    idx = 0
    for gold, pred in zip(gold_info, pred_info):
        question = gold["question"]
        keywords = gold["keywords"]
        gold = gold["answer"].strip()
        pred = pred[field].strip()
        if gold == "无答案" and pred != gold:
            score = 0.0
        elif gold == "无答案" and pred == gold:
            score = 1.0
        else:
            semantic_score = semantic_search(simModel.encode([gold]), simModel.encode(pred), top_k=1)[0][0]['score']
            join_keywords = [word for word in keywords if word in pred]
            keyword_score = calc_jaccard(join_keywords, keywords)
            score = 0.5 * keyword_score + 0.5 * semantic_score
        gold_info[idx]["score"] = score
        gold_info[idx]["predict"] = pred 
        idx += 1
        if verbose:
            print(f"预测: {question}, 得分: {score}")

    return gold_info


if __name__ == "__main__":
    '''
      online evaluation
    '''

    # 标准答案路径
    gold_path = "./data/gold.json" 
    print("Read gold from %s" % gold_path)

    # 预测文件路径
    predict_path = sys.argv[1] if len(sys.argv) > 1 else "./data/result.json"
    print("Read predict file from %s" % predict_path)

    # 四种答案的对比: 1=向量+BM25合并, 2=仅BM25, 3=仅向量, 4=多路召回+重排
    names = {"answer_1": "向量+BM25合并", "answer_2": "仅BM25", "answer_3": "仅向量", "answer_4": "多路召回+重排"}
    print("\n各方案得分对比:")
    for field, name in names.items():
        s = np.mean([item["score"] for item in report_score(gold_path, predict_path, field, verbose=False)])
        print(f"  {field} ({name}): {s:.4f}")

    results = report_score(gold_path, predict_path, "answer_4")

    # 输出最终得分
    final_score = np.mean([item["score"] for item in results])
    print("\n")
    print("="*100)
    print(f"预测问题数：{len(results)}, 预测最终得分：{final_score}")
    print("="*100)

    # 结果文件路径
    metric_path = "./data/metrics.json" 
    results_info = json.dumps(results, ensure_ascii=False, indent=2)
    with open(metric_path, "w") as fd:
        fd.write(results_info)
    print(f"\n结果文件保存至{metric_path}")

