"""路由评估。

数据：
- 清晰 (clear_manual_query)：用代码从 gold.json 里取有答案的问题。
- 不想干 (unrelated)：gold.json 里没有答案的问题，加上 data/router_eval.json 里手写的无关问题。
- 不清晰 (unclear_manual_query)：data/router_eval.json 里把 gold 问题去掉关键信息得到的含糊问法，
  每条带用户的补充 clarification、期望的改写结果 rewritten_query（就是原来的完整问题）和关键词 keywords。

评估：
1. 三类标签判得对不对（准确率、混淆矩阵、每类精确率和召回率）。
2. 对不清晰的查询：用 clarification 回答追问后，改写出的查询是否包含关键词，
   以及和 rewritten_query 的语义相似度。

用法：python eval_router.py [--classify-only]
"""
import json
import os
import random
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from langchain_core.messages import HumanMessage

from agent_graph import CLEAR, UNCLEAR, UNRELATED, build_graph, classify_intent, run_turn
from api_llm import get_llm, provider_of
from config import EMBEDDING_DEVICE

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
LABELS = [CLEAR, UNCLEAR, UNRELATED]


class NoRetriever:
    def retrieve(self, query: str) -> list[str]:
        return []


def load_cases() -> tuple[list[dict], list[str]]:
    gold = json.load(open(os.path.join(BASE, "data/gold.json"), encoding="utf-8"))
    gold_questions = [g["question"] for g in gold]
    cases = [{"query": g["question"], "label": UNRELATED if g["answer"].strip() == "无答案" else CLEAR} for g in gold]

    for item in json.load(open(os.path.join(BASE, "data/router_eval.json"), encoding="utf-8")):
        if item["label"] == UNCLEAR:
            assert item["rewritten_query"] in gold_questions, f"rewritten_query 不在 gold.json 里: {item['rewritten_query']}"
            for keyword in item["keywords"]:
                assert keyword in item["rewritten_query"] and keyword in item["clarification"], \
                    f"关键词 {keyword!r} 必须同时出现在 rewritten_query 和 clarification 里: {item['query']}"
        cases.append(item)
    return cases, gold_questions


def evaluate_classification(llm, cases: list[dict]) -> None:
    def classify(case):
        try:
            return classify_intent(llm, [HumanMessage(case["query"])], fail_open=False).intent
        except ValueError:
            return None  # 结构化输出不合法

    with ThreadPoolExecutor(max_workers=8) as pool:
        preds = list(pool.map(classify, cases))
    for case, pred in zip(cases, preds):
        case["pred"] = pred

    correct = sum(c["pred"] == c["label"] for c in cases)
    print(f"\n{'=' * 70}\n一、三类标签判得对不对\n{'=' * 70}")
    print(f"总数 {len(cases)}, 正确 {correct}, 准确率 {correct / len(cases):.1%}, "
          f"结构化输出失败 {sum(c['pred'] is None for c in cases)}")

    columns = LABELS + [None]
    print(f"\n混淆矩阵 (行=真实, 列=预测)")
    print(f"{'':<22}" + "".join(f"{(c or '输出失败'):>24}" for c in columns))
    for label in LABELS:
        row = [sum(1 for c in cases if c["label"] == label and c["pred"] == col) for col in columns]
        print(f"{label:<22}" + "".join(f"{n:>24}" for n in row))

    print(f"\n{'类别':<22}{'样本数':>8}{'精确率':>10}{'召回率':>10}{'F1':>8}")
    f1s = []
    for label in LABELS:
        tp = sum(1 for c in cases if c["label"] == label and c["pred"] == label)
        n_true = sum(1 for c in cases if c["label"] == label)
        n_pred = sum(1 for c in cases if c["pred"] == label)
        precision, recall = tp / n_pred if n_pred else 0.0, tp / n_true if n_true else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        print(f"{label:<22}{n_true:>8}{precision:>10.1%}{recall:>10.1%}{f1:>8.2f}")
    print(f"宏平均 F1: {np.mean(f1s):.3f}")

    n_clear = sum(1 for c in cases if c["label"] == CLEAR)
    false_refusals = sum(1 for c in cases if c["label"] == CLEAR and c["pred"] == UNRELATED)
    print(f"误拒率 (清晰被判成不想干): {false_refusals}/{n_clear} = {false_refusals / n_clear:.1%}  <- 最伤用户体验")

    errors = [c for c in cases if c["pred"] != c["label"]]
    print(f"\n判错的 {len(errors)} 条:")
    for c in errors[:40]:
        print(f"  {c['query'][:32]}  真实={c['label']}  预测={c['pred']}")


def evaluate_rewrite(llm, cases: list[dict], gold_questions: list[str]) -> None:
    from text2vec import SentenceModel

    sim_model = SentenceModel(os.path.join(BASE, "pre_train_model/text2vec-base-chinese"), device=EMBEDDING_DEVICE)

    def similarity(a: str, b: str) -> float:
        va, vb = sim_model.encode([a, b])
        return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))

    def clarify_and_rewrite(args):
        index, case = args
        graph = build_graph(llm, NoRetriever())  # 每条用独立的图和记忆，互不影响
        thread = f"eval-{index}"
        first = run_turn(graph, thread, case["query"])
        if first["type"] != "clarification":
            return {"case": case, "asked": False, "first_type": first["type"]}
        final = run_turn(graph, thread, case["clarification"])
        return {"case": case, "asked": True, "question": first["text"], "rewritten": final["initial_rewritten_query"]}

    unclear = [c for c in cases if c["label"] == UNCLEAR]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(clarify_and_rewrite, enumerate(unclear)))

    asked = [r for r in results if r["asked"]]
    print(f"\n{'=' * 70}\n二、不清晰的查询：补充后改写得对不对\n{'=' * 70}")
    print(f"不清晰查询 {len(results)} 条, 系统追问了 {len(asked)} 条, 没追问的:")
    for r in results:
        if not r["asked"]:
            print(f"  {r['case']['query']}  -> {r['first_type']}")

    rng = random.Random(0)
    for r in asked:
        case = r["case"]
        r["hit"] = all(k in r["rewritten"] for k in case["keywords"])
        r["before"] = similarity(case["query"], case["rewritten_query"])
        r["after"] = similarity(r["rewritten"], case["rewritten_query"])
        r["random"] = similarity(rng.choice([q for q in gold_questions if q != case["rewritten_query"]]),
                                 case["rewritten_query"])

    hits = sum(r["hit"] for r in asked)
    print(f"\n改写结果包含关键词: {hits}/{len(asked)} = {hits / len(asked):.1%}")
    print(f"\n改写结果和期望改写的语义相似度 (text2vec, 越高越像):")
    print(f"  随机另一道题 (下限参照)    {np.mean([r['random'] for r in asked]):.3f}")
    print(f"  补充前 (含糊的原话)        {np.mean([r['before'] for r in asked]):.3f}")
    print(f"  补充后 (改写结果)          {np.mean([r['after'] for r in asked]):.3f}")

    misses = [r for r in asked if not r["hit"]]
    print(f"\n没包含关键词的 {len(misses)} 条:")
    for r in misses:
        case = r["case"]
        print(f"  含糊原话: {case['query']}  | 关键词: {case['keywords']}")
        print(f"    系统追问: {r['question']}")
        print(f"    用户补充: {case['clarification']}")
        print(f"    改写结果: {r['rewritten']}")
        print(f"    期望改写: {case['rewritten_query']}  (相似度 {r['after']:.3f})")


if __name__ == "__main__":
    llm = get_llm()
    print(f"模型: {provider_of(llm)}/{llm.model_name}")
    cases, gold_questions = load_cases()
    evaluate_classification(llm, cases)
    if "--classify-only" not in sys.argv:
        evaluate_rewrite(llm, cases, gold_questions)
