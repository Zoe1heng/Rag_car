"""比较"检索到什么就直接回答"(run.py 的 answer_4) 和"判断证据够不够、不够就换个角度重新检索"
这两种做法，在同一份 103 题测试集上的最终答案质量差多少。

检索用的是和 answer_4 完全相同的流程（FAISS+BM25 各召回 15、bge-reranker 精排取前 6），
第一轮检索到的文档应当和 run.py 跑出来的一致；唯一的变量是回答这一步：加了 AnswerDecision
结构化判断，证据不足时换个角度重新检索（最多 MAX_SEARCH_ROUNDS 轮，逻辑与 agent_graph.py 一致）。

跳过了 agent_graph.py 里的路由/澄清：这 103 题都是问清楚的完整问题，不需要澄清；路由准确率
已经由 eval_router.py 单独评估过，混进这里会分不清分数变化是路由的功劳还是这里新加的判断+
重试的功劳。

用法：python eval_answer_agent.py
"""
import json
import os
import warnings

import numpy as np
from tqdm import tqdm

from agent_graph import AnswerDecision, MAX_SEARCH_ROUNDS, REFINE_SEARCH_PROMPT, ManualHybridRetriever, prepare_evidence
from api_llm import get_llm, provider_of, with_system
from test_score import report_score

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(BASE, "data/result_answer_agent.json")


def judge(llm, query, current_query, docs):
    """给定原始问题、本轮实际用于检索的查询、召回并重排好的文档，判断证据够不够并给出回答。

    证据的整理（去重/截断/编号/拼 prompt）复用 agent_graph.py 的 prepare_evidence，
    保证批量评估和真实图里的行为完全一致。"""

    prompt, sources = prepare_evidence(docs, query, current_query)
    judge_llm = llm.with_structured_output(AnswerDecision, method="function_calling")
    try:
        decision = judge_llm.invoke(with_system(prompt))
        if decision is None:
            raise ValueError("模型没有返回结构化结果")
    except ValueError:
        decision = AnswerDecision(has_sufficient_evidence=False, answer="抱歉，暂时无法处理这个问题，请换个说法再试一次。",
                                  missing_information="模型未能返回有效的结构化结果")
    valid_ids = {s["id"] for s in sources}
    valid_citations = [c for c in decision.citations if c in valid_ids]
    sufficient = decision.has_sufficient_evidence and bool(valid_citations)
    return decision, valid_citations, sufficient


def answer_with_retry(llm, retriever, query):
    """检索 -> 判断 -> 不够就换角度重新检索，最多 MAX_SEARCH_ROUNDS 轮，逻辑与 agent_graph.py 的
    retrieve / judge_and_answer / refine_search 一致，这里展开成一个函数方便批量跑。

    新一轮检索到的证据会累加到已有证据后面（按内容去重），不覆盖上一轮已经找到的内容。"""

    current_query = query
    docs = []
    for round_ in range(1, MAX_SEARCH_ROUNDS + 1):
        new_docs = retriever.retrieve(current_query)
        docs = docs + [d for d in new_docs if d not in docs]
        decision, citations, sufficient = judge(llm, query, current_query, docs)
        if sufficient or round_ >= MAX_SEARCH_ROUNDS:
            return {"answer_agent": decision.answer, "search_round": round_,
                    "has_sufficient_evidence": sufficient, "citations": citations}
        refine_prompt = REFINE_SEARCH_PROMPT.format(
            query=query, previous_query=current_query, missing_information=decision.missing_information or "（未说明）")
        current_query = llm.invoke(with_system(refine_prompt)).text.strip().strip("\"'`") or current_query


def main():
    llm = get_llm()
    print(f"模型: {provider_of(llm)}/{llm.model_name}")
    retriever = ManualHybridRetriever()

    questions = json.load(open(os.path.join(BASE, "data/test_question.json"), encoding="utf-8"))
    results = []
    for line in tqdm(questions):
        results.append({"question": line["question"], **answer_with_retry(llm, retriever, line["question"])})
    json.dump(results, open(OUT_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"结果已保存到 {OUT_PATH}")

    scored = report_score(os.path.join(BASE, "data/gold.json"), OUT_PATH, "answer_agent", verbose=False)
    agent_score = float(np.mean([s["score"] for s in scored]))
    rounds = [r["search_round"] for r in results]
    print(f"\n判断+重试(answer_agent) 均分: {agent_score:.4f}  (103 题)")
    print(f"一次检索就通过: {rounds.count(1)}/{len(rounds)}  重试了一次: {rounds.count(2)}/{len(rounds)}")
    print(f"重试后仍判定证据不足的题数: {sum(1 for r in results if not r['has_sufficient_evidence'])}")

    baseline_path = os.path.join(BASE, "data/result_deepseek.json")
    if os.path.exists(baseline_path):
        base_scored = report_score(os.path.join(BASE, "data/gold.json"), baseline_path, "answer_4", verbose=False)
        base_score = float(np.mean([s["score"] for s in base_scored]))
        print(f"\n对照 - 直接回答，不判断不重试(answer_4) 均分: {base_score:.4f}")
        print(f"差值 (判断+重试 - 直接回答): {agent_score - base_score:+.4f}")


if __name__ == "__main__":
    main()
