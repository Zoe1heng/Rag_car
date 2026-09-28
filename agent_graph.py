"""车主手册问答助手的多轮 LangGraph 工作流。

START -> router -+-> clarify -> rewrite -+-> retrieve -> judge_and_answer -+-> END
                 +-> rewrite ------------+                                 |
                 +-> unrelated -> END                      不够证据且未到重试上限 |
                                                                            v
                                                        refine_search -> retrieve（回到判断）
"""
import os
from typing import Annotated, Literal, Protocol, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field, model_validator

from api_llm import with_system

HISTORY_WINDOW = 6
TOP_K_RECALL = 15
TOP_K_RERANK = 6
MAX_SEARCH_ROUNDS = 2  # 初次检索 + 最多 1 次基于缺失信息的重新检索
MAX_EVIDENCE_CHARS = 4000  # 证据总长度上限，和 run.py 里其他路径的截断长度保持一致
UNRELATED_MESSAGE = "我只能回答与车辆手册相关的问题。"

CLEAR, UNCLEAR, UNRELATED = "clear_manual_query", "unclear_manual_query", "unrelated"

ROUTER_PROMPT = """你是车辆用户手册问答助手的意图路由器。请判断用户最新一条消息属于哪一类，必要时结合此前的对话。

类别：
- clear_manual_query：与本车的使用、保养、故障或事故处理有关（包括车机界面、账号、App 等功能），并且点明了要查的主题，可以直接在手册中检索。问得宽泛也算，比如问某个界面有哪些操作、遇到某类情形一般怎么处理。如果是追问，但结合对话能明确所指，也算 clear_manual_query。
- unclear_manual_query：与车辆有关，但没有点明要查的对象，看不出指的是哪个功能、部件或系统，且对话中也无法补全。只要主题已经点明，即使问得宽泛，也不属于这一类。
- unrelated：与车辆无关（闲聊、常识问答、其他话题）。

此前的对话：
{history}

用户最新消息：
{query}"""

REWRITE_PROMPT = """请把用户的请求改写成一条独立的、用于检索车辆用户手册的查询。
- 结合对话历史，补全指代和省略的内容。
- 如果用户给出了澄清回答，把它融入查询。
- 使用规范的术语，避免口语化说法和缩写。
- 只输出这条查询。

对话历史：
{history}

当前问题：
{query}

澄清回答：
{clarification}"""

ANSWER_PROMPT = """你是车辆用户手册问答助手。

手册摘录（已编号）：
{evidence}

用户最新消息：{query}
独立问题（仅供参考）：{rewritten}

请判断上面的摘录能否支撑回答这个问题：
- 摘录足够时：has_sufficient_evidence 填 true；answer 只依据摘录给出简洁回答，不要编造操作步骤；citations 填写你实际用到的摘录编号（从 1 开始，可以多个）。
- 摘录不够时：has_sufficient_evidence 填 false；answer 写一句如实的说明（例如手册中未找到相关内容），不要编造；missing_information 简要说明还缺什么信息，会用于换个角度重新检索。"""

REFINE_SEARCH_PROMPT = """上一次检索没能找到足够信息来回答用户的问题，请把检索查询换一个角度或换一种说法，以便找到手册里可能遗漏的相关内容。

原始问题：{query}
上一次检索用的查询：{previous_query}
缺失的信息：{missing_information}

只输出新的检索查询，不要输出其他内容。"""


class RouterDecision(BaseModel):
    """路由器的结构化输出：判断用户最新消息的意图。"""

    intent: Literal["clear_manual_query", "unclear_manual_query", "unrelated"] = Field(
        description="用户最新消息的类别"
    )
    missing_information: list[str] = Field(
        default_factory=list,
        description="只有 unclear_manual_query 才填写：检索所缺的关键信息；其余情况为空列表",
    )
    clarification_question: str = Field(
        default="",
        description="只有 unclear_manual_query 才填写：只问最关键的一个缺失信息，用一句简短的中文提问；其余情况为空字符串",
    )

    @model_validator(mode="after")
    def _unclear_needs_question(self):
        if self.intent == UNCLEAR and not self.clarification_question.strip():
            raise ValueError("unclear_manual_query 必须给出澄清问题")
        return self


class AnswerDecision(BaseModel):
    """回答节点的结构化输出：判断检索到的证据是否足够，并给出回答。"""

    has_sufficient_evidence: bool = Field(description="手册摘录是否足以回答用户的问题")
    answer: str = Field(description="给用户的回复；证据不足时也要给出如实的说明，不能为空")
    citations: list[int] = Field(
        default_factory=list,
        description="实际用到的摘录编号（从 1 开始）；只在 has_sufficient_evidence 为 true 时填写，不足时为空列表",
    )
    missing_information: str = Field(
        default="",
        description="只有 has_sufficient_evidence 为 false 时才填写：还缺什么信息，用于换个角度重新检索；证据足够时留空",
    )

    @model_validator(mode="after")
    def _consistency(self):
        if not self.answer.strip():
            raise ValueError("answer 不能为空")
        if self.has_sufficient_evidence and not self.citations:
            raise ValueError("has_sufficient_evidence 为 true 时必须给出至少一个引用编号")
        if not self.has_sufficient_evidence and not self.missing_information.strip():
            raise ValueError("has_sufficient_evidence 为 false 时必须说明缺什么信息")
        return self


class Retriever(Protocol):
    """检索器的最小接口约定。

    任何对象只要实现 retrieve(query) -> list[str]，就可以作为图里的检索节点。
    这样 LangGraph 工作流不需要关心底层用 BM25、FAISS、Chroma 还是别的检索器。
    """

    def retrieve(self, query: str) -> list[str]: ...


class AgentState(TypedDict, total=False):
    """LangGraph 在各个节点之间传递的状态。

    total=False 表示这些字段不要求一开始全部存在；节点会逐步写入字段。
    messages 使用 add_messages 聚合器，用于跨轮保存多轮对话消息。
    """

    messages: Annotated[list[AnyMessage], add_messages]
    history_len: int  # 本轮之前的消息条数

    user_query: str

    intent: str
    missing_information: list[str]
    clarification_question: str
    clarification_response: str

    rewritten_query: str  # 当前用于检索的查询，refine_search 会覆盖它
    initial_rewritten_query: str  # rewrite 节点结合上下文消歧后的独立问题，不受 refine_search 影响

    retrieved_docs: list[str]
    search_round: int  # 已经检索的次数，达到 MAX_SEARCH_ROUNDS 后不再重试

    has_sufficient_evidence: bool
    citations: list[int]  # 回答实际用到的 retrieved_docs 编号（从 1 开始）
    final_answer: str


def _history_text(messages: list[AnyMessage], history_len: int) -> str:
    """把最近几轮历史消息格式化成 prompt 可读的中文文本。"""

    recent = messages[max(0, history_len - HISTORY_WINDOW):history_len]
    lines = [f"{'用户' if isinstance(m, HumanMessage) else '助手'}: {m.content}" for m in recent]
    return "\n".join(lines) or "（无）"


def prepare_evidence(docs: list[str], query: str, rewritten: str) -> tuple[str, list[dict]]:
    """把检索到的文档整理成可以直接发给 LLM 的完整提示词，以及供 judge_and_answer 校验引用用的
    sources 列表（每条含编号 id 和原文 text；以后要接重排分数、页码这类元数据，也加在这里）。

    依次做：去重（防御性，retrieve 节点累加证据时已经按内容去重过一次）、按总长度截断
    （超过 MAX_EVIDENCE_CHARS 就不再往后加，保证至少留一条，避免累加多轮证据后无限变长）、
    编号、拼成 evidence 文本、套进 ANSWER_PROMPT 模板。这个函数是纯数据处理，不调用 LLM，
    也不依赖 retriever，方便单独复用（eval_answer_agent.py 批量评估时也是调这个函数）。
    """

    sources, total = [], 0
    for text in docs:
        if any(s["text"] == text for s in sources):
            continue
        if sources and total + len(text) > MAX_EVIDENCE_CHARS:
            break
        sources.append({"id": len(sources) + 1, "text": text})
        total += len(text)

    evidence = "\n\n".join(f"[{s['id']}] {s['text']}" for s in sources)
    prompt = ANSWER_PROMPT.format(evidence=evidence, query=query, rewritten=rewritten)
    return prompt, sources


def classify_intent(llm: BaseChatModel, messages: list[AnyMessage], fail_open: bool = True) -> RouterDecision:
    """判断最新一条用户消息的意图。

    结构化输出不合法时，fail_open=True 默认按 clear_manual_query 处理，
    避免误拒答或问出无意义的澄清问题；评估时用 fail_open=False 让这类失败暴露出来。
    """

    history_len = len(messages) - 1
    prompt = ROUTER_PROMPT.format(history=_history_text(messages, history_len), query=messages[-1].content)
    # function_calling 在 OpenAI 和 DeepSeek 上都可用；默认的 json_schema 方式 DeepSeek 不支持
    router = llm.with_structured_output(RouterDecision, method="function_calling")
    try:
        decision = router.invoke(with_system(prompt))
        if decision is None:
            raise ValueError("模型没有返回结构化结果")
        return decision
    except ValueError:
        if not fail_open:
            raise
        return RouterDecision(intent=CLEAR)


def build_graph(llm: BaseChatModel, retriever: Retriever, checkpointer=None):
    """构建车辆手册问答 LangGraph 工作流。

    参数：
    - llm：负责意图判断、问题改写、答案生成。
    - retriever：负责根据改写后的问题检索手册片段。
    - checkpointer：保存多轮对话状态；默认使用内存版 MemorySaver。

    工作流：
    router -> clarify/rewrite/unrelated -> retrieve -> answer。
    """

    def router(state: AgentState) -> dict:
        """意图路由节点。

        判断用户最新消息属于：
        - clear_manual_query：可以直接检索手册；
        - unclear_manual_query：与车辆有关但缺关键信息，需要澄清；
        - unrelated：与车辆手册无关，直接拒答。
        """

        messages = state["messages"]
        decision = classify_intent(llm, messages)

        # checkpointer 会跨轮保留状态，所以每轮开头要重置本轮字段
        return {
            "history_len": len(messages) - 1,
            "user_query": messages[-1].content,
            "intent": decision.intent,
            "missing_information": decision.missing_information,
            "clarification_question": decision.clarification_question.strip(),
            "clarification_response": "",
            "rewritten_query": "",
            "initial_rewritten_query": "",
            "retrieved_docs": [],
            "search_round": 0,
            "has_sufficient_evidence": False,
            "citations": [],
            "final_answer": "",
        }

    def route_after_router(state: AgentState) -> str:
        """根据 router 的 intent 决定下一步走哪个节点。"""
        if state["intent"] == UNCLEAR:
            return "clarify"
        elif state["intent"] == UNRELATED:
            return "unrelated"
        else:
            return "rewrite"

    def clarify(state: AgentState) -> dict:
        """澄清节点。

        使用 interrupt 暂停图执行，把澄清问题交给外部调用方。
        用户下一次输入会通过 Command(resume=...) 恢复执行，并作为澄清回答。
        """

        # 恢复执行时本节点会从头重跑，所以要等 interrupt() 返回后才更新状态
        response = interrupt({
            "clarification_question": state["clarification_question"],
            "missing_information": state["missing_information"],
        })
        return {
            "messages": [AIMessage(content=state["clarification_question"]), HumanMessage(content=response)],
            "clarification_response": response,
        }

    def rewrite(state: AgentState) -> dict:
        """问题改写节点。

        将多轮上下文、当前问题、澄清回答合并，改写成一条独立、规范、
        适合检索车辆手册的查询。
        """

        prompt = REWRITE_PROMPT.format(
            history=_history_text(state["messages"], state["history_len"]),
            query=state["user_query"],
            clarification=state.get("clarification_response") or "（无）",
        )
        rewritten = llm.invoke(with_system(prompt)).text.strip().strip("\"'`") or state["user_query"]
        return {"rewritten_query": rewritten, "initial_rewritten_query": rewritten}

    def retrieve(state: AgentState) -> dict:
        """检索节点。

        调用外部 retriever，把改写后的查询转换成若干手册证据片段。
        可能被 refine_search 重新触发；新检索到的证据会累加到已有证据后面（按内容去重），
        不会覆盖掉上一轮已经找到的内容——重试只会让证据变多，不会变少。
        """

        existing = state.get("retrieved_docs", [])
        new_docs = retriever.retrieve(state["rewritten_query"])
        return {
            "retrieved_docs": existing + [d for d in new_docs if d not in existing],
            "search_round": state.get("search_round", 0) + 1,
        }

    def judge_and_answer(state: AgentState) -> dict:
        """回答节点。

        用 prepare_evidence 把证据整理好之后，让 LLM 判断证据是否足以回答问题：
        - 足够：给出基于摘录的回答，并标注用到了哪几段摘录；
        - 不够：如实说明，并给出缺什么信息，供 refine_search 用来换个角度重新检索。
        结构化输出失败，或模型声称足够却没给出有效引用，都按"证据不足"处理（保守优先）。
        """

        prompt, sources = prepare_evidence(state["retrieved_docs"], state["user_query"], state["rewritten_query"])
        judge = llm.with_structured_output(AnswerDecision, method="function_calling")
        try:
            decision = judge.invoke(with_system(prompt))
            if decision is None:
                raise ValueError("模型没有返回结构化结果")
        except ValueError:
            decision = AnswerDecision(has_sufficient_evidence=False, answer="抱歉，暂时无法处理这个问题，请换个说法再试一次。",
                                      missing_information="模型未能返回有效的结构化结果")

        valid_ids = {s["id"] for s in sources}
        valid_citations = [c for c in decision.citations if c in valid_ids]
        sufficient = decision.has_sufficient_evidence and bool(valid_citations)
        return {
            "final_answer": decision.answer,
            "citations": valid_citations,
            "has_sufficient_evidence": sufficient,
            "missing_information": [decision.missing_information] if decision.missing_information.strip() else [],
            "messages": [AIMessage(content=decision.answer)],
        }

    def route_after_answer(state: AgentState) -> str:
        """证据足够，或已经达到重试上限，就结束；否则换个角度重新检索。"""

        if state["has_sufficient_evidence"] or state["search_round"] >= MAX_SEARCH_ROUNDS:
            return END
        return "refine_search"

    def refine_search(state: AgentState) -> dict:
        """重新检索前的查询改写节点。

        结合原始问题、上一次检索用的查询、judge_and_answer 指出缺什么信息，
        换一个角度或说法，希望这次能找到手册里遗漏的相关内容。
        """

        prompt = REFINE_SEARCH_PROMPT.format(
            query=state["user_query"],
            previous_query=state["rewritten_query"],
            missing_information="；".join(state.get("missing_information") or []) or "（未说明）",
        )
        refined = llm.invoke(with_system(prompt)).text.strip().strip("\"'`")
        return {"rewritten_query": refined or state["rewritten_query"]}

    def unrelated(state: AgentState) -> dict:
        """无关问题处理节点。

        对闲聊或非车辆手册问题返回固定拒答话术。
        """

        return {"final_answer": UNRELATED_MESSAGE, "messages": [AIMessage(content=UNRELATED_MESSAGE)]}

    # 定义状态图，并注册所有节点。
    graph = StateGraph(AgentState)
    graph.add_node("router", router)
    graph.add_node("clarify", clarify)
    graph.add_node("rewrite", rewrite)
    graph.add_node("retrieve", retrieve)
    graph.add_node("judge_and_answer", judge_and_answer)
    graph.add_node("refine_search", refine_search)
    graph.add_node("unrelated", unrelated)

    # 定义节点之间的跳转关系。
    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", route_after_router, ["clarify", "rewrite", "unrelated"])
    graph.add_edge("clarify", "rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "judge_and_answer")
    graph.add_conditional_edges("judge_and_answer", route_after_answer, ["refine_search", END])
    graph.add_edge("refine_search", "retrieve")
    graph.add_edge("unrelated", END)

    return graph.compile(checkpointer=checkpointer or MemorySaver())


def run_turn(graph, thread_id: str, user_text: str) -> dict:
    """处理一条用户消息；如果上一轮停在澄清问题上，就自动作为澄清回答恢复执行。"""

    config = {"configurable": {"thread_id": thread_id}}
    if graph.get_state(config).next:
        # 上一轮图停在 interrupt，说明系统正在等待用户澄清。
        graph.invoke(Command(resume=user_text), config)
    else:
        # 正常新一轮用户输入，从 messages 开始执行。
        graph.invoke({"messages": [HumanMessage(content=user_text)]}, config)

    snapshot = graph.get_state(config)
    if snapshot.next:
        # 图仍然处于 interrupt 状态，返回澄清问题给调用方。
        return {"type": "clarification", "text": snapshot.tasks[0].interrupts[0].value["clarification_question"]}
    state = snapshot.values
    return {
        "type": "unrelated" if state["intent"] == UNRELATED else "answer",
        "text": state["final_answer"],
        "rewritten_query": state["rewritten_query"],
        "initial_rewritten_query": state.get("initial_rewritten_query", state["rewritten_query"]),
        "citations": state.get("citations", []),
        "has_sufficient_evidence": state.get("has_sufficient_evidence", False),
        "search_round": state.get("search_round", 0),
    }


class ManualHybridRetriever:
    """沿用 run.py 里现有的混合检索：FAISS 与 BM25 各自召回，再用 bge 重排。"""

    def __init__(self, base: str = os.path.dirname(os.path.abspath(__file__))):
        """初始化车辆手册检索器。

        这里复用了项目 1 原有的 PDF 解析、FAISS、BM25、rerank 逻辑。
        初始化时会解析 train_a.pdf 并构建/加载检索索引，所以第一次启动会比较慢。
        """

        from bm25_retriever import BM25
        from faiss_retriever import FaissRetriever
        from pdf_parse import DataProcess
        from rerank_model import reRankLLM

        dp = DataProcess(pdf_path=os.path.join(base, "data/train_a.pdf"))
        for parse, size in (("ParseBlock", 1024), ("ParseBlock", 512),
                            ("ParseAllPage", 256), ("ParseAllPage", 512),
                            ("ParseOnePageWithRule", 256), ("ParseOnePageWithRule", 512)):
            getattr(dp, parse)(max_seq=size)

        self.faiss = FaissRetriever(os.path.join(base, "pre_train_model/m3e-large"), dp.data)
        self.bm25 = BM25(dp.data)
        self.rerank = reRankLLM(os.path.join(base, "pre_train_model/bge-reranker-large"))

    def retrieve(self, query: str) -> list[str]:
        """执行混合检索。

        1. FAISS 召回 TOP_K_RECALL 个语义相关片段；
        2. BM25 召回 TOP_K_RECALL 个关键词相关片段；
        3. 合并候选后用 bge-reranker 精排；
        4. 返回前 TOP_K_RERANK 个片段的纯文本。
        """

        faiss_docs = [doc for doc, _ in self.faiss.GetTopK(query, TOP_K_RECALL)]
        bm25_docs = self.bm25.GetBM25TopK(query, TOP_K_RECALL)
        ranked = self.rerank.predict(query, faiss_docs + bm25_docs)[:TOP_K_RERANK]
        return [doc.page_content for doc in ranked]


if __name__ == "__main__":
    from api_llm import get_llm

    app = build_graph(get_llm(), ManualHybridRetriever())
    while True:
        text = input("你: ").strip()
        if not text:
            break
        reply = run_turn(app, "cli", text)
        print(f"助手 [{reply['type']}]: {reply['text']}")
