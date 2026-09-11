"""Agentic RAG as a LangGraph state machine.

          question
             │
          [agent] ── small talk ──► reply directly ──► END
             │ calls search_pdf(query)
         [retrieve] ◄──────────────┐
             │                     │
          [grade] ── not relevant ─► [rewrite]   (at most MAX_REWRITES times)
             │ relevant
          [answer] ──► END
"""
from typing import Literal

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

import config
from llm import LocalChatModel

AGENT_PROMPT = """You are an assistant that answers questions about PDF documents the user uploaded.
You cannot see the documents directly: use the search_pdf tool to read them.

- For any question about the documents, call search_pdf with a short, standalone search query. \
If the question refers to earlier messages ("it", "that method"), put the actual topic in the query.
- Only for greetings or small talk, reply briefly without searching."""

GRADE_PROMPT = """You are checking search results for a question.

Search results:
{context}

Question: {question}

Do the search results contain information that helps answer the question? Reply with only "yes" or "no"."""

REWRITE_PROMPT = """A search over a PDF document did not find useful passages.

Question: {question}
Search query that failed: {query}

Write a better search query using different keywords or synonyms. Reply with only the new query."""

ANSWER_PROMPT = """You answer questions about the user's PDF documents using only the given context.
- Cite the pages you used, like (report.pdf, page 3).
- If the context does not contain the answer, say you could not find it in the documents. Never make up facts.
- Keep the answer clear and concise."""


class RAGState(MessagesState):
    rewrites: int  # how many times the search query was rewritten for this question
    relevant: bool  # did the grader accept the latest search results?


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{doc.metadata['source']}, page {doc.metadata.get('page', 0) + 1}]\n{doc.page_content}"
        for doc in docs
    )


def split_turn(messages: list[BaseMessage]) -> tuple[list[BaseMessage], HumanMessage, list[BaseMessage]]:
    """Split messages into (earlier chat, latest question, agent steps taken for that question)."""
    last = max(i for i, message in enumerate(messages) if isinstance(message, HumanMessage))
    return messages[:last], messages[last], messages[last + 1:]


def make_search_tool(index: FAISS, top_k: int):
    @tool(response_format="content_and_artifact", parse_docstring=True)
    def search_pdf(query: str) -> tuple[str, list[Document]]:
        """Search the uploaded PDF documents for passages relevant to a query.

        Args:
            query: A short, standalone search query, e.g. "size of the training dataset".
        """
        docs = index.similarity_search(query, k=top_k)
        return format_docs(docs), docs  # text for the LLM, Documents for showing sources

    return search_pdf


def build_agent(llm: LocalChatModel, index: FAISS, top_k: int = config.TOP_K):
    search_pdf = make_search_tool(index, top_k)
    llm_with_tools = llm.bind_tools([search_pdf])

    def agent(state: RAGState):
        """Decide: search the PDF, or reply directly."""
        response = llm_with_tools.invoke([SystemMessage(AGENT_PROMPT), *state["messages"]])
        return {"messages": [response]}

    def grade(state: RAGState):
        """Ask the LLM whether the passages just retrieved can answer the question."""
        _, question, _ = split_turn(state["messages"])
        prompt = GRADE_PROMPT.format(context=state["messages"][-1].content, question=question.content)
        verdict = llm.invoke([HumanMessage(prompt)], max_new_tokens=3)
        return {"relevant": verdict.content.strip().lower().startswith("yes")}

    def rewrite(state: RAGState):
        """Rephrase the failed search query and send it back to the search tool."""
        _, question, steps = split_turn(state["messages"])
        last_search = [m for m in steps if isinstance(m, AIMessage) and m.tool_calls][-1]
        failed_query = last_search.tool_calls[0]["args"].get("query", question.content)
        prompt = REWRITE_PROMPT.format(question=question.content, query=failed_query)
        lines = llm.invoke([HumanMessage(prompt)], max_new_tokens=32).content.strip().splitlines()
        new_query = lines[0].strip('" ') if lines else question.content

        rewrites = state.get("rewrites", 0) + 1
        search = AIMessage(
            content="",
            tool_calls=[{"name": "search_pdf", "args": {"query": new_query}, "id": f"rewrite_{rewrites}"}],
        )
        return {"messages": [search], "rewrites": rewrites}

    def answer(state: RAGState):
        """Write the final answer from every passage found for this question."""
        history, question, steps = split_turn(state["messages"])
        context = "\n\n".join(m.content for m in steps if isinstance(m, ToolMessage))
        prompt = f"Context from the documents:\n{context}\n\nQuestion: {question.content}"
        response = llm.invoke([SystemMessage(ANSWER_PROMPT), *history, HumanMessage(prompt)])
        return {"messages": [response]}

    def after_grade(state: RAGState) -> Literal["answer", "rewrite"]:
        if state["relevant"] or state.get("rewrites", 0) >= config.MAX_REWRITES:
            return "answer"
        return "rewrite"

    graph = StateGraph(RAGState)
    graph.add_node("agent", agent)
    graph.add_node("retrieve", ToolNode([search_pdf]))
    graph.add_node("grade", grade)
    graph.add_node("rewrite", rewrite)
    graph.add_node("answer", answer)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "retrieve", END: END})
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", after_grade)
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("answer", END)
    return graph.compile()
