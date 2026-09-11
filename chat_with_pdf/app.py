"""Chat with your PDF: Streamlit UI.

Run with:  streamlit run app.py
"""
import streamlit as st
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

import config
from agent import build_agent
from ingest import build_index, load_embeddings, load_pdfs, split_pages
from llm import LocalChatModel

st.set_page_config(page_title="Chat with your PDF", page_icon="📄")


@st.cache_resource(show_spinner="Loading the embedding model...")
def get_embeddings():
    return load_embeddings()


@st.cache_resource(show_spinner="Loading the language model (the first run downloads it)...")
def get_llm(model_id: str):
    return LocalChatModel(model_id=model_id)


def describe_step(node: str, update: dict) -> str:
    """One line in the "agent steps" box for each graph node that finished."""
    if node == "grade":
        return "✅ Passages look relevant" if update["relevant"] else "❌ Passages don't look relevant"
    message = update["messages"][-1]
    if node == "retrieve":
        return f"📑 Found {len(message.artifact or [])} passages"
    if node == "answer":
        return "✍️ Wrote the answer"
    if message.tool_calls:
        verb = "Searching" if node == "agent" else "Searching again"
        return f"🔎 {verb}: *{message.tool_calls[0]['args'].get('query', '')}*"
    return "💬 Replied without searching"


def show_steps(steps: list[str]):
    with st.status(f"{len(steps)} agent steps", state="complete"):
        for step in steps:
            st.markdown(step)


def show_sources(sources):
    with st.expander(f"Sources ({len(sources)})"):
        for doc in sources:
            st.markdown(f"**{doc.metadata['source']}**, page {doc.metadata.get('page', 0) + 1}")
            st.caption(doc.page_content)


if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("📄 Documents")
    files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    st.header("⚙️ Settings")
    model_id = st.selectbox("Language model", config.LLM_MODELS)
    top_k = st.slider("Passages per search", 1, 10, config.TOP_K)
    if st.button("Clear chat"):
        st.session_state.messages = []

st.title("Chat with your PDF")
st.caption("Agentic RAG · LangChain + LangGraph · FAISS · Hugging Face · PyTorch")

if not files:
    st.info("Upload one or more PDFs in the sidebar to start chatting.")
    st.stop()

# Re-index only when the uploaded files change
files_key = [(f.name, f.size) for f in files]
if st.session_state.get("files_key") != files_key:
    with st.spinner("Reading and indexing your PDFs..."):
        pages = load_pdfs([(f.name, f.getvalue()) for f in files])
        chunks = split_pages(pages)
        if not chunks:
            st.error("No text found in these PDFs. Scanned PDFs need OCR first.")
            st.stop()
        st.session_state.index = build_index(chunks, get_embeddings())
    st.session_state.files_key = files_key
    st.session_state.index_info = f"Indexed {len(pages)} pages → {len(chunks)} chunks"

st.sidebar.success(st.session_state.index_info)
llm = get_llm(model_id)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("steps"):
            show_steps(message["steps"])
        st.markdown(message["content"])
        if message.get("sources"):
            show_sources(message["sources"])

if question := st.chat_input("Ask a question about your PDF"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    history = [
        HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"])
        for m in st.session_state.messages[-(config.MAX_HISTORY_MESSAGES + 1):]
    ]
    agent = build_agent(llm, st.session_state.index, top_k)

    with st.chat_message("assistant"):
        status = st.status("Thinking...", expanded=True)
        answer_box = st.empty()
        answer, steps, sources = "", [], []

        for mode, data in agent.stream({"messages": history, "rewrites": 0}, stream_mode=["updates", "messages"]):
            if mode == "messages":
                # Token-by-token output of the LLM; only the final answer is shown live
                chunk, metadata = data
                if metadata["langgraph_node"] == "answer" and isinstance(chunk, AIMessageChunk):
                    answer += chunk.content
                    answer_box.markdown(answer + "▌")
                continue

            # mode == "updates": a graph node finished
            for node, update in data.items():
                steps.append(describe_step(node, update))
                status.markdown(steps[-1])
                if node == "retrieve":
                    sources += update["messages"][-1].artifact or []
                if node in ("agent", "answer") and not update["messages"][-1].tool_calls:
                    answer = update["messages"][-1].content

        status.update(label=f"{len(steps)} agent steps", state="complete", expanded=False)
        answer_box.markdown(answer)
        sources = list({doc.page_content: doc for doc in sources}.values())  # drop duplicates
        if sources:
            show_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "steps": steps, "sources": sources}
    )
