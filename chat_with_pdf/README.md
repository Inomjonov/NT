# Chat with your PDF: Agentic RAG

Upload PDFs, ask questions, and get answers with page citations. Everything runs locally
with open-source Hugging Face models: no API keys, no cost.

**Stack:** LangChain + LangGraph (agent) · FAISS (vector search) · Hugging Face transformers
and sentence-transformers on PyTorch (models) · Streamlit (UI)

## Plain RAG vs. agentic RAG

Plain RAG always does the same thing: *search → stuff the results into the prompt → answer*.

In **agentic** RAG, an LLM agent makes decisions along the way:

```
          question
             │
          [agent] ── small talk ──► reply directly ──► END
             │ calls search_pdf(query)
         [retrieve] ◄──────────────┐
             │                     │
          [grade] ── not relevant ─► [rewrite]   (retry with a better query)
             │ relevant
          [answer] ──► END
```

1. **agent**: decides whether the question needs the PDF at all, and writes the search query
   (for follow-ups like "why is *it* faster?", it puts the real topic into the query).
2. **retrieve**: the `search_pdf` tool finds the most similar chunks in the FAISS index.
3. **grade**: the LLM checks whether those chunks can actually answer the question.
4. **rewrite**: if not, the LLM rephrases the query and searches again.
5. **answer**: the LLM answers only from the retrieved chunks and cites pages.

## Files

| File | What it does |
|---|---|
| [app.py](app.py) | Streamlit UI: file upload, chat, live agent steps, sources |
| [agent.py](agent.py) | The LangGraph agent: nodes, prompts, and the `search_pdf` tool |
| [ingest.py](ingest.py) | PDF → pages → chunks → embeddings → FAISS index |
| [llm.py](llm.py) | Runs a local Hugging Face model with PyTorch as a LangChain chat model with tool calling |
| [config.py](config.py) | Model names, chunk sizes, top-k, device selection |

## Run

```bash
cd NT/chat_with_pdf
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads the models: `BAAI/bge-small-en-v1.5` for embeddings (~130 MB) and
`Qwen/Qwen2.5-1.5B-Instruct` as the LLM (~3 GB). It uses a CUDA GPU or Apple Silicon (MPS)
when one is available, and falls back to the CPU otherwise.

## Things to try

- Say **"hi"**: the agent replies without searching.
- Ask something the PDF answers: watch *search → grade → answer* in the agent steps and check the sources.
- Ask a follow-up that uses "it" or "that": look at the search query the agent writes.
- Ask something the PDF does **not** cover: the grader rejects the passages, the agent rewrites
  the query, and the answer says it could not find it.
- Switch to `Qwen2.5-3B-Instruct` in the sidebar and compare the answers.
- Change `CHUNK_SIZE`, `TOP_K`, or `MAX_REWRITES` in [config.py](config.py).
