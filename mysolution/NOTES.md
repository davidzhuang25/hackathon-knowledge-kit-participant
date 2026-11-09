# AI solution setup

This submission uses LangChain with OpenAI embeddings/chat and a persistent local Chroma database.
The selected LangChain 1.x dependencies require Python 3.10 or newer; the base harness itself supports Python 3.9.

From the package root:

```bash
source .venv/bin/activate
pip install -r mysolution/requirements.txt
python harness/run_eval.py --submission mysolution --eval eval/public_eval.jsonl
```

The solution loads `OPENAI_API_KEY` and optional LangSmith settings from the package-root `.env`
when it starts. Existing shell environment variables take precedence. `OPENAI_API_KEY` is required
for indexing and answers. The embedding model defaults to
`text-embedding-3-small`; the chat model defaults to `gpt-4.1-mini`. Override them with
`OPENAI_EMBEDDING_MODEL` or `OPENAI_CHAT_MODEL` if needed.

LangSmith tracing is optional. Set `LANGSMITH_TRACING=true` in `.env` along with
`LANGSMITH_API_KEY` to send LangChain runs to LangSmith. Traces can include the question and
retrieved page excerpts. For
short-lived scorer runs, set `LANGCHAIN_CALLBACKS_BACKGROUND=false` so trace callbacks finish
before the process exits. Disable tracing during final timing measurements if its overhead matters.

Chroma data is stored in `mysolution/.chroma/` and ignored by Git. It can be deleted to rebuild
the local index; the scorer's source snapshots remain unchanged.