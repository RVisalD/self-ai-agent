# Connect your existing Paragon RAG index

Your original RAG stores vectors with `faiss.read_index("data/piu.index")` and loads chunk dictionaries using `pickle.load("data/piu_chunks.pkl")`. That **is not** LangChain's `FAISS.save_local` format. This project preserves your raw FAISS + lexical BM25-like retrieval and RRF ranking (with neighboring-chunk expansion) rather than using `FAISS.load_local`.

## 1. Copy the two files

Place these *original* files here:

```
everyday_groq_agent/
  paragon_directory/
    paragon_store/
      piu.index
      piu_chunks.pkl
    paragon_search.py
    paragon_tool.py
  agent.py
  app.py
```

You can instead set `PARAGON_INDEX_PATH` and `PARAGON_CHUNKS_PATH` in `.env` to absolute paths on your Mac. Both files must belong to the SAME index build; the number/order of vectors and chunks must match.

**Security:** `.pkl` files can run arbitrary code while unpickling. Only load the `.pkl` file you personally created and trust. Do not commit or publish private university data, or expose it to unauthorized users.

## 2. Install

```bash
cd /Users/mac/Documents/projects/python/langchain/everyday_groq_agent
source ../.venv/bin/activate
python -m pip install -r requirements.txt
```

These commands use the existing virtual environment at `langchain/.venv`. For a new environment, follow [README.md](README.md#install--run) instead. Run the remaining commands from `everyday_groq_agent/`.

The current index uses `sentence-transformers/all-MiniLM-L6-v2` to embed search questions, matching the model used by your original code. The embedding calculation happens locally. The code does not load Qwen or Transformers for text generation; Groq supplies the answer. The first Paragon query may take longer while SentenceTransformer downloads the model if it has not been cached.

## 3. Test retrieval directly (no Groq required)

```bash
python -c "from paragon_directory.paragon_search import search, format_context; print(format_context(search('What degree programs does Paragon offer?'))[:3000])"
```

Then run the app:

```bash
python -m streamlit run app.py
```

Ask: `What programs does Paragon offer?` and then `What about their tuition?`.

## 4. What was reused

* Raw FAISS vector search with normalized MiniLM query embeddings
* BM25-like lexical keyword search over title and text
* Reciprocal Rank Fusion and leadership/department keyword boosts
* Adaptive `top_k` (5, 6, 15)
* Original MIN_SCORE threshold of 0.30
* Adjacent chunk expansion with `NEIGHBOR_WINDOW=1`

The existing agent (LangGraph checkpointer) retains chat history. Groq should turn referential follow-ups into standalone queries when calling `search_paragon`.

## 5. Important accuracy notes

* `MIN_SCORE=0.30` in the original code assumes that *higher* FAISS scores indicate better semantic matches. This is appropriate for inner-product indexes but **not** for L2-distance indexes. If your index metric is L2, review and retune that original threshold before relying on it; lexical/RRF fusion still works.
* Passing source documents through a hosted Groq API sends the selected university snippets to that service. Confirm data sharing is acceptable and avoid sensitive/private content without authorization.
* `search_paragon` is chosen by the LLM. A prompt makes it likely, but does **not** guarantee every Paragon query is routed correctly. Add deterministic routing if your application needs an absolute guarantee.
* The model generates from retrieved evidence, but you should still verify citations and critical facts. Older documents may not reflect updated tuition or policies.
