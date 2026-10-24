"""Paragon retrieval backed by the user's existing raw FAISS index and pickle chunks.

No Qwen model is loaded here: Groq handles generation in the agent.

IMPORTANT: Only load .pkl files you created and trust. Unpickling arbitrary data
can execute code. The FAISS index and chunk list must have matching row order.
"""
from __future__ import annotations

import math
import os
import pickle
import re
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

# Work around the macOS native-thread issue in the original Paragon script.
# Must be set before importing FAISS, numpy, or sentence-transformers.
if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parent
INDEX_PATH = Path(os.getenv("PARAGON_INDEX_PATH", str(ROOT / "paragon_store" / "piu.index")))
CHUNKS_PATH = Path(os.getenv("PARAGON_CHUNKS_PATH", str(ROOT / "paragon_store" / "piu_chunks.pkl")))
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5
EXPANDED_TOP_K = 6
MAX_TOP_K = 15
MIN_SCORE = 0.30   # Preserved from the original search; tune for your FAISS metric.
NEIGHBOR_WINDOW = 1


def choose_top_k(question: str) -> int:
    q = question.casefold()
    if re.search(r"\b(all|every|each|complete|entire|list|compare)\b", q):
        return MAX_TOP_K
    if re.search(r"\b(and|also|plus|along with|as well as)\b", q):
        return EXPANDED_TOP_K
    return TOP_K


def _tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (value or "").lower())


def _role_boost(question: str, chunk: dict) -> float:
    q = " ".join(_tokens(question))
    haystack = " ".join((str(chunk.get("title", "")), str(chunk.get("text", "")))).lower()
    boost = 0.0
    for role in ("head", "dean", "director", "rector", "provost", "coordinator", "manager"):
        if role in q and role in haystack:
            boost += 0.08
    important = set(_tokens(question)) & set(_tokens(haystack)) & {
        "computer", "science", "information", "technology", "technologies", "management",
        "systems", "engineering", "civil", "industrial", "architecture", "business",
        "administration", "economics", "finance", "banking", "trade", "logistics",
        "graduate", "faculty", "department",
    }
    return boost + min(len(important) * 0.025, 0.15)


class ParagonSearch:
    def __init__(self):
        if not INDEX_PATH.is_file() or not CHUNKS_PATH.is_file():
            raise FileNotFoundError(
                f"Paragon index not found. Expected {INDEX_PATH} and {CHUNKS_PATH}. "
                "Copy your original .index/.pkl files to paragon_store/."
            )
        # Lazy imports keep the everyday agent lightweight at startup.
        import faiss
        from sentence_transformers import SentenceTransformer

        self.index = faiss.read_index(str(INDEX_PATH))
        # SECURITY: only unpickle your own trusted files.
        with CHUNKS_PATH.open("rb") as handle:
            self.chunks = pickle.load(handle)
        if not isinstance(self.chunks, list) or not all(isinstance(c, dict) for c in self.chunks):
            raise TypeError("Expected piu_chunks.pkl to contain a list of chunk dictionaries")
        if self.index.ntotal != len(self.chunks):
            raise ValueError(
                f"Paragon index/chunk mismatch: {self.index.ntotal} vectors vs "
                f"{len(self.chunks)} chunks. They must come from the same build."
            )
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        dim = self.embedding_model.get_sentence_embedding_dimension()
        if dim != self.index.d:
            raise ValueError(
                f"Embedding dimension mismatch: model produces {dim}; FAISS expects "
                f"{self.index.d}. Verify the original embedding model."
            )
        self.by_doc = defaultdict(list)
        self.postings = defaultdict(list)
        self.doc_lengths = []
        self.doc_freq = Counter()
        for idx, chunk in enumerate(self.chunks):
            if chunk.get("doc_id") is not None:
                self.by_doc[chunk["doc_id"]].append(chunk)
            toks = _tokens(" ".join((str(chunk.get("title", "")), str(chunk.get("text", "")))))
            counts = Counter(toks)
            self.doc_lengths.append(max(len(toks), 1))
            for token, frequency in counts.items():
                self.doc_freq[token] += 1
                self.postings[token].append((idx, frequency))
        for group in self.by_doc.values():
            group.sort(key=lambda item: item.get("chunk_id", 0))
        self.avg_doc_len = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)

    def bm25_scores(self, question: str) -> dict[int, float]:
        scores: dict[int, float] = {}
        n = len(self.chunks)
        k1, b = 1.5, 0.75
        for term in _tokens(question):
            if term not in self.postings:
                continue
            df = self.doc_freq[term]
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            for idx, tf in self.postings[term]:
                dl = self.doc_lengths[idx]
                denom = tf + k1 * (1 - b + b * dl / max(self.avg_doc_len, 1.0))
                scores[idx] = scores.get(idx, 0.0) + idf * (tf * (k1 + 1)) / denom
        return scores

    def retrieve(self, question: str, top_k: int) -> list[dict]:
        # Exactly the normalized SentenceTransformer query encoding of the original code.
        vector = self.embedding_model.encode(
            [question], normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")
        search_k = min(max(top_k * 10, 50), self.index.ntotal)
        distances, ids = self.index.search(vector, search_k)
        vector_rank: dict[int, int] = {}
        vector_score: dict[int, float] = {}
        for rank, (score, idx) in enumerate(zip(distances[0], ids[0]), start=1):
            if idx == -1:
                continue
            vector_rank[int(idx)] = rank
            vector_score[int(idx)] = float(score)

        lexical_scores = self.bm25_scores(question)
        lexical_order = sorted(lexical_scores, key=lexical_scores.get, reverse=True)
        lexical_rank = {idx: rank for rank, idx in enumerate(lexical_order, 1)}
        fused = []
        for idx in set(vector_rank) | set(lexical_rank):
            v_rank, l_rank = vector_rank.get(idx), lexical_rank.get(idx)
            score = (1 / (60 + v_rank) if v_rank is not None else 0.0)
            score += (1 / (60 + l_rank) if l_rank is not None else 0.0)
            score += _role_boost(question, self.chunks[idx])
            fused.append((score, idx))
        fused.sort(reverse=True)
        results = []
        seen = set()
        for score, idx in fused:
            chunk = self.chunks[idx]
            key = (chunk.get("doc_id"), chunk.get("chunk_id"))
            if key in seen:
                continue
            seen.add(key)
            sem = vector_score.get(idx, -1.0)
            lex = lexical_scores.get(idx, 0.0)
            # Preserved heuristic from the original; assumes larger semantic scores
            # indicate closer matches (inner-product FAISS index).
            if lex <= 0 and sem < MIN_SCORE:
                continue
            results.append({
                "score": score, "semantic_score": sem, "lexical_score": lex,
                "doc_id": chunk.get("doc_id"), "chunk_id": chunk.get("chunk_id"),
                "title": chunk.get("title", ""), "text": chunk.get("text", ""),
            })
            if len(results) >= top_k:
                break
        return results

    def expand_with_neighbors(self, result: dict, window: int = NEIGHBOR_WINDOW) -> dict:
        doc_id, chunk_id = result.get("doc_id"), result.get("chunk_id")
        group = self.by_doc.get(doc_id, ())
        position = next((i for i, chunk in enumerate(group) if chunk.get("chunk_id") == chunk_id), None)
        if position is None:
            return result
        selected = group[max(position - window, 0):position + window + 1]
        expanded = dict(result)
        expanded["text"] = "\n\n".join(c.get("text", "") for c in selected if c.get("text", "").strip())
        expanded["context_chunk_ids"] = [c.get("chunk_id") for c in selected]
        return expanded


@lru_cache(maxsize=1)
def _searcher() -> ParagonSearch:
    """Load the embedding model and index once per Python process."""
    return ParagonSearch()


def search(question: str) -> list[dict]:
    """Return expanded evidence, not a generated answer."""
    engine = _searcher()
    return [engine.expand_with_neighbors(x) for x in engine.retrieve(question, choose_top_k(question))]


def format_context(results: list[dict]) -> str:
    if not results:
        return "No matching information found in the Paragon knowledge base."
    return "\n\n".join(
        f"SOURCE {i}\nTitle: {result['title']}\n"
        f"Document ID: {result.get('doc_id')} | Chunk ID: {result.get('chunk_id')}\n"
        f"Information:\n{result['text']}"
        for i, result in enumerate(results, start=1)
    )
