"""GraphRAG document ingestion (spec §7 / build order #6).

Ingests the bank policy (agent/policy.yaml), the dataset README's policy
sections (R1-R10 + SAR), and any typology docs; chunks by heading; builds a
TF-IDF index; persists to graphrag/index.json.

Verify (per spec): retrieval returns the right clause for a policy question.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = ROOT / "graphrag" / "index.json"

SOURCES = [
    ROOT / "agent" / "policy.yaml",
    ROOT / "data" / "HHGOA_IEEE" / "README.md",
    ROOT / "agent" / "gate.py",  # cost model docstrings are policy-relevant
]


def _chunk_markdown(text: str) -> list[dict]:
    """Split markdown on headings; keep the heading with its body."""
    chunks: list[dict] = []
    cur_h, cur = "(intro)", []
    for line in text.splitlines():
        m = re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            if cur:
                chunks.append({"heading": cur_h, "text": "\n".join(cur).strip()})
            cur_h, cur = m.group(2).strip(), [line]
        else:
            cur.append(line)
    if cur:
        chunks.append({"heading": cur_h, "text": "\n".join(cur).strip()})
    return [c for c in chunks if len(c["text"]) > 60]


def _chunk_yaml(text: str) -> list[dict]:
    """Top-level yaml sections as chunks."""
    chunks, cur_h, cur = [], "(preamble)", []
    for line in text.splitlines():
        if line and not line[0].isspace() and line.rstrip().endswith(":"):
            if cur:
                chunks.append({"heading": cur_h, "text": "\n".join(cur).strip()})
            cur_h, cur = line.strip().rstrip(":"), [line]
        else:
            cur.append(line)
    if cur:
        chunks.append({"heading": cur_h, "text": "\n".join(cur).strip()})
    return [c for c in chunks if len(c["text"]) > 40]


def _chunk_python(text: str) -> list[dict]:
    """Docstring blocks (policy rationale lives there)."""
    chunks = []
    for m in re.finditer(r'"""(.*?)"""', text, re.S):
        body = m.group(1).strip()
        if len(body) > 80:
            first = body.splitlines()[0].strip()
            chunks.append({"heading": first[:80], "text": body})
    return chunks


def build_index() -> dict:
    docs: list[dict] = []
    for src in SOURCES:
        if not src.exists():
            continue
        text = src.read_text()
        if src.suffix in (".md",):
            parts = _chunk_markdown(text)
        elif src.suffix in (".yaml", ".yml"):
            parts = _chunk_yaml(text)
        else:
            parts = _chunk_python(text)
        for p in parts:
            docs.append({"source": str(src.relative_to(ROOT)), **p})

    corpus = [f"{d['heading']}\n{d['text']}" for d in docs]
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    matrix = vec.fit_transform(corpus)
    index = {
        "docs": docs,
        "vocabulary": vec.vocabulary_,
        "idf": vec.idf_.tolist(),
        "matrix_shape": list(matrix.shape),
        # CSR pieces for exact reconstruction
        "matrix_data": matrix.data.tolist(),
        "matrix_indices": matrix.indices.tolist(),
        "matrix_indptr": matrix.indptr.tolist(),
    }
    INDEX_PATH.write_text(json.dumps(index))
    return index


def load_index() -> tuple[dict, "TfidfVectorizer"]:
    import numpy as np

    state = json.loads(INDEX_PATH.read_text())
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    # Reconstruct vocabulary/idf without refitting on docs.
    vec.vocabulary_ = state["vocabulary"]
    vec.idf_ = np.asarray(state["idf"], dtype=float)
    from scipy.sparse import csr_matrix

    mat = csr_matrix((state["matrix_data"], state["matrix_indices"], state["matrix_indptr"]),
                     shape=tuple(state["matrix_shape"]))
    return {"docs": state["docs"], "matrix": mat}, vec


def retrieve(query: str, k: int = 4) -> list[dict]:
    idx, vec = load_index()
    from sklearn.metrics.pairwise import linear_kernel

    q = vec.transform([query])
    scores = linear_kernel(q, idx["matrix"]).ravel()
    ranked = scores.argsort()[::-1][:k]
    return [
        {"score": round(float(scores[i]), 4), **idx["docs"][i]}
        for i in ranked
        if scores[i] > 0
    ]


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        for r in retrieve(" ".join(sys.argv[1:])):
            print(f"[{r['score']:.3f}] {r['source']} :: {r['heading']}")
    else:
        idx = build_index()
        print(f"indexed {len(idx['docs'])} chunks from {len(SOURCES)} sources -> {INDEX_PATH}")
