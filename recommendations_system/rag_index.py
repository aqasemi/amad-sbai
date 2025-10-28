from __future__ import annotations

import os
import io
import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from dotenv import load_dotenv; load_dotenv()
import numpy as np


import google.genai as genai
from pypdf import PdfReader
import logging

client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

INDEX_DIR = Path("/home/ubuntu/datathon/recommendations_system/index")
GUIDELINES_DIR = Path("/home/ubuntu/datathon/recommendations_system/guidelines")


@dataclass
class ChunkMetadata:
    source_path: str
    page_start: int
    page_end: int
    chunk_index: int


@dataclass
class IndexedChunk:
    text: str
    metadata: ChunkMetadata


@dataclass
class RAGStore:
    chunks: List[IndexedChunk]
    embeddings: np.ndarray  # shape: (n_chunks, dim)
    embedding_model: str
    sources_mtime: Dict[str, float]
    chunk_size: int
    chunk_overlap: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _embed_call(model: str, text: str) -> Any:
    """Compatibility wrapper for embed_content parameter naming across SDK versions."""
    try:
        return client.models.embed_content(model=model, contents=text)
    except TypeError:
        try:
            return client.models.embed_content(model=model, input=text)
        except TypeError:
            return client.models.embed_content(model=model, content=text)


def _extract_embedding(resp: Any) -> List[float]:
    """Best-effort extraction across google-genai response variants."""
    # Object-like access
    if hasattr(resp, "embedding") and resp.embedding is not None:
        emb = resp.embedding
        if hasattr(emb, "values"):
            return list(emb.values)
        if isinstance(emb, (list, tuple)):
            return list(emb)
    if hasattr(resp, "embeddings") and resp.embeddings:
        emb0 = resp.embeddings[0]
        if hasattr(emb0, "values"):
            return list(emb0.values)
        if isinstance(emb0, (list, tuple)):
            return list(emb0)
        if hasattr(emb0, "embedding"):
            e = emb0.embedding
            if hasattr(e, "values"):
                return list(e.values)
            if isinstance(e, (list, tuple)):
                return list(e)
    # Dict-like fallback
    if isinstance(resp, dict):
        if "embedding" in resp:
            return list(resp["embedding"])  # type: ignore[index]
        if "embeddings" in resp and resp["embeddings"]:
            e0 = resp["embeddings"][0]
            if isinstance(e0, dict) and "embedding" in e0:
                return list(e0["embedding"])  # type: ignore[index]
            return list(e0)
    raise RuntimeError("Could not extract embedding from response")


def _pdf_pages_text(pdf_path: Path) -> List[str]:
    # Suppress verbose warnings from pypdf about malformed objects
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    reader = PdfReader(str(pdf_path))
    pages: List[str] = []
    for p in reader.pages:
        try:
            t = p.extract_text() or ""
        except Exception:
            t = ""
        pages.append(t)
    return pages


def _chunk_pages(
    pages: List[str],
    source_path: Path,
    chunk_size: int = 1600,
    chunk_overlap: int = 200,
) -> List[IndexedChunk]:
    chunks: List[IndexedChunk] = []
    buffer: List[Tuple[int, str]] = []  # (page_number, text)
    current_chars = 0
    chunk_index = 0
    for i, page_text in enumerate(pages, start=1):
        safe_text = (page_text or "").strip()
        if not safe_text:
            continue
        buffer.append((i, safe_text))
        current_chars += len(safe_text)
        if current_chars >= chunk_size:
            start_page = buffer[0][0]
            end_page = buffer[-1][0]
            combined = "\n\n".join(t for _, t in buffer)
            chunks.append(
                IndexedChunk(
                    text=combined,
                    metadata=ChunkMetadata(
                        source_path=str(source_path),
                        page_start=start_page,
                        page_end=end_page,
                        chunk_index=chunk_index,
                    ),
                )
            )
            chunk_index += 1
            # create overlap
            overlap_chars = 0
            overlapped: List[Tuple[int, str]] = []
            for pg, txt in reversed(buffer):
                overlapped.append((pg, txt))
                overlap_chars += len(txt)
                if overlap_chars >= chunk_overlap:
                    break
            buffer = list(reversed(overlapped))
            current_chars = sum(len(t) for _, t in buffer)
    # tail
    if buffer:
        start_page = buffer[0][0]
        end_page = buffer[-1][0]
        combined = "\n\n".join(t for _, t in buffer)
        chunks.append(
            IndexedChunk(
                text=combined,
                metadata=ChunkMetadata(
                    source_path=str(source_path),
                    page_start=start_page,
                    page_end=end_page,
                    chunk_index=chunk_index,
                ),
            )
        )
    return chunks


def _embed_texts(texts: List[str], model: str = "text-embedding-004", batch_size: int = 32) -> np.ndarray:
    vectors: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        for t in batch:
            # Some PDFs can have extremely long chunks; truncate defensively
            content = t if len(t) <= 8000 else t[:8000]
            resp = _embed_call(model=model, text=content)
            vec = _extract_embedding(resp)
            vectors.append(vec)
    arr = np.asarray(vectors, dtype=np.float32)
    # Normalize to unit length for cosine via dot product
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    arr = arr / norms
    return arr


def _guideline_files(directory: Path) -> List[Path]:
    files: List[Path] = []
    if not directory.exists():
        return files
    for p in directory.iterdir():
        if p.is_file() and p.suffix.lower() == ".pdf":
            files.append(p)
    return sorted(files)


def _sources_mtime_map(files: List[Path]) -> Dict[str, float]:
    return {str(p): float(p.stat().st_mtime) for p in files}


def _load_store(path: Path) -> Optional[RAGStore]:
    if not path.exists():
        return None
    with path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, RAGStore):
        return None
    return obj


def _save_store(path: Path, store: RAGStore) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(store, f)


class RAGIndex:
    def __init__(
        self,
        guidelines_dir: Path = GUIDELINES_DIR,
        index_path: Path = INDEX_DIR / "guidelines_index.pkl",
        embedding_model: str = "text-embedding-004",
        chunk_size: int = 1600,
        chunk_overlap: int = 200,
    ) -> None:
        self.guidelines_dir = guidelines_dir
        self.index_path = index_path
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._store: Optional[RAGStore] = None

    def is_loaded(self) -> bool:
        return self._store is not None

    def load(self) -> bool:
        store = _load_store(self.index_path)
        if store is None:
            return False
        # Basic compatibility check
        if store.embedding_model != self.embedding_model:
            return False
        if store.chunk_size != self.chunk_size or store.chunk_overlap != self.chunk_overlap:
            return False
        self._store = store
        return True

    def is_stale(self) -> bool:
        if self._store is None:
            return True
        files = _guideline_files(self.guidelines_dir)
        current = _sources_mtime_map(files)
        return current != self._store.sources_mtime

    def build(self) -> None:
        files = _guideline_files(self.guidelines_dir)
        _require(len(files) > 0, f"No PDF guidelines found under {self.guidelines_dir}")
        all_chunks: List[IndexedChunk] = []
        for fpath in files:
            pages = _pdf_pages_text(fpath)
            chunks = _chunk_pages(pages, fpath, self.chunk_size, self.chunk_overlap)
            all_chunks.extend(chunks)
        texts = [c.text for c in all_chunks]
        embeddings = _embed_texts(texts, model=self.embedding_model)
        store = RAGStore(
            chunks=all_chunks,
            embeddings=embeddings,
            embedding_model=self.embedding_model,
            sources_mtime=_sources_mtime_map(files),
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        _save_store(self.index_path, store)
        self._store = store

    def ensure(self) -> None:
        loaded = self.load()
        if not loaded or self.is_stale():
            self.build()

    def search(self, query: str, top_k: int = 6) -> List[Dict[str, Any]]:
        _require(self._store is not None, "Index not loaded. Call ensure() first.")
        resp = _embed_call(model=self.embedding_model, text=query[:8000])
        q_vec = np.asarray(_extract_embedding(resp), dtype=np.float32)
        q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-12)
        sims = np.dot(self._store.embeddings, q_vec)  # (n_chunks,)
        idx = np.argsort(-sims)[: max(1, top_k)]
        results: List[Dict[str, Any]] = []
        for i in idx.tolist():
            chunk = self._store.chunks[int(i)]
            score = float(sims[int(i)])
            md = chunk.metadata
            results.append(
                {
                    "text": chunk.text,
                    "score": score,
                    "source_path": md.source_path,
                    "page_start": md.page_start,
                    "page_end": md.page_end,
                    "chunk_index": md.chunk_index,
                }
            )
        return results


def ensure_built_index(
    embedding_model: str = "text-embedding-004",
    chunk_size: int = 1600,
    chunk_overlap: int = 200,
) -> RAGIndex:
    idx = RAGIndex(
        guidelines_dir=GUIDELINES_DIR,
        index_path=INDEX_DIR / "guidelines_index.pkl",
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    idx.ensure()
    return idx


__all__ = [
    "RAGIndex",
    "ensure_built_index",
]


