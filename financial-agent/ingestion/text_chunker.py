"""Rule-based text chunking utilities for FinQA pre/post context."""

from __future__ import annotations

from typing import Dict, List


def _window_chunks(sentences: List[str], chunk_size: int, stride: int) -> List[Dict]:
    chunks: List[Dict] = []
    if not sentences:
        return chunks

    size = max(1, chunk_size)
    step = max(1, stride)

    idx = 0
    while idx < len(sentences):
        window = sentences[idx : idx + size]
        if not window:
            break
        chunks.append(
            {
                "start_sentence": idx,
                "end_sentence": idx + len(window) - 1,
                "text": " ".join(window),
                "sentence_count": len(window),
            }
        )
        if idx + size >= len(sentences):
            break
        idx += step

    return chunks


def chunk_context(pre_text: List[str], post_text: List[str], chunk_size: int = 3, stride: int = 2) -> List[Dict]:
    """Chunk pre/post text sentence lists into overlapping windows."""
    pre = [str(s).strip() for s in pre_text if str(s).strip()] if isinstance(pre_text, list) else []
    post = [str(s).strip() for s in post_text if str(s).strip()] if isinstance(post_text, list) else []

    out: List[Dict] = []
    for c in _window_chunks(pre, chunk_size, stride):
        c["source"] = "pre_text"
        out.append(c)
    for c in _window_chunks(post, chunk_size, stride):
        c["source"] = "post_text"
        out.append(c)
    return out
