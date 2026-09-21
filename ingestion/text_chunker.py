"""Rule-based text chunking utilities for FinQA pre/post context."""

from __future__ import annotations


def _is_low_quality(text: str) -> bool:
    """True for windows with no real lexical content.

    FinQA tables converted from charts pad pre_text with '.' sentinel lines;
    windows built from them embed as garbage ('. . .') and pollute vector
    search (6.5k pure-dot chunks in the current store).
    """
    return sum(ch.isalnum() for ch in text) < 10


def _window_chunks(
    sentences: list[str], chunk_size: int, stride: int, index_offset: int = 0
) -> list[dict]:
    chunks: list[dict] = []
    if not sentences:
        return chunks

    size = max(1, chunk_size)
    step = max(1, stride)

    idx = 0
    while idx < len(sentences):
        window = sentences[idx : idx + size]
        if not window:
            break
        text = " ".join(window)
        chunks.append(
            {
                # Global sentence indices across pre+post concatenation:
                # FinQA gold_inds text_M indexes that same concatenation, so
                # a gold sentence can be resolved to the chunk covering it.
                "start_sentence": idx + index_offset,
                "end_sentence": idx + len(window) - 1 + index_offset,
                "text": text,
                "sentence_count": len(window),
                "low_quality": _is_low_quality(text),
            }
        )
        if idx + size >= len(sentences):
            break
        idx += step

    return chunks


def chunk_context(
    pre_text: list[str], post_text: list[str], chunk_size: int = 3, stride: int = 2
) -> list[dict]:
    """Chunk pre/post text sentence lists into overlapping windows.

    Sentence indices are global over the pre_text+post_text concatenation
    (post chunks continue after pre), matching FinQA's gold text_M indexing.
    """
    pre = (
        [str(s).strip() for s in pre_text if str(s).strip()]
        if isinstance(pre_text, list)
        else []
    )
    post = (
        [str(s).strip() for s in post_text if str(s).strip()]
        if isinstance(post_text, list)
        else []
    )

    out: list[dict] = []
    for c in _window_chunks(pre, chunk_size, stride):
        c["source"] = "pre_text"
        out.append(c)
    for c in _window_chunks(post, chunk_size, stride, index_offset=len(pre)):
        c["source"] = "post_text"
        out.append(c)
    return out
