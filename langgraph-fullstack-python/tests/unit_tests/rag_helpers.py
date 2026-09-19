"""Test helpers for the RAG layer.

The retrieval *mechanics* (metadata contract, school filtering, top_k, id
handling) must be testable without downloading BGE-M3, so a deterministic local
embedder is provided here. It is a hashing embedder: it maps character n-grams
into a fixed-size vector, so texts sharing vocabulary end up closer together.
That is enough to exercise ranking and filtering, but it is **not** a semantic
model - semantic retrieval tests use the real BGE-M3 when it is installed.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable
from pathlib import Path

from langchain_core.embeddings import Embeddings

VECTOR_SIZE = 96


def _ngrams(text: str, width: int = 2) -> Iterable[str]:
    """Yield character n-grams of a text.

    Args:
        text: Input text.
        width: n-gram width.

    Yields:
        Each n-gram.
    """
    cleaned = "".join(text.split())
    if len(cleaned) < width:
        if cleaned:
            yield cleaned
        return
    for index in range(len(cleaned) - width + 1):
        yield cleaned[index : index + width]


class HashingEmbeddings(Embeddings):
    """Deterministic local embedder used only by tests."""

    def __init__(self, size: int = VECTOR_SIZE) -> None:
        self.size = size

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.size
        for gram in _ngrams(text):
            digest = hashlib.sha1(gram.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.size
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents.

        Args:
            texts: Document texts.

        Returns:
            One normalised vector per text.
        """
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query.

        Args:
            text: Query text.

        Returns:
            A normalised vector.
        """
        return self._embed(text)


def write_minimal_pdf(path: Path, pages: list[str]) -> Path:
    """Write a small, valid PDF with one text line per page.

    The file is assembled by hand (including the cross-reference table) so the
    test suite does not need a PDF writing dependency. Text is ASCII only, which
    keeps the Helvetica base font usable.

    Args:
        path: Destination file; parent directories must exist.
        pages: One text line per page.

    Returns:
        The written path.
    """
    font_number = 3 + 2 * len(pages)
    kids = " ".join(f"{3 + 2 * index} 0 R" for index in range(len(pages)))
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode(),
    ]
    for index, text in enumerate(pages):
        content_number = 3 + 2 * index + 1
        stream = f"BT /F1 12 Tf 20 100 Td ({text}) Tj ET".encode()
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 200] "
                f"/Contents {content_number} 0 R "
                f"/Resources << /Font << /F1 {font_number} 0 R >> >> >>"
            ).encode()
        )
        objects.append(
            b"<< /Length "
            + str(len(stream)).encode()
            + b" >>\nstream\n"
            + stream
            + b"\nendstream"
        )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_offset = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    ).encode()
    path.write_bytes(bytes(out))
    return path
