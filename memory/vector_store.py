"""
Vector Store — Long-term semantic memory using Chroma vector database.

Stores and retrieves research findings with semantic similarity search.
Implements the schema from Section A3.3 with intelligent chunking strategies
for different document types.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime
from typing import Any

from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


class VectorMemory:
    """Long-term semantic memory backed by Chroma vector database.

    Schema per Section A3.3:
        id, content, embedding, ticker, source_type, date, confidence,
        researcher_session, verified

    Chunking strategies:
        - SEC Filings: by section (Risk Factors, MD&A, etc.)
        - Earnings Transcripts: by speaker turn / Q&A pair
        - News Articles: by paragraph with headline context
        - Financial Statements: structured JSON with metadata filtering
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._collection = None
        self._embedding_fn = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazily initialize Chroma and embedding function."""
        if self._initialized:
            return

        try:
            import chromadb

            client = chromadb.PersistentClient(
                path=str(self._settings.chroma_path)
            )

            # Use Chroma's default embedding function or OpenAI embeddings
            embedding_fn = None
            if self._settings.openai_api_key:
                try:
                    from chromadb.utils.embedding_functions import (
                        OpenAIEmbeddingFunction,
                    )

                    embedding_fn = OpenAIEmbeddingFunction(
                        api_key=self._settings.openai_api_key,
                        model_name=self._settings.embedding_model,
                    )
                except Exception:
                    logger.warning(
                        "OpenAI embedding function unavailable, using default"
                    )

            self._collection = client.get_or_create_collection(
                name="afra_research_memory",
                embedding_function=embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )
            self._initialized = True
            logger.info(
                "Vector memory initialized at %s (%d documents)",
                self._settings.chroma_path,
                self._collection.count(),
            )

        except Exception as e:
            logger.error("Failed to initialize vector memory: %s", e)
            raise

    async def store(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        doc_id: str | None = None,
    ) -> str:
        """Store content in the vector database.

        Args:
            content: Text content to embed and store.
            metadata: Associated metadata (ticker, source_type, date, confidence, etc.).
            doc_id: Optional explicit document ID (auto-generated if not provided).

        Returns:
            The document ID of the stored record.
        """
        self._ensure_initialized()

        if not doc_id:
            # Generate a deterministic ID from content hash
            content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
            ticker = (metadata or {}).get("ticker", "unknown")
            source = (metadata or {}).get("source_type", "unknown")
            doc_id = f"{ticker}-{source}-{content_hash}"

        # Prepare metadata (Chroma only supports str, int, float, bool)
        clean_metadata = self._clean_metadata(metadata or {})
        clean_metadata.setdefault("stored_at", datetime.now().isoformat())

        # Chunk if content is large
        chunks = self._chunk_content(
            content,
            doc_type=clean_metadata.get("source_type", "general"),
        )

        ids = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}-{i:03d}" if len(chunks) > 1 else doc_id
            chunk_metadata = {**clean_metadata, "chunk_index": i, "total_chunks": len(chunks)}

            self._collection.upsert(
                ids=[chunk_id],
                documents=[chunk],
                metadatas=[chunk_metadata],
            )
            ids.append(chunk_id)

        logger.debug("Stored %d chunks for document %s", len(chunks), doc_id)
        return doc_id

    async def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search the vector database for relevant documents.

        Args:
            query: Search query for semantic similarity.
            top_k: Number of top results to return.
            filters: Optional metadata filters (e.g., {"ticker": "AAPL"}).

        Returns:
            List of results with content, score, and metadata.
        """
        self._ensure_initialized()

        where = None
        if filters:
            # Build Chroma where clause
            conditions = []
            for key, value in filters.items():
                conditions.append({key: {"$eq": value}})
            if len(conditions) == 1:
                where = conditions[0]
            elif len(conditions) > 1:
                where = {"$and": conditions}

        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=min(top_k, 20),
                where=where,
                include=["documents", "metadatas", "distances"],
            )

            output = []
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for doc, meta, dist in zip(docs, metas, distances):
                # Convert cosine distance to similarity score (0-1)
                similarity = 1 - dist if dist is not None else 0.0
                output.append({
                    "content": doc,
                    "score": round(similarity, 4),
                    "metadata": meta or {},
                })

            return output

        except Exception as e:
            logger.error("Vector search error: %s", e)
            return []

    async def delete(self, doc_id: str) -> bool:
        """Delete a document by ID."""
        self._ensure_initialized()
        try:
            self._collection.delete(ids=[doc_id])
            return True
        except Exception as e:
            logger.error("Vector delete error: %s", e)
            return False

    @property
    def count(self) -> int:
        """Number of documents in the vector store."""
        self._ensure_initialized()
        return self._collection.count()

    # ── Chunking ──────────────────────────────────────────────────────────

    def _chunk_content(
        self, content: str, doc_type: str = "general", max_chunk_size: int = 1000
    ) -> list[str]:
        """Intelligently chunk content based on document type.

        Chunking strategies from Section A3.3:
        - SEC Filings: by section headers
        - Earnings Transcripts: by speaker turn / Q&A pair
        - News Articles: by paragraph with headline context
        - General: by paragraph with overlap
        """
        if len(content) <= max_chunk_size:
            return [content]

        if doc_type in ("10-K", "10-Q", "8-K", "DEF 14A", "sec_filing"):
            return self._chunk_sec_filing(content, max_chunk_size)
        elif doc_type in ("earnings_call", "transcript"):
            return self._chunk_transcript(content, max_chunk_size)
        elif doc_type == "news":
            return self._chunk_news(content, max_chunk_size)
        else:
            return self._chunk_by_paragraph(content, max_chunk_size)

    def _chunk_sec_filing(self, content: str, max_size: int) -> list[str]:
        """Chunk SEC filings by section headers."""
        import re

        # Look for common SEC filing section headers
        section_patterns = [
            r'(?:ITEM\s+\d+[A-Z]?[\.\:])',
            r'(?:PART\s+[IVX]+)',
            r'(?:^#{1,3}\s+)',
        ]

        combined_pattern = '|'.join(section_patterns)
        parts = re.split(f'({combined_pattern})', content, flags=re.MULTILINE | re.IGNORECASE)

        chunks = []
        current = ""
        for part in parts:
            if len(current) + len(part) > max_size and current:
                chunks.append(current.strip())
                current = part
            else:
                current += part

        if current.strip():
            chunks.append(current.strip())

        return chunks if chunks else [content[:max_size]]

    def _chunk_transcript(self, content: str, max_size: int) -> list[str]:
        """Chunk earnings transcripts by speaker turn."""
        import re

        # Split by speaker labels (e.g., "John Smith - CEO:", "Analyst:")
        speaker_pattern = r'\n(?=[A-Z][a-zA-Z\s]+(?:\s*[-–—]\s*[A-Za-z\s,]+)?:)'
        turns = re.split(speaker_pattern, content)

        chunks = []
        current = ""
        for turn in turns:
            if len(current) + len(turn) > max_size and current:
                chunks.append(current.strip())
                current = turn
            else:
                current += "\n" + turn

        if current.strip():
            chunks.append(current.strip())

        return chunks if chunks else [content[:max_size]]

    def _chunk_news(self, content: str, max_size: int) -> list[str]:
        """Chunk news articles by paragraph, keeping headline context."""
        paragraphs = content.split('\n\n')
        headline = paragraphs[0] if paragraphs else ""

        chunks = []
        current = headline + "\n\n"

        for para in paragraphs[1:]:
            if len(current) + len(para) > max_size and current.strip() != headline.strip():
                chunks.append(current.strip())
                current = headline + "\n\n" + para
            else:
                current += para + "\n\n"

        if current.strip():
            chunks.append(current.strip())

        return chunks if chunks else [content[:max_size]]

    def _chunk_by_paragraph(self, content: str, max_size: int) -> list[str]:
        """Generic paragraph-based chunking with overlap."""
        paragraphs = content.split('\n\n')
        if not paragraphs:
            paragraphs = content.split('\n')

        chunks = []
        current = ""

        for para in paragraphs:
            if len(current) + len(para) > max_size and current:
                chunks.append(current.strip())
                # Keep last sentence as overlap
                sentences = current.strip().split('. ')
                overlap = sentences[-1] if sentences else ""
                current = overlap + "\n\n" + para
            else:
                current += "\n\n" + para if current else para

        if current.strip():
            chunks.append(current.strip())

        return chunks if chunks else [content[:max_size]]

    def _clean_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Clean metadata to Chroma-compatible types (str, int, float, bool)."""
        clean = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                clean[key] = value
            elif isinstance(value, (list, dict)):
                import json
                clean[key] = json.dumps(value, default=str)
            elif value is None:
                clean[key] = ""
            else:
                clean[key] = str(value)
        return clean
