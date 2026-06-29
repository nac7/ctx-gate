"""
CodebaseIndexer: Walks a project directory, chunks source files, and indexes them.

Incremental: only re-indexes files that have changed (via file hash comparison).
Respects .claudeignore / .gitignore patterns to skip irrelevant files.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

from .store import chunk_file, VectorStore, Embedder, LANG_MAP


# Default ignore patterns (same as recommended .claudeignore)
DEFAULT_IGNORE = {
    "node_modules", "dist", "build", ".git", "__pycache__",
    ".pytest_cache", ".mypy_cache", "venv", ".venv", "env",
    "coverage", ".coverage", "htmlcov", ".tox", "eggs",
    ".eggs", "*.egg-info", "*.pyc", "*.pyo", "*.pyd",
    "*.so", "*.dll", "*.dylib", "*.db", "*.sqlite", "*.sqlite3",
    "*.lock", "package-lock.json", "yarn.lock", "Cargo.lock",
    "*.min.js", "*.min.css", "*.map",
}

INDEXED_EXTENSIONS = set(LANG_MAP.keys())


class CodebaseIndexer:
    """
    Incrementally indexes a codebase for RAG retrieval.

    Usage:
        indexer = CodebaseIndexer("/path/to/project")
        indexer.index()                     # initial or incremental index
        results = indexer.retrieve("how does auth work?", top_k=5)
    """

    def __init__(
        self,
        project_root: str | Path,
        db_path: str = ".ctx-gate/rag.db",
        model_name: str = "all-MiniLM-L6-v2",
        max_file_size_kb: int = 500,
        chunk_lines: int = 60,
    ):
        self.root = Path(project_root).resolve()
        self.db_path = db_path
        self.max_file_size_bytes = max_file_size_kb * 1024
        self.chunk_lines = chunk_lines

        self.store = VectorStore(db_path=db_path)
        self.embedder = Embedder(model_name=model_name)
        self._ignore_patterns = self._load_ignore_patterns()
        self._stats = {
            "files_indexed": 0,
            "files_skipped": 0,
            "chunks_indexed": 0,
            "last_index_time": 0.0,
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def index(self, force: bool = False) -> dict:
        """
        Walk the project and index new/changed files.
        
        Args:
            force: Re-index everything even if unchanged.
            
        Returns:
            Stats dict with counts of what was indexed.
        """
        t0 = time.time()
        known_hashes = self.store.file_hashes()
        source_files = list(self._walk())

        new_chunks = []
        files_indexed = 0
        files_skipped = 0

        for path in source_files:
            try:
                file_hash = self._hash_file(path)
            except OSError:
                continue

            stored_hash = known_hashes.get(str(path))
            if not force and stored_hash == file_hash:
                files_skipped += 1
                continue

            chunks = chunk_file(path, max_chunk_lines=self.chunk_lines)
            if not chunks:
                continue
            new_chunks.extend(chunks)
            files_indexed += 1

        if new_chunks:
            # Fit TF-IDF vocab if using fallback embedder
            if self.embedder._fallback:
                self.embedder.fit_vocab([c.content for c in new_chunks])

            # Embed in batches (avoid OOM on large codebases)
            embeddings = self._embed_batched([c.content for c in new_chunks])
            self.store.upsert(new_chunks, embeddings)

        elapsed = time.time() - t0
        self._stats.update({
            "files_indexed": files_indexed,
            "files_skipped": files_skipped,
            "chunks_indexed": len(new_chunks),
            "last_index_time": elapsed,
            "total_chunks": self.store.total_chunks(),
        })
        return dict(self._stats)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.2,
    ) -> "RetrievalResult":
        """
        Find the most relevant code chunks for a query.

        Args:
            query:     Natural language query (e.g., "how does auth work?")
            top_k:     Max number of chunks to return
            min_score: Minimum cosine similarity threshold

        Returns:
            RetrievalResult with chunks formatted for LLM injection
        """
        from .store import RetrievalResult

        query_vec = self.embedder.embed_one(query)
        raw = self.store.search(query_vec, top_k=top_k * 2)  # over-fetch, then filter

        filtered = [(chunk, score) for chunk, score in raw if score >= min_score]
        filtered = filtered[:top_k]

        chunks = []
        scores = []
        total_full_tokens = 0

        for record, score in filtered:
            from .store import Chunk
            chunk = Chunk(
                chunk_id=record["chunk_id"],
                file_path=record["file_path"],
                content=record["content"],
                start_line=record["start_line"],
                end_line=record["end_line"],
                language=record["language"],
                symbol=record.get("symbol", ""),
                file_hash=record["file_hash"],
            )
            chunks.append(chunk)
            scores.append(score)
            # Rough token estimate for the full file
            try:
                total_full_tokens += Path(record["file_path"]).stat().st_size // 4
            except OSError:
                total_full_tokens += len(record["content"]) // 4

        chunk_tokens = sum(len(c.content) // 4 for c in chunks)
        tokens_saved = max(0, total_full_tokens - chunk_tokens)

        return RetrievalResult(
            chunks=chunks,
            prompt_tokens_saved=tokens_saved,
            query=query,
            scores=scores,
        )

    def format_for_prompt(self, result: "RetrievalResult") -> str:
        """
        Format retrieved chunks as a compact context block for injection
        into the LLM's system prompt or user message.
        """
        if not result.chunks:
            return ""

        lines = [f"[RAG CONTEXT — {len(result.chunks)} relevant chunks retrieved]"]
        for chunk, score in zip(result.chunks, result.scores):
            rel_path = self._rel(chunk.file_path)
            header = f"{rel_path}:{chunk.start_line}-{chunk.end_line}"
            if chunk.symbol:
                header += f" ({chunk.symbol})"
            lines.append(f"\n--- {header} [relevance: {score:.2f}] ---")
            lines.append(f"```{chunk.language}")
            lines.append(chunk.content.strip())
            lines.append("```")

        return "\n".join(lines)

    @property
    def stats(self) -> dict:
        return dict(self._stats)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _walk(self) -> list[Path]:
        """Yield source files that should be indexed."""
        results = []
        for root, dirs, files in os.walk(self.root):
            root_path = Path(root)

            # Prune ignored directories in place
            dirs[:] = [
                d for d in dirs
                if not self._is_ignored(root_path / d)
            ]

            for fname in files:
                fpath = root_path / fname
                if fpath.suffix.lower() not in INDEXED_EXTENSIONS:
                    continue
                if self._is_ignored(fpath):
                    continue
                try:
                    if fpath.stat().st_size > self.max_file_size_bytes:
                        continue
                except OSError:
                    continue
                results.append(fpath)
        return results

    def _is_ignored(self, path: Path) -> bool:
        """Check if a path should be excluded from indexing."""
        name = path.name
        for pattern in self._ignore_patterns:
            if re.fullmatch(pattern, name, re.IGNORECASE):
                return True
            if name in DEFAULT_IGNORE:
                return True
        return False

    def _load_ignore_patterns(self) -> list[str]:
        """Load patterns from .claudeignore or .gitignore."""
        patterns = []
        for ignore_file in [".claudeignore", ".gitignore"]:
            ignore_path = self.root / ignore_file
            if ignore_path.exists():
                for line in ignore_path.read_text().splitlines():
                    line = line.strip()
                    if line and not line.startswith("#"):
                        # Convert glob to regex
                        pattern = line.replace(".", r"\.").replace("*", ".*").replace("?", ".")
                        patterns.append(pattern)
        return patterns

    def _hash_file(self, path: Path) -> str:
        import hashlib
        h = hashlib.sha256()
        h.update(path.read_bytes())
        return h.hexdigest()[:16]

    def _embed_batched(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            results.extend(self.embedder.embed(batch))
        return results

    def _rel(self, path: str) -> str:
        try:
            return str(Path(path).relative_to(self.root))
        except ValueError:
            return path
