"""SQLite document metadata and FTS5 full-text search."""

import re
import sqlite3
import threading
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

from .errors import DatabaseError


@dataclass(frozen=True)
class DocumentRecord:
    id: int
    file_path: str
    document_date: str | None
    text: str
    indexed_at: str


class SQLiteDocumentStore:
    """Thread-safe SQLite storage with an FTS5 index kept in sync by triggers."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = str(database_path)
        self._lock = threading.RLock()
        self._closed = False
        try:
            if self.database_path != ":memory:":
                Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(
                self.database_path, timeout=30, check_same_thread=False
            )
            self._connection.row_factory = sqlite3.Row
            self._initialize()
        except (OSError, sqlite3.Error) as error:
            connection = getattr(self, "_connection", None)
            if connection is not None:
                connection.close()
            raise DatabaseError(
                f"Could not initialize SQLite FTS5 store at "
                f"{self.database_path}: {error}"
            ) from error

    def _initialize(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY,
                file_path TEXT NOT NULL UNIQUE,
                document_date TEXT,
                text TEXT NOT NULL,
                indexed_at TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                text, content='documents', content_rowid='id'
            );
            CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
                INSERT INTO documents_fts(rowid, text) VALUES (new.id, new.text);
            END;
            CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
                INSERT INTO documents_fts(documents_fts, rowid, text)
                VALUES ('delete', old.id, old.text);
            END;
            CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN
                INSERT INTO documents_fts(documents_fts, rowid, text)
                VALUES ('delete', old.id, old.text);
                INSERT INTO documents_fts(rowid, text) VALUES (new.id, new.text);
            END;
            """
        )

    def upsert_document(
        self,
        file_path: str | Path,
        text: str,
        *,
        document_date: date | str | None = None,
    ) -> int:
        """Insert or update a document and return its database ID."""
        normalized_path = str(file_path)
        if not normalized_path.strip():
            raise ValueError("file_path must not be empty.")
        if not isinstance(text, str):
            raise TypeError("text must be a string.")
        if isinstance(document_date, datetime):
            date_value = document_date.date().isoformat()
        elif isinstance(document_date, date):
            date_value = document_date.isoformat()
        elif isinstance(document_date, str):
            try:
                date_value = date.fromisoformat(document_date).isoformat()
            except ValueError as error:
                raise ValueError(
                    "document_date must be an ISO-8601 date (YYYY-MM-DD)."
                ) from error
        elif document_date is None:
            date_value = None
        else:
            raise TypeError("document_date must be a date, ISO-8601 string, or None.")

        indexed_at = datetime.now(timezone.utc).isoformat()
        try:
            with self._lock:
                self._ensure_open()
                self._connection.execute(
                    """
                    INSERT INTO documents(file_path, document_date, text, indexed_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(file_path) DO UPDATE SET
                        document_date=excluded.document_date,
                        text=excluded.text,
                        indexed_at=excluded.indexed_at
                    """,
                    (normalized_path, date_value, text, indexed_at),
                )
                row = self._connection.execute(
                    "SELECT id FROM documents WHERE file_path = ?", (normalized_path,)
                ).fetchone()
                self._connection.commit()
                return int(row["id"])
        except sqlite3.Error as error:
            self._rollback()
            raise DatabaseError(
                f"Could not save document {normalized_path}: {error}"
            ) from error

    def search(self, query: str, *, limit: int = 50) -> list[DocumentRecord]:
        """Search literal terms, requiring every term to match."""
        if not isinstance(query, str):
            raise TypeError("query must be a string.")
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000.")
        terms = re.findall(r"\w+", query, flags=re.UNICODE)
        if not terms:
            return []
        match_expression = " AND ".join(
            '"' + term.replace('"', '""') + '"' for term in terms
        )
        try:
            with self._lock:
                self._ensure_open()
                rows = self._connection.execute(
                    """
                    SELECT d.id, d.file_path, d.document_date, d.text, d.indexed_at
                    FROM documents_fts
                    JOIN documents AS d ON d.id = documents_fts.rowid
                    WHERE documents_fts MATCH ?
                    ORDER BY bm25(documents_fts), d.id
                    LIMIT ?
                    """,
                    (match_expression, limit),
                ).fetchall()
            return [
                DocumentRecord(
                    id=int(row["id"]),
                    file_path=str(row["file_path"]),
                    document_date=row["document_date"],
                    text=str(row["text"]),
                    indexed_at=str(row["indexed_at"]),
                )
                for row in rows
            ]
        except sqlite3.Error as error:
            raise DatabaseError(f"Could not search documents: {error}") from error

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._connection.close()
                self._closed = True

    def __enter__(self) -> "SQLiteDocumentStore":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise DatabaseError("The document store is closed.")

    def _rollback(self) -> None:
        with self._lock:
            if not self._closed:
                self._connection.rollback()
