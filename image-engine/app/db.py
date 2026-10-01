"""SQLite, WAL. One jobs table is the whole store — a finished job carries its
own image path and params, so the gallery is just the done rows."""
import sqlite3


def connect(path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def init(conn) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            model       TEXT NOT NULL,
            status      TEXT NOT NULL,
            params      TEXT NOT NULL,
            image       TEXT DEFAULT '',
            error       TEXT DEFAULT '',
            seed        INTEGER DEFAULT 0,
            seconds     REAL DEFAULT 0,
            created_at  TEXT,
            started_at  TEXT,
            finished_at TEXT
        )""")
    conn.commit()
