"""The generation queue (same shape as the audio app's).

The web app never touches a graphics card: it writes a row. The worker, beside
the cards, claims one row at a time — one at a time is deliberate, a second job
would race the first for the same card, not share it.
"""
import json
import time

QUEUED, RUNNING, DONE, FAILED = "queued", "running", "done", "failed"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def enqueue(conn, model: str, params: dict) -> int:
    cur = conn.execute(
        "INSERT INTO jobs (model, status, params, seed, created_at) VALUES (?,?,?,?,?)",
        (model, QUEUED, json.dumps(params, ensure_ascii=False), int(params.get("seed", 0)), now()))
    conn.commit()
    return cur.lastrowid


def claim(conn):
    """Oldest queued job, marked running — or None if one is already running."""
    if conn.execute("SELECT 1 FROM jobs WHERE status=?", (RUNNING,)).fetchone():
        return None
    row = conn.execute("SELECT * FROM jobs WHERE status=? ORDER BY id LIMIT 1", (QUEUED,)).fetchone()
    if not row:
        return None
    conn.execute("UPDATE jobs SET status=?, started_at=? WHERE id=? AND status=?",
                 (RUNNING, now(), row["id"], QUEUED))
    conn.commit()
    d = dict(row)
    d["params"] = json.loads(d["params"] or "{}")
    d["status"] = RUNNING
    return d


def finish(conn, job_id, image="", error="", seconds=None) -> None:
    conn.execute(
        "UPDATE jobs SET status=?, image=?, error=?, finished_at=?, seconds=? WHERE id=?",
        (FAILED if error else DONE, image, error[-400:], now(), seconds or 0, job_id))
    conn.commit()


def positions(conn) -> dict:
    rows = conn.execute("SELECT id FROM jobs WHERE status=? ORDER BY id", (QUEUED,)).fetchall()
    return {r["id"]: i + 1 for i, r in enumerate(rows)}


def get(conn, job_id: int) -> dict | None:
    r = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    if not r:
        return None
    d = dict(r)
    d["params"] = json.loads(d["params"] or "{}")
    d["position"] = positions(conn).get(job_id, 0)
    return d


def recent(conn, limit=60, status=DONE) -> list:
    sql = "SELECT * FROM jobs"
    args = []
    if status:
        sql += " WHERE status=?"
        args.append(status)
    sql += " ORDER BY id DESC LIMIT ?"
    args.append(int(limit))
    out = []
    for r in conn.execute(sql, args):
        d = dict(r)
        d["params"] = json.loads(d["params"] or "{}")
        out.append(d)
    return out


def depth(conn) -> dict:
    q = conn.execute("SELECT COUNT(*) FROM jobs WHERE status=?", (QUEUED,)).fetchone()[0]
    r = conn.execute("SELECT COUNT(*) FROM jobs WHERE status=?", (RUNNING,)).fetchone()[0]
    return {"queued": q, "running": r}


def cancel(conn, job_id: int) -> bool:
    cur = conn.execute("DELETE FROM jobs WHERE id=? AND status=?", (job_id, QUEUED))
    conn.commit()
    return cur.rowcount > 0


def reset_stale(conn) -> int:
    """A job left RUNNING when the worker starts was interrupted by a restart."""
    cur = conn.execute("UPDATE jobs SET status=?, error=? WHERE status=?",
                       (FAILED, "interrupted — the worker restarted", RUNNING))
    conn.commit()
    return cur.rowcount
