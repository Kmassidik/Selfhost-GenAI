#!/usr/bin/env python3
"""The graphics-card side. Runs on the box under systemd, beside the cards.

Claims one job at a time and runs sd.cpp, pinned to the catalogue's device
(GPU 1 — GPU 0 is the audio service, never touched). Finishes the job in hand
on SIGTERM rather than throwing away a half-rendered image.
"""
import os
import signal
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import db
import jobs
import registry

S = config.settings()
FREE_MIB = int(os.environ.get("CARD_FREE_MIB", "800"))


def card_free(device: int, timeout=120) -> bool:
    """Wait until the target card reports free. An unloaded model is not a freed
    card — the next load races the teardown and dies with 'unable to allocate'."""
    end = time.time() + timeout
    while time.time() < end:
        try:
            out = subprocess.run(
                ["nvidia-smi", f"--id={device}", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
            used = int(out.strip() or "0")
            if used <= FREE_MIB:
                return True
        except Exception:
            return True
        time.sleep(0.5)
    return True   # proceed anyway; sd.cpp --offload-to-cpu tolerates a busy-ish card


def tick(conn) -> bool:
    job = jobs.claim(conn)
    if not job:
        return False
    t0 = time.time()
    try:
        spec = registry.get(job["model"])
        if not spec:
            jobs.finish(conn, job["id"], error=f"unknown model {job['model']}")
            return True
        out_file = S.out / f"{int(time.time())}-{job['id']}.png"
        argv, env = registry.command(spec, job["params"], str(out_file), S.root)
        card_free(spec.get("device", 1))
        r = subprocess.run(argv, env=env, cwd=str(S.root), capture_output=True,
                           text=True, timeout=3600)
        if r.returncode != 0 or not out_file.exists():
            tail = "\n".join((r.stderr or r.stdout or "").strip().splitlines()[-8:])
            jobs.finish(conn, job["id"], error=tail or f"sd exited {r.returncode}",
                        seconds=round(time.time() - t0, 1))
            return True
        # sidecar params next to the image (handy outside the db too)
        out_file.with_suffix(".json").write_text(
            __import__("json").dumps({**job["params"], "file": out_file.name}, indent=2))
        jobs.finish(conn, job["id"], image=f"/img/{out_file.name}",
                    seconds=round(time.time() - t0, 1))
    except subprocess.TimeoutExpired:
        jobs.finish(conn, job["id"], error="timed out (60 min)", seconds=round(time.time() - t0, 1))
    except Exception as e:                      # a crash must never wedge the queue
        jobs.finish(conn, job["id"], error=f"{type(e).__name__}: {e}",
                    seconds=round(time.time() - t0, 1))
    return True


_stopping = False


def _stop(*_):
    global _stopping
    _stopping = True
    print("stop asked for; finishing the job in hand, then exiting", flush=True)


def main():
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    conn = db.connect(S.db_path)
    db.init(conn)
    stale = jobs.reset_stale(conn)
    print(f"worker up · out {S.out} · {stale} interrupted job(s) cleared", flush=True)
    while not _stopping:
        if not tick(conn):
            for _ in range(20):
                if _stopping:
                    break
                time.sleep(0.1)


if __name__ == "__main__":
    main()
