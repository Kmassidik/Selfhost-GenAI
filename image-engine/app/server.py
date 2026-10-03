#!/usr/bin/env python3
"""selfhostimageai web app — a prompt->image page over stable-diffusion.cpp.

The app never touches a card: it writes a job row; the worker (worker.py, a
separate systemd unit beside the cards) runs it one at a time on GPU 1.
Creating is gated by OWNER_PASSWORD; viewing the gallery is open.
"""
import ipaddress
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

import config
import db
import jobs
import registry

S = config.settings()
db.init(db.connect(S.db_path))            # create the table once at import
app = FastAPI(title="selfhostimageai")


def conn():
    return db.connect(S.db_path)


def _private(ip: str) -> bool:
    try:
        a = ipaddress.ip_address(ip)
        return a.is_private or a.is_loopback or ip.startswith("100.")  # LAN + tailscale
    except ValueError:
        return False


def require_owner(request: Request, authorization: str = Header(default="")):
    """Creating costs a GPU, so it is gated. With OWNER_PASSWORD set, a request
    must carry it. With it empty, only private/LAN/tailscale clients may create,
    so an unconfigured box is never open to the internet."""
    if S.owner_password:
        if authorization.removeprefix("Bearer ").strip() != S.owner_password:
            raise HTTPException(401, "generation needs the password")
        return
    ip = request.client.host if request.client else ""
    if not _private(ip):
        raise HTTPException(403, "set OWNER_PASSWORD to create from the internet")


class GenReq(BaseModel):
    prompt: str
    negative: str = ""
    width: int | None = None
    height: int | None = None
    steps: int | None = None
    cfg: float | None = None
    seed: int | None = None
    model: str | None = None


@app.post("/api/generate")
def generate(req: GenReq, request: Request, authorization: str = Header(default="")):
    require_owner(request, authorization)
    spec = (registry.get(req.model) if req.model else None) or registry.default_model()
    p = registry.clamp(spec, req.model_dump())
    if not p["prompt"]:
        raise HTTPException(400, "prompt is empty")
    if not p["seed"]:
        p["seed"] = random.randint(1, 2**31 - 1)
    c = conn()
    jid = jobs.enqueue(c, spec["id"], p)
    j = jobs.get(c, jid)
    c.close()
    return {"job_id": jid, "position": j["position"], "params": p}


@app.get("/api/job/{job_id}")
def job(job_id: int):
    c = conn()
    j = jobs.get(c, job_id)
    c.close()
    if not j:
        raise HTTPException(404, "no such job")
    return j


@app.post("/api/cancel/{job_id}")
def cancel(job_id: int, request: Request, authorization: str = Header(default="")):
    require_owner(request, authorization)
    c = conn()
    ok = jobs.cancel(c, job_id)
    c.close()
    return {"cancelled": ok}


@app.get("/api/gallery")
def gallery(limit: int = 24, offset: int = 0):
    c = conn()
    out = [{"id": r["id"], "image": r["image"], "seconds": r["seconds"],
            "model": r["model"], **r["params"]}
           for r in jobs.recent(c, limit, offset=offset) if r["image"]]
    c.close()
    return out


@app.get("/api/jobs")
def jobs_feed(limit: int = 24):
    """The live feed: recent jobs across ALL statuses, with queue position for
    the ones still waiting. Lets the page show a queue that fills as you stack
    requests and drains one render at a time."""
    c = conn()
    rows = jobs.recent(c, limit, status=None)
    pos = jobs.positions(c)
    c.close()
    return [{"id": r["id"], "status": r["status"], "image": r["image"],
             "seconds": r["seconds"], "error": r["error"], "model": r["model"],
             "position": pos.get(r["id"]), **r["params"]} for r in rows]


@app.get("/api/queue")
def queue():
    c = conn()
    d = jobs.depth(c)
    c.close()
    return d


@app.get("/img/{name}")
def img(name: str):
    f = S.out / name
    if ".." in name or "/" in name or not f.exists():
        raise HTTPException(404)
    return FileResponse(f)


@app.get("/api/model")
def model():
    s = registry.default_model()
    return {"id": s["id"], "name": s["name"], "blurb": s.get("blurb", ""),
            "defaults": s["defaults"], "limits": s["limits"],
            "auth_required": bool(S.owner_password)}


@app.get("/api/models")
def models():
    """Every enabled model, for the picker. The page lets you choose one per job;
    the worker loads it on demand (one model on the 8 GB card at a time)."""
    ms = registry.models()  # enabled only
    return {"models": [{"id": s["id"], "name": s["name"], "blurb": s.get("blurb", ""),
                        "defaults": s["defaults"], "limits": s["limits"],
                        "speed": s.get("speed", "")} for s in ms.values()],
            "auth_required": bool(S.owner_password)}


@app.get("/healthz")
def healthz():
    import pathlib
    sd = registry._sd_bin(S.root)
    return {"ok": True, "sd_bin": os.path.exists(sd), "images": len(list(S.out.glob("*.png")))}


@app.get("/", response_class=HTMLResponse)
def index():
    return (S.app / "web" / "index.html").read_text()
