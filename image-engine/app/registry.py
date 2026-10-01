"""The model catalogue: one JSON per model, no code per model (audio-app style).

A model that runs on sd.cpp is a file in catalog/. The page never learns model
names; it asks for a picture and the catalogue decides the command.
"""
import glob
import json
import os
import pathlib

import config

CATALOG = config.S.app / "catalog"
REQUIRED = ("id", "name", "engine", "defaults", "limits")


class BadModel(ValueError):
    pass


def _validate(d: dict, path: str) -> dict:
    missing = [k for k in REQUIRED if k not in d]
    if missing:
        raise BadModel(f"{os.path.basename(path)} missing {', '.join(missing)}")
    if d["engine"] not in ("sdcpp", "diffusers"):
        raise BadModel(f"{d['id']}: unknown engine {d['engine']}")
    if d["engine"] == "sdcpp" and "run" not in d:
        raise BadModel(f"{d['id']}: sdcpp model needs a run block")
    d.setdefault("enabled", True)
    d.setdefault("device", 1)
    return d


def models(include_disabled=False) -> dict:
    found = {}
    for p in sorted(glob.glob(str(CATALOG / "*.json"))):
        spec = _validate(json.loads(pathlib.Path(p).read_text()), p)
        found[spec["id"]] = spec
    return {k: v for k, v in found.items() if include_disabled or v["enabled"]}


def default_model() -> dict:
    ms = models()
    if not ms:
        raise BadModel("no enabled model in catalog/")
    return next(iter(ms.values()))


def get(model_id: str) -> dict | None:
    return models().get(model_id)


class _Safe(dict):
    def __missing__(self, k):
        return "{" + k + "}"


def _sd_bin(root) -> str:
    if config.S.sd_bin and os.path.exists(config.S.sd_bin):
        return config.S.sd_bin
    return str(root / "stable-diffusion.cpp" / "build" / "bin" / "sd-cli")


def command(spec: dict, params: dict, out_path: str, root) -> tuple:
    """(argv, env) for one render. Paths resolved against the project root; the
    job is pinned to the catalogue's device (never GPU 0, the audio card)."""
    run = spec["run"]
    vals = {
        "diffusion_model": str(root / run["diffusion_model"]),
        "vae": str(root / run["vae"]),
        "llm": str(root / run["llm"]),
        "prompt": params["prompt"], "negative": params.get("negative", ""),
        "cfg": str(params["cfg"]), "steps": str(params["steps"]),
        "width": str(params["width"]), "height": str(params["height"]),
        "seed": str(params["seed"]), "out": out_path,
    }
    def fill(args):
        return [a.format_map(_Safe(vals)) for a in args]
    argv = [_sd_bin(root)] + fill(run["args"])
    if params.get("negative", "").strip() and run.get("negative_arg"):
        argv += fill(run["negative_arg"])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(spec.get("device", 1)))
    return argv, env


def clamp(spec: dict, req: dict) -> dict:
    """Apply the declared defaults + limits to a request."""
    d, lim = spec["defaults"], spec["limits"]
    def r16(v, lo, hi):
        v = max(lo, min(hi, int(v)))
        return max(lo, (v // 16) * 16)
    return {
        "prompt": (req.get("prompt") or "").strip(),
        "negative": (req.get("negative") or "").strip(),
        "width": r16(req.get("width") or d["width"], lim["min_side"], lim["max_side"]),
        "height": r16(req.get("height") or d["height"], lim["min_side"], lim["max_side"]),
        "steps": max(1, min(lim["max_steps"], int(req.get("steps") or d["steps"]))),
        "cfg": float(req.get("cfg") if req.get("cfg") is not None else d["cfg"]),
        "seed": int(req["seed"]) if req.get("seed") else 0,
    }
