#!/usr/bin/env python3
"""SigLIP 2 image and text embedding API with zero-shot image classification, for Railway.

google/siglip2-base-patch16-224 (Apache-2.0 per its Hugging Face model card) on CPU behind one
authenticated HTTP API. The model puts images and text in the same 768-dimensional space, so one
service gives a vector database both halves of image search, and classifies a picture against labels
you choose at request time with no training.

What this file adds around the model, because it ships no server:

  1. Bearer API_KEY on every route except GET /health. An empty API_KEY refuses to start.
  2. GET /health answers 503 until the model is loaded and warmed, then 200, unauthenticated.
  3. POST /embed/image   one or more images -> one vector each (L2-normalised unless normalize=false).
  4. POST /v1/embeddings (alias /embed/text) text -> vectors, in the OpenAI embeddings response shape.
  5. POST /classify      one image + candidate labels -> a probability per label.
  6. Upload guards: body size cap, pixel-count cap (decompression bombs), magic-byte format check.
  7. Weights are baked into the image: nothing downloads at boot.
  8. Requests run one at a time behind a lock; threads are sized from the container's cgroup quota.
     Images and text are held in memory for the request only and are never written to disk or logged.
"""

import io
import json
import os
import socket
import sys
import threading
import time

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

API_KEY = os.environ.get("API_KEY") or ""
THREADS = int(os.environ.get("TORCH_THREADS") or os.cpu_count() or 1)
MODEL_DIR = os.environ.get("MODEL_DIR") or "/models/siglip2"
MODEL_ID = "siglip2-base-patch16-224"
REPO = "google/siglip2-base-patch16-224"
MAX_IMAGE_BYTES = int(os.environ.get("MAX_IMAGE_BYTES") or 10 * 1024 * 1024)
MAX_FILES = int(os.environ.get("MAX_FILES") or 16)
MAX_PIXELS = int(os.environ.get("MAX_PIXELS") or 50_000_000)
MAX_TEXTS = int(os.environ.get("MAX_TEXTS") or 128)
MAX_TEXT_CHARS = int(os.environ.get("MAX_TEXT_CHARS") or 2000)
MAX_LABELS = int(os.environ.get("MAX_LABELS") or 128)
DEFAULT_TEMPLATE = os.environ.get("LABEL_TEMPLATE") or "This is a photo of {}."
TEXT_TOKENS = 64  # the length SigLIP 2 was trained with: every text is padded or cut to it
IMAGE_SIZE = 224
PUBLIC_PATHS = {"/health"}
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "GIF", "BMP", "TIFF"}
TEXT_BATCH = 32

Image.MAX_IMAGE_PIXELS = MAX_PIXELS  # PIL raises DecompressionBombError past 2x this

app = FastAPI(
    title="SigLIP 2 embedding and zero-shot classification API",
    version="1.0.0",
    description=f"{REPO} (Apache-2.0) on CPU. Authorization: Bearer <API_KEY>.",
)

_state = {"error": None, "loaded_in": None, "ready": False}
_m: dict = {}
_lock = threading.Lock()


def _tensor(out):
    """transformers 5 may hand back a model-output object instead of a bare tensor."""
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    return out


def _load() -> None:
    t0 = time.time()
    try:
        import numpy as np
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch.set_num_threads(THREADS)
        model = AutoModel.from_pretrained(MODEL_DIR).eval()
        tok = AutoTokenizer.from_pretrained(MODEL_DIR)
        _m.update(torch=torch, np=np, model=model, tok=tok)
        _m["dim"] = int(model.config.text_config.hidden_size)
        _m["scale"] = float(model.logit_scale.exp().item())
        _m["bias"] = float(model.logit_bias.item())
        _embed_images([Image.new("RGB", (64, 64), (128, 128, 128))], True)
        _embed_texts(["warm up"], True)
        _state["ready"] = True
        _state["loaded_in"] = round(time.time() - t0, 1)
        print(f"[siglip2] ready dim={_m['dim']} threads={THREADS} in {_state['loaded_in']}s", flush=True)
    except Exception as exc:  # noqa: BLE001 - reported over /health, never fatal
        _state["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[siglip2] LOAD FAILED: {_state['error']}", flush=True)


def _norm(t):
    return t / t.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _embed_images(images: list, normalize: bool):
    """Preprocessing is SiglipImageProcessor's: RGB, bilinear resize to 224x224 (aspect ratio is not
    kept), scale to 0-1, normalise with mean 0.5 and std 0.5."""
    np, torch, model = _m["np"], _m["torch"], _m["model"]
    arrs = []
    for img in images:
        a = np.asarray(img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR),
                       dtype=np.float32)
        arrs.append(((a / 255.0 - 0.5) / 0.5).transpose(2, 0, 1))
    pixel_values = torch.from_numpy(np.stack(arrs))
    with torch.inference_mode():
        feats = _tensor(model.get_image_features(pixel_values=pixel_values))
        return _norm(feats) if normalize else feats


def _embed_texts(texts: list, normalize: bool):
    """The model was trained on lower-cased text padded to 64 tokens; both are applied here."""
    torch, model, tok = _m["torch"], _m["model"], _m["tok"]
    out = []
    for i in range(0, len(texts), TEXT_BATCH):
        chunk = [t.lower() for t in texts[i:i + TEXT_BATCH]]
        enc = tok(chunk, padding="max_length", truncation=True, max_length=TEXT_TOKENS, return_tensors="pt")
        with torch.inference_mode():
            feats = _tensor(model.get_text_features(input_ids=enc["input_ids"]))
        out.append(_norm(feats) if normalize else feats)
    return torch.cat(out)


def _decode(data: bytes, name: str) -> Image.Image:
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(413, f"{name}: larger than the {MAX_IMAGE_BYTES} byte limit")
    try:
        probe = Image.open(io.BytesIO(data))
        if probe.format not in ALLOWED_FORMATS:
            raise HTTPException(422, f"{name}: format {probe.format} is not accepted "
                                     f"({', '.join(sorted(ALLOWED_FORMATS))})")
        probe.load()
        return probe
    except HTTPException:
        raise
    except Image.DecompressionBombError:
        raise HTTPException(413, f"{name}: more than {MAX_PIXELS} pixels") from None
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise HTTPException(422, f"{name}: not a readable image ({type(exc).__name__})") from None


@app.middleware("http")
async def guards(request: Request, call_next):
    path = request.url.path.rstrip("/") or "/"
    if path in PUBLIC_PATHS:
        return await call_next(request)
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if not (scheme.lower() == "bearer" and API_KEY and token == API_KEY):
        return JSONResponse({"error": "Missing or invalid Authorization: Bearer <API_KEY>."},
                            status_code=401, headers={"WWW-Authenticate": "Bearer"})
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > MAX_IMAGE_BYTES * MAX_FILES + 65536:
        return JSONResponse({"error": "request body is too large"}, status_code=413)
    return await call_next(request)


def _status_kb(field: str) -> int:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(field + ":"):
                return int(line.split()[1])
    return 0


def _need_ready() -> None:
    if not _state["ready"]:
        raise HTTPException(503, _state["error"] or "model is still loading, retry in a few seconds")


def _bool(value, default: bool) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def _vectors(t, digits: int = 6) -> list:
    return [[round(x, digits) for x in row] for row in t.tolist()]


async def _read_images(request: Request, max_files: int) -> tuple[list, dict]:
    """Images from multipart `file` parts or one raw image body; returns (files, text form fields)."""
    ctype = request.headers.get("content-type", "")
    files: list[tuple[str, bytes]] = []
    fields: dict = {}
    if ctype.startswith("multipart/form-data"):
        form = await request.form(max_files=max_files, max_fields=max_files + 8)
        for field, value in form.multi_items():
            if hasattr(value, "read"):
                files.append((value.filename or field, await value.read(MAX_IMAGE_BYTES + 1)))
            else:
                fields[field] = value
    elif ctype.startswith("image/") or ctype == "application/octet-stream":
        body = bytearray()
        async for chunk in request.stream():
            body += chunk
            if len(body) > MAX_IMAGE_BYTES:
                raise HTTPException(413, f"body: larger than the {MAX_IMAGE_BYTES} byte limit")
        files.append(("body", bytes(body)))
    else:
        raise HTTPException(415, "send multipart/form-data with `file` parts, or a raw image body "
                                 "with an image/* Content-Type")
    if not files:
        raise HTTPException(422, "no image found: attach it as multipart field `file`")
    if len(files) > max_files:
        raise HTTPException(413, f"at most {max_files} images per request")
    return files, fields


@app.get("/health", include_in_schema=False)
def health():
    if not _state["ready"]:
        return JSONResponse({"status": "loading" if not _state["error"] else "failed",
                             "error": _state["error"]}, status_code=503)
    return {"status": "ok", "loaded_in_s": _state["loaded_in"], "threads": THREADS,
            "rss_mb": _status_kb("VmRSS") // 1024, "peak_rss_mb": _status_kb("VmHWM") // 1024}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{
        "id": MODEL_ID, "object": "model", "repo": REPO, "license": "apache-2.0",
        "dimensions": _m.get("dim", 768), "image_input": f"{IMAGE_SIZE}x{IMAGE_SIZE}",
        "text_max_tokens": TEXT_TOKENS, "multilingual": True}],
        "limits": {"max_image_bytes": MAX_IMAGE_BYTES, "max_files": MAX_FILES, "max_texts": MAX_TEXTS,
                   "max_text_chars": MAX_TEXT_CHARS, "max_labels": MAX_LABELS}}


@app.post("/embed/image")
async def embed_image(request: Request):
    _need_ready()
    files, fields = await _read_images(request, MAX_FILES)
    normalize = _bool(request.query_params.get("normalize", fields.get("normalize")), True)
    images = [(name, _decode(data, name), len(data)) for name, data in files]
    t0 = time.time()
    with _lock:
        try:
            vecs = _embed_images([img for _, img, _ in images], normalize)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, f"inference failed: {type(exc).__name__}: {exc}") from exc
    rows = _vectors(vecs)
    data = [{"object": "embedding", "index": i, "filename": name, "width": img.width, "height": img.height,
             "embedding": rows[i]} for i, (name, img, _) in enumerate(images)]
    return JSONResponse({"object": "list", "model": MODEL_ID, "dimensions": len(rows[0]),
                         "normalized": normalize, "data": data},
                        headers={"X-Processing-Seconds": f"{time.time() - t0:.3f}"})


async def _embed_text(request: Request):
    _need_ready()
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        raise HTTPException(422, 'send JSON: {"input": "text" or ["text", ...]}') from None
    if not isinstance(body, dict):
        raise HTTPException(422, 'send a JSON object: {"input": "text" or ["text", ...]}')
    inp = body.get("input")
    texts = [inp] if isinstance(inp, str) else inp
    if not isinstance(texts, list) or not texts or not all(isinstance(t, str) and t.strip() for t in texts):
        raise HTTPException(422, "`input` must be a non-empty string or a list of non-empty strings")
    if len(texts) > MAX_TEXTS:
        raise HTTPException(413, f"at most {MAX_TEXTS} texts per request")
    if any(len(t) > MAX_TEXT_CHARS for t in texts):
        raise HTTPException(413, f"a text is longer than {MAX_TEXT_CHARS} characters")
    normalize = _bool(body.get("normalize"), True)
    t0 = time.time()
    with _lock:
        try:
            vecs = _embed_texts(texts, normalize)
            tokens = int(sum(min(len(_m["tok"](t.lower())["input_ids"]), TEXT_TOKENS) for t in texts))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, f"inference failed: {type(exc).__name__}: {exc}") from exc
    rows = _vectors(vecs)
    return JSONResponse({"object": "list", "model": MODEL_ID, "normalized": normalize,
                         "data": [{"object": "embedding", "index": i, "embedding": r} for i, r in enumerate(rows)],
                         "usage": {"prompt_tokens": tokens, "total_tokens": tokens}},
                        headers={"X-Processing-Seconds": f"{time.time() - t0:.3f}"})


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    return await _embed_text(request)


@app.post("/embed/text")
async def embed_text(request: Request):
    return await _embed_text(request)


def _labels(raw) -> list:
    if raw is None or not str(raw).strip():
        raise HTTPException(422, "`labels` is required: a comma-separated list or a JSON array of strings")
    raw = str(raw).strip()
    if raw.startswith("["):
        try:
            labels = json.loads(raw)
        except ValueError:
            raise HTTPException(422, "`labels` looks like JSON but does not parse") from None
    else:
        labels = [x.strip() for x in raw.split(",")]
    if not isinstance(labels, list) or not all(isinstance(x, str) and x.strip() for x in labels):
        raise HTTPException(422, "`labels` must be non-empty strings")
    labels = [x.strip() for x in labels]
    if len(labels) > MAX_LABELS:
        raise HTTPException(413, f"at most {MAX_LABELS} labels per request")
    if any(len(x) > MAX_TEXT_CHARS for x in labels):
        raise HTTPException(413, f"a label is longer than {MAX_TEXT_CHARS} characters")
    return labels


@app.post("/classify")
async def classify(request: Request):
    """Zero-shot classification. SigLIP scores each label on its own with a sigmoid, so `probability`
    values do not sum to 1 and several labels can be true at once; `softmax` ranks them against each
    other when exactly one should win."""
    _need_ready()
    files, fields = await _read_images(request, 1)
    qp = request.query_params
    labels = _labels(qp.get("labels", fields.get("labels")))
    template = qp.get("template", fields.get("template", DEFAULT_TEMPLATE))
    if template and "{}" not in template:
        raise HTTPException(422, "`template` must contain {} where the label goes, e.g. 'a photo of {}.'")
    name, data = files[0]
    img = _decode(data, name)
    prompts = [template.replace("{}", lab) if template else lab for lab in labels]
    t0 = time.time()
    with _lock:
        try:
            torch = _m["torch"]
            iv = _embed_images([img], True)
            tv = _embed_texts(prompts, True)
            logits = (iv @ tv.T)[0] * _m["scale"] + _m["bias"]
            sig = torch.sigmoid(logits).tolist()
            soft = torch.softmax(logits, dim=0).tolist()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, f"inference failed: {type(exc).__name__}: {exc}") from exc
    results = sorted(({"label": lab, "probability": round(sig[i], 5), "softmax": round(soft[i], 5)}
                      for i, lab in enumerate(labels)), key=lambda r: -r["probability"])
    return JSONResponse({"model": MODEL_ID, "filename": name, "width": img.width, "height": img.height,
                         "template": template, "best": results[0]["label"], "results": results},
                        headers={"X-Processing-Seconds": f"{time.time() - t0:.3f}"})


def main() -> int:
    if not API_KEY:
        print("[siglip2] FATAL: API_KEY is empty; refusing to publish the API to anyone who finds the URL.",
              file=sys.stderr, flush=True)
        return 1
    threading.Thread(target=_load, name="load", daemon=True).start()
    # Railway injects PORT and its healthcheck dials that injected port. asyncio sets
    # IPV6_V6ONLY on a "::" listener, so one socket per family: IPv4 for the edge and
    # healthcheck, IPv6 for the private network (<service>.railway.internal).
    port = int(os.environ.get("PORT") or 8080)
    socks = []
    for family, addr in ((socket.AF_INET, "0.0.0.0"), (socket.AF_INET6, "::")):
        try:
            sock = socket.socket(family, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if family == socket.AF_INET6:
                sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            sock.bind((addr, port))
            sock.set_inheritable(True)
            socks.append(sock)
        except OSError as exc:
            print(f"[siglip2] cannot bind {addr}:{port}: {exc}", flush=True)
    if not socks:
        return 1
    uvicorn.Server(uvicorn.Config(app, log_level=os.environ.get("LOG_LEVEL", "warning"))).run(sockets=socks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
