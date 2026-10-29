"""
Disaster XAI-Net demo backend.

    uvicorn backend.app:app --port 8000        (run from the project folder)
    open http://localhost:8000

Serves the dashboard, the precomputed demo data (demo_data/), zone prioritization computed live,
live analysis of a new social-media post (news_bundle), and optional live satellite analysis of an
uploaded image pair (deployment_bundle + localization_bundle, if present in models/).
"""
import os, io, sys, json, time, importlib.util, tempfile, threading
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("DXN_DATA", os.path.join(ROOT, "demo_data"))
MODELS = os.environ.get("DXN_MODELS", os.path.join(ROOT, "models"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prioritization import prioritize, PriorityConfig  # noqa: E402

app = FastAPI(title="Disaster XAI-Net demo API")
if os.path.isdir(DATA):
    app.mount("/data", StaticFiles(directory=DATA), name="data")
app.mount("/static", StaticFiles(directory=os.path.join(ROOT, "frontend")), name="static")

_cache = {}
def _json(path):
    if path not in _cache:
        if not os.path.exists(path):
            raise HTTPException(404, f"missing data file: {os.path.relpath(path, ROOT)}")
        _cache[path] = json.load(open(path))
    return _cache[path]


def _buildings(ev):
    """GeoJSON features -> flat building records (centroid lat/lon, severity, p_damaged)."""
    key = ("bld", ev)
    if key not in _cache:
        g = _json(os.path.join(DATA, "events", ev, "buildings.geojson"))
        out = []
        for f in g["features"]:
            ring = f["geometry"]["coordinates"][0][:-1] or f["geometry"]["coordinates"][0]
            p = f["properties"]
            out.append({"id": f"{p.get('tile')}#{p.get('pred_building_id')}", "tile": p.get("tile"),
                        "lon": sum(x for x, _ in ring) / len(ring), "lat": sum(y for _, y in ring) / len(ring),
                        "severity": p.get("severity"), "p_damaged": p.get("p_damaged"),
                        "confidence": p.get("severity_confidence"), "probabilities": p.get("probabilities")})
        _cache[key] = out
    return _cache[key]


def _news(ev, until=None):
    items = _json(os.path.join(DATA, "events", ev, "news.json"))["items"]
    if until:
        items = [i for i in items if (i.get("timestamp") or "") <= until]
    return items


def _news_points(items):
    pts = []
    for it in items:
        for pl in it.get("places", []):
            pts.append({"id": it.get("item_id"), "lat": pl["lat"], "lon": pl["lon"],
                        "score": (it.get("priority") or {}).get("score", 0), "text": it.get("text")})
    return pts


# ------------------------------------------------------------------ pages and data
@app.get("/")
def index():
    return FileResponse(os.path.join(ROOT, "frontend", "index.html"))


@app.get("/api/events")
def events():
    return _json(os.path.join(DATA, "index.json"))


@app.get("/api/events/{ev}/buildings")
def buildings(ev: str):
    return _json(os.path.join(DATA, "events", ev, "buildings.geojson"))


@app.get("/api/events/{ev}/news")
def news(ev: str, until: Optional[str] = None):
    return {"items": _news(ev, until)}


@app.get("/api/events/{ev}/zones")
def zones(ev: str, until: Optional[str] = None, cell_m: float = 150, w1: float = 0.5, w2: float = 0.35, w3: float = 0.15,
          t1: float = 0.20, t2: float = 0.35, t3: float = 0.50, use_news: bool = True):
    s = (w1 + w2 + w3) or 1.0
    cfg = PriorityConfig(cell_m=max(50.0, min(cell_m, 1000.0)), weights=(w1 / s, w2 / s, w3 / s), thresholds=(t1, t2, t3))
    pts = _news_points(_news(ev, until)) if use_news else []
    return prioritize(_buildings(ev), pts, cfg)


from pydantic import BaseModel
class ZoneRequest(BaseModel):
    until: Optional[str] = None
    cell_m: float = 150; w1: float = 0.5; w2: float = 0.35; w3: float = 0.15
    t1: float = 0.20; t2: float = 0.35; t3: float = 0.50
    use_news: bool = True
    extra_news: list = []          # live-analyzed posts: [{"id","lat","lon","score","text"}]


@app.post("/api/events/{ev}/zones")
def zones_post(ev: str, r: ZoneRequest):
    s = (r.w1 + r.w2 + r.w3) or 1.0
    cfg = PriorityConfig(cell_m=max(50.0, min(r.cell_m, 1000.0)), weights=(r.w1 / s, r.w2 / s, r.w3 / s), thresholds=(r.t1, r.t2, r.t3))
    pts = (_news_points(_news(ev, r.until)) + list(r.extra_news)) if r.use_news else []
    return prioritize(_buildings(ev), pts, cfg)


@app.get("/api/tiles/{tile}")
def tile(tile: str):
    return _json(os.path.join(DATA, "tiles", tile, "meta.json"))


# ------------------------------------------------------------------ live models (optional)
_lock = threading.Lock()
_mods = {}
def _load(name, rel, loader):
    if name in _mods:
        return _mods[name]
    path = os.path.join(MODELS, rel)
    if not os.path.exists(path):
        raise HTTPException(503, f"{name} not installed: expected {os.path.relpath(path, ROOT)} (see README)")
    spec = importlib.util.spec_from_file_location(f"dxn_{name}", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    loader(mod, os.path.dirname(path))
    _mods[name] = mod
    return mod


@app.get("/api/status")
def status():
    have = {n: os.path.exists(os.path.join(MODELS, r)) for n, r in
            [("news", "news_bundle/news_inference.py"), ("damage", "deployment_bundle/inference.py"),
             ("localization", "localization_bundle/unet_localization.pth")]}
    return {"data": os.path.isdir(DATA), "models": have, "loaded": list(_mods)}


@app.post("/api/news/analyze")
async def analyze_post(text: str = Form(""), event: str = Form(""), image: Optional[UploadFile] = File(None)):
    """Live analysis of a new post with the trained NLP models; places are geocoded and kept only near the event."""
    with _lock:
        ni = _load("news", "news_bundle/news_inference.py", lambda m, d: m.load(d))
        img = None
        if image is not None and image.filename:
            from PIL import Image
            img = Image.open(io.BytesIO(await image.read())).convert("RGB")
        t0 = time.time()
        out = ni.analyze_item(text=text or None, image=img, do_geocode=False)
        out["latency_s"] = round(time.time() - t0, 2)
    if event:
        try:
            bl = _buildings(event)
            clat = sum(b["lat"] for b in bl) / len(bl); clon = sum(b["lon"] for b in bl) / len(bl)
            sys.path.insert(0, ROOT)
            from prepare_demo_data import geocode, km
            places = []
            for loc in out.get("locations", []):
                for c in geocode(loc["name"], os.path.join(DATA, "geocode_cache.json")):
                    if km(clat, clon, c["lat"], c["lon"]) <= 150:
                        places.append({"name": loc["name"], "lat": c["lat"], "lon": c["lon"], "resolved": c["name"]}); break
            out["places"] = places
        except HTTPException:
            out["places"] = []
    return JSONResponse(out)


_six_cache = {}
@app.get("/api/tiles/{tile}/sixpanel.png")
def six_panel(tile: str, w1: float = 0.5, w2: float = 0.35, w3: float = 0.15, cell_m: float = 150):
    """Six-panel figure for one exported tile: pre, post, ground truth, pipeline, heatmap, priority zones."""
    from fastapi.responses import Response
    d = os.path.join(DATA, "tiles", tile)
    if not os.path.exists(os.path.join(d, "meta.json")):
        raise HTTPException(404, "imagery for this tile is not in the demo data")
    key = (tile, round(w1, 3), round(w2, 3), round(w3, 3), round(cell_m))
    if key not in _six_cache:
        ev = _json(os.path.join(d, "meta.json"))["event"]
        feats = [f for f in _json(os.path.join(DATA, "events", ev, "buildings.geojson"))["features"]
                 if f["properties"].get("tile") == tile]
        from sixpanel import render
        with _lock:
            _six_cache[key] = render(d, feats, (w1, w2, w3), cell_m=max(50.0, min(cell_m, 1000.0)))
        if len(_six_cache) > 64:
            _six_cache.pop(next(iter(_six_cache)))
    return Response(_six_cache[key], media_type="image/png")


class ExplainRequest(BaseModel):
    tile: str
    ring: list                       # building polygon [[lon, lat], ...] as shown on the map


@app.post("/api/explain")
def explain(r: ExplainRequest):
    """Branch-wise Grad-CAM for one predicted building (needs deployment_bundle + exported tile imagery)."""
    import numpy as np
    from PIL import Image
    d = os.path.join(DATA, "tiles", r.tile)
    if not os.path.exists(os.path.join(d, "meta.json")):
        raise HTTPException(404, "imagery for this tile is not in the demo data (only the exported tiles have Grad-CAM)")
    meta = _json(os.path.join(d, "meta.json"))
    T = np.asarray(meta["affine"])                    # [x, y, 1] @ T = [lon, lat]
    A, b0 = T[:2].T, T[2]
    px = np.array([np.linalg.solve(A, np.asarray(p[:2]) - b0) for p in r.ring])
    bbox = (float(px[:, 0].min()), float(px[:, 1].min()), float(px[:, 0].max()), float(px[:, 1].max()))
    pre_im = Image.open(os.path.join(d, "pre.jpg")).convert("RGB")
    post_im = Image.open(os.path.join(d, "post.jpg")).convert("RGB")
    with _lock:
        dmg = _load("damage", "deployment_bundle/inference.py", lambda m, dd: m.load_models(dd))
        from explain import gradcam_building
        out = gradcam_building(dmg, pre_im, post_im, bbox)
    out["tile"] = r.tile; out["bbox_px"] = [round(v, 1) for v in bbox]
    return JSONResponse(out)


@app.post("/api/tiles/analyze")
async def analyze_tile(pre: UploadFile = File(...), post: UploadFile = File(...)):
    """Live satellite analysis: U-Net localization on the pre image, then hazard + damage for every building."""
    from PIL import Image
    import traceback
    lb = os.path.join(MODELS, "localization_bundle")
    missing = [f for f in ("localization_inference.py", "unet_localization.pth", "postprocess_config.json")
               if not os.path.exists(os.path.join(lb, f))]
    if missing:
        raise HTTPException(503, "models/localization_bundle is missing: " + ", ".join(missing) +
                            " (use the FULL zip of the Localization tuning output, which contains the .pth weights)")
    try:
        pre_im = Image.open(io.BytesIO(await pre.read())).convert("RGB")
        post_im = Image.open(io.BytesIO(await post.read())).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"could not read the uploaded images: {e}")
    if pre_im.size != post_im.size:
        raise HTTPException(400, f"pre and post images must have the same size (got {pre_im.size} and {post_im.size})")
    try:
        return _analyze_pair(pre_im, post_im)
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"{type(e).__name__}: {e}")


def _analyze_pair(pre_im, post_im):
    with _lock:
        loc = _load("localization", "localization_bundle/localization_inference.py",
                    lambda m, d: m.load(os.path.join(d, "unet_localization.pth")))
        dmg = _load("damage", "deployment_bundle/inference.py", lambda m, d: m.load_models(d))
        t0 = time.time()
        blds = loc.predict_buildings(pre_im)
        t1 = time.time()
        req = [{"building_id": f"b{b['id']}", "polygon": b["polygon"]} for b in blds]
        res = dmg.predict_full(pre_im, post_im, buildings=req if req else None)
        t2 = time.time()
    res["localization"] = {"n_buildings": len(blds), "seconds": round(t1 - t0, 2)}
    res["damage_seconds"] = round(t2 - t1, 2)
    return JSONResponse(res)
