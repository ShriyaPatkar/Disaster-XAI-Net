"""
Builds the `demo_data/` folder the dashboard reads. Run it once (on Kaggle or on the laptop).

Inputs (all are outputs you already have):
  --hazard-data   folder `hazard_data` (lubdham/disaster-separate-data)  -> tile images + labels
  --buildings     predicted_buildings_test_tuned.geojson (Localization tuning output)
  --nlp           folder `case_study` from the NLP notebook output (feed_<event>/...)
  --out           output folder (default demo_data)

Kaggle: put this file's content in a cell, set ARGS below, run; then download demo_data.zip.
"""
import os, re, sys, json, glob, time, shutil, argparse, zipfile, urllib.parse, urllib.request
import numpy as np
from PIL import Image

# satellite event (xBD) -> CrisisMMD event used for the news feed
EVENTS = {
    "hurricane-harvey":    {"title": "Hurricane Harvey (2017)",   "hazard": "cyclone",    "news": "hurricane_harvey"},
    "mexico-earthquake":   {"title": "Mexico earthquake (2017)",  "hazard": "earthquake", "news": "mexico_earthquake"},
    "santa-rosa-wildfire": {"title": "Santa Rosa wildfire (2017)", "hazard": "wildfire",   "news": "california_wildfires"},
    "hurricane-michael":   {"title": "Hurricane Michael (2018)",  "hazard": "cyclone",    "news": None},
    "hurricane-matthew":   {"title": "Hurricane Matthew (2016)",  "hazard": "cyclone",    "news": None},
    "hurricane-florence":  {"title": "Hurricane Florence (2018)", "hazard": "cyclone",    "news": None},
    "midwest-flooding":    {"title": "Midwest flooding (2019)",   "hazard": "flood",      "news": None},
    "socal-fire":          {"title": "SoCal fire (2018)",         "hazard": "wildfire",   "news": None},
}
MAX_TILES_WITH_IMAGES = 40          # per event, to keep the download small
# tiles used in the paper's six-panel figures: always exported
SHOWCASE = {"santa-rosa-wildfire_00000170", "mexico-earthquake_00000085", "hurricane-harvey_00000467",
            "hurricane-michael_00000307", "hurricane-matthew_00000140", "hurricane-florence_00000048",
            "midwest-flooding_00000234", "socal-fire_00000501"}
GEOCODE_MARGIN_KM = 150             # news places farther than this from the event's buildings are dropped


def fix_text(t):
    """repair common UTF-8-read-as-Latin-1 mojibake (e.g. 'Itâ€™s' -> 'It’s', 'ChiropracticÂ Care')"""
    if not isinstance(t, str):
        return t
    try:
        import ftfy
        return ftfy.fix_text(t)
    except Exception:
        pass
    if any(m in t for m in ("â€", "Â", "Ã")):
        try:
            return t.encode("latin-1", errors="strict").decode("utf-8", errors="strict")
        except Exception:
            try:
                return t.encode("cp1252", errors="strict").decode("utf-8", errors="strict")
            except Exception:
                return t.replace("Â", "")
    return t


WKT = re.compile(r"\(\((.*?)\)\)")
def parse(w):
    m = WKT.search(w or "")
    if not m:
        return None
    try:
        return [tuple(float(v) for v in c.strip().split()[:2]) for c in m.group(1).split(",")]
    except ValueError:
        return None


def affine_for(label_path):
    js = json.load(open(label_path)); f = js.get("features", {})
    ll = {x.get("properties", {}).get("uid"): parse(x.get("wkt")) for x in f.get("lng_lat", []) or []}
    src, dst = [], []
    for x in f.get("xy", []) or []:
        p = parse(x.get("wkt")); q = ll.get(x.get("properties", {}).get("uid"))
        if p and q and len(p) == len(q):
            src += p; dst += q
    if len(src) < 3:
        return None
    A = np.c_[np.asarray(src), np.ones(len(src))]
    if np.linalg.matrix_rank(A) < 3:
        return None
    return np.linalg.lstsq(A, np.asarray(dst), rcond=None)[0]


_GC = {}
def geocode(name, cache_path):
    if not _GC and os.path.exists(cache_path):
        _GC.update(json.load(open(cache_path)))
    if name in _GC:
        return _GC[name]
    try:
        url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode({"q": name, "format": "json", "limit": 3})
        req = urllib.request.Request(url, headers={"User-Agent": "disaster-xai-net-demo/1.0 (student project)"})
        res = json.loads(urllib.request.urlopen(req, timeout=15).read().decode())
        _GC[name] = [{"lat": float(r["lat"]), "lon": float(r["lon"]), "name": r["display_name"]} for r in res]
        time.sleep(1.1)                                    # Nominatim usage policy: max 1 request/second
    except Exception as e:
        print("  geocode failed:", name, repr(e)); _GC[name] = []
    json.dump(_GC, open(cache_path, "w"))
    return _GC[name]


def km(a, b, c, d):
    from math import radians, sin, cos, asin, sqrt
    x = sin(radians(c - a) / 2) ** 2 + cos(radians(a)) * cos(radians(c)) * sin(radians(d - b) / 2) ** 2
    return 2 * 6371 * asin(sqrt(x))


def main(a):
    out = a.out; os.makedirs(out, exist_ok=True)
    # ---------- buildings (predicted footprints + predicted severity), split by event ----------
    g = json.load(open(a.buildings))
    per_event = {}
    for f in g["features"]:
        ev = f["properties"]["tile"].rsplit("_", 1)[0]
        if ev in EVENTS:
            per_event.setdefault(ev, []).append(f)
    # ---------- tiles: images + geographic bounds ----------
    tiles_index = {}
    for ev in EVENTS:
        cands = sorted(glob.glob(os.path.join(a.hazard_data, "*", "test", "images", f"{ev}_*_post_disaster.png")))
        counts = {}
        for f in per_event.get(ev, []):
            counts[f["properties"]["tile"]] = counts.get(f["properties"]["tile"], 0) + 1
        tid_of = lambda p: os.path.basename(p).replace("_post_disaster.png", "")
        chosen = sorted(cands, key=lambda p: (tid_of(p) not in SHOWCASE, -counts.get(tid_of(p), 0)))[:MAX_TILES_WITH_IMAGES]
        tiles = []
        for post in chosen:
            tid = os.path.basename(post).replace("_post_disaster.png", "")
            pre = post.replace("_post_disaster.png", "_pre_disaster.png")
            lbl = post.replace("/images/", "/labels/").replace(".png", ".json")
            if not (os.path.exists(pre) and os.path.exists(lbl)):
                continue
            T = affine_for(lbl)
            if T is None:
                continue
            W, H = Image.open(post).size
            corners = np.c_[np.array([[0, 0], [W, 0], [W, H], [0, H]]), np.ones(4)] @ T   # lon, lat
            d = os.path.join(out, "tiles", tid); os.makedirs(d, exist_ok=True)
            for src, name in [(pre, "pre.jpg"), (post, "post.jpg")]:
                Image.open(src).convert("RGB").save(os.path.join(d, name), quality=82)
            meta = {"tile": tid, "event": ev, "size": [W, H],
                    "bounds": [[float(corners[:, 1].min()), float(corners[:, 0].min())],
                               [float(corners[:, 1].max()), float(corners[:, 0].max())]],
                    "corners_lonlat": corners.tolist(), "affine": T.tolist(), "n_buildings": counts.get(tid, 0)}
            json.dump(meta, open(os.path.join(d, "meta.json"), "w"))
            # xBD ground truth (pixel polygons + damage class) for the six-panel view
            gt = [{"poly": [list(c) for c in parse(x.get("wkt"))], "sev": x.get("properties", {}).get("subtype")}
                  for x in (json.load(open(lbl)).get("features", {}).get("xy", []) or []) if parse(x.get("wkt"))]
            json.dump(gt, open(os.path.join(d, "gt.json"), "w"))
            tiles.append({k: meta[k] for k in ("tile", "bounds", "n_buildings")})
        tiles_index[ev] = tiles
        print(f"{ev}: {len(per_event.get(ev, []))} buildings, {len(tiles)} tiles with imagery")
    # ---------- news feeds (replayed CrisisMMD test items), with filtered geocoding ----------
    cache = os.path.join(out, "geocode_cache.json")
    for ev, cfg in EVENTS.items():
        feats = per_event.get(ev, [])
        evd = os.path.join(out, "events", ev); os.makedirs(evd, exist_ok=True)
        json.dump({"type": "FeatureCollection", "features": feats}, open(os.path.join(evd, "buildings.geojson"), "w"))
        if not feats:
            continue
        lats = [y for f in feats for x, y in f["geometry"]["coordinates"][0]]
        lons = [x for f in feats for x, y in f["geometry"]["coordinates"][0]]
        center = (float(np.mean(lats)), float(np.mean(lons)))
        items = []
        if cfg["news"] and a.nlp:
            fp = os.path.join(a.nlp, f"feed_{cfg['news']}", f"replay_feed_{cfg['news']}.json")
            if os.path.exists(fp):
                feed = json.load(open(fp))
                img_out = os.path.join(out, "news_images", ev); os.makedirs(img_out, exist_ok=True)
                for it in feed["items"]:
                    it["text"] = fix_text(it.get("text"))
                    places = []
                    for loc in it.get("locations", []):
                        name = fix_text(loc["name"])
                        if len(name) < 3 or name.lower() in {"harvey", "irma", "maria"}:
                            continue                       # storm names are not places
                        cands = geocode(name, cache)
                        ok = [c for c in cands if km(center[0], center[1], c["lat"], c["lon"]) <= GEOCODE_MARGIN_KM]
                        if ok:
                            places.append({"name": name, "lat": ok[0]["lat"], "lon": ok[0]["lon"], "resolved": ok[0]["name"]})
                    it["places"] = places
                    src_img = os.path.join(a.nlp, f"feed_{cfg['news']}", it.get("image_file", ""))
                    if it.get("image_file") and os.path.exists(src_img):
                        shutil.copy2(src_img, os.path.join(img_out, os.path.basename(src_img)))
                        it["image_url"] = f"/data/news_images/{ev}/{os.path.basename(src_img)}"
                    items.append(it)
        json.dump({"event": ev, "news_source": cfg["news"], "items": items}, open(os.path.join(evd, "news.json"), "w"))
        print(f"  news items: {len(items)}, with an in-region place: {sum(bool(i['places']) for i in items)}")
    json.dump({"events": [{"id": ev, **cfg, "n_buildings": len(per_event.get(ev, [])), "tiles": tiles_index.get(ev, [])}
                          for ev, cfg in EVENTS.items() if per_event.get(ev)]},
              open(os.path.join(out, "index.json"), "w"), indent=1)
    if a.zip:
        z = out.rstrip("/") + ".zip"
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, fs in os.walk(out):
                for fn in fs:
                    p = os.path.join(root, fn); zf.write(p, os.path.relpath(p, os.path.dirname(out.rstrip("/"))))
        print("wrote", z, f"{os.path.getsize(z)/1e6:.1f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hazard-data", required=True); ap.add_argument("--buildings", required=True)
    ap.add_argument("--nlp", default=None); ap.add_argument("--out", default="demo_data"); ap.add_argument("--zip", action="store_true")
    main(ap.parse_args(None if len(sys.argv) > 1 else globals().get("ARGS")))
