"""
Zone-level rescue prioritization (Disaster XAI-Net, Section 3.12 of the paper).

    P(z) = w1*D(z) + w2*C(z) + w3*N(z)   ->  Low / Medium / High / Critical

D(z)  mean expected damage severity of the buildings in zone z            (0..1)
C(z)  density of damaged buildings, sum P(damaged) / (area_km2 * rho_ref)  (0..1, capped)
N(z)  social-media impact: sum of item scores s(r) of geocoded items near z / kappa (0..1, capped)

Zones are square grid cells of `cell_m` metres. Everything here is arithmetic on model
outputs - no learned parameters. The weights and thresholds are UNCALIBRATED defaults and
must be validated before operational use; outputs are decision-support indicators only.
"""
import math
from dataclasses import dataclass, field, asdict

SEVERITY_INDEX = {"no-damage": 0, "minor-damage": 1, "major-damage": 2, "destroyed": 3}
CLASSES = ["Low", "Medium", "High", "Critical"]


@dataclass
class PriorityConfig:
    cell_m: float = 150.0                 # zone size (metres)
    weights: tuple = (0.5, 0.35, 0.15)    # (damage severity, damage concentration, news impact)
    thresholds: tuple = (0.20, 0.35, 0.50)  # Low < t1 <= Medium < t2 <= High < t3 <= Critical
    rho_ref: float = 400.0                # damaged buildings per km2 that saturates C(z)
    kappa: float = 2.0                    # summed item score that saturates N(z)
    news_radius_km: float = 10.0          # an item informs zones within this radius of its place
    min_buildings: int = 3                # zones with fewer buildings are not ranked
    def to_dict(self):
        return asdict(self)


def expected_severity(b):
    """E[y]/3 in [0,1]. Uses class probabilities when available, else the predicted class."""
    p = b.get("probabilities")
    if p:
        return sum(SEVERITY_INDEX[k] * float(v) for k, v in p.items() if k in SEVERITY_INDEX) / 3.0
    return SEVERITY_INDEX.get(b.get("severity"), 0) / 3.0


def _cell_of(lat, lon, lat0, cell_m):
    dlat = cell_m / 111320.0
    dlon = cell_m / (111320.0 * math.cos(math.radians(lat0)))
    return int(math.floor(lat / dlat)), int(math.floor(lon / dlon)), dlat, dlon


def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def classify(score, thresholds):
    t1, t2, t3 = thresholds
    return "Critical" if score >= t3 else "High" if score >= t2 else "Medium" if score >= t1 else "Low"


def prioritize(buildings, news_items=(), cfg: PriorityConfig = None):
    """
    buildings : list of dicts with 'lat', 'lon', and 'probabilities' (class->prob) or 'severity',
                optional 'p_damaged', 'id'.
    news_items: list of dicts with 'lat', 'lon', 'score' (item score s(r)), optional 'id', 'text'.
    Returns a GeoJSON FeatureCollection of zones (polygons) with indicators, score and class,
    plus a summary.
    """
    cfg = cfg or PriorityConfig()
    pts = [b for b in buildings if b.get("lat") is not None and b.get("lon") is not None]
    if not pts:
        return {"type": "FeatureCollection", "features": [], "summary": {"n_zones": 0}, "config": cfg.to_dict()}
    lat0 = sum(b["lat"] for b in pts) / len(pts)
    zones = {}
    for b in pts:
        i, j, dlat, dlon = _cell_of(b["lat"], b["lon"], lat0, cfg.cell_m)
        z = zones.setdefault((i, j), {"buildings": [], "i": i, "j": j})
        z["buildings"].append(b)
    area_km2 = (cfg.cell_m / 1000.0) ** 2
    w1, w2, w3 = cfg.weights
    feats = []
    for (i, j), z in zones.items():
        bs = z["buildings"]
        if len(bs) < cfg.min_buildings:
            continue
        sev = [expected_severity(b) for b in bs]
        pdam = [float(b["p_damaged"]) if b.get("p_damaged") is not None else (1.0 if SEVERITY_INDEX.get(b.get("severity"), 0) > 0 else 0.0) for b in bs]
        D = sum(sev) / len(sev)
        C = min(1.0, sum(pdam) / (area_km2 * cfg.rho_ref))
        clat, clon = (i + 0.5) * dlat, (j + 0.5) * dlon
        near = [r for r in news_items if r.get("lat") is not None and _haversine_km(clat, clon, r["lat"], r["lon"]) <= cfg.news_radius_km]
        N = min(1.0, sum(float(r.get("score", 0)) for r in near) / cfg.kappa)
        P = w1 * D + w2 * C + w3 * N
        counts = {k: 0 for k in SEVERITY_INDEX}
        for b in bs:
            s = b.get("severity") or max(b["probabilities"], key=b["probabilities"].get)
            counts[s] = counts.get(s, 0) + 1
        ring = [[j * dlon, i * dlat], [(j + 1) * dlon, i * dlat], [(j + 1) * dlon, (i + 1) * dlat],
                [j * dlon, (i + 1) * dlat], [j * dlon, i * dlat]]
        feats.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring]},
                      "properties": {"zone_id": f"z{i}_{j}", "priority": classify(P, cfg.thresholds),
                                     "score": round(P, 4), "D_severity": round(D, 4), "C_concentration": round(C, 4),
                                     "N_news": round(N, 4), "n_buildings": len(bs),
                                     "n_damaged_expected": round(sum(pdam), 1), "severity_counts": counts,
                                     "n_news_items": len(near),
                                     "news_item_ids": [r.get("id") for r in near][:20],
                                     "center": [round(clat, 6), round(clon, 6)]}})
    feats.sort(key=lambda f: -f["properties"]["score"])
    for rank, f in enumerate(feats, 1):
        f["properties"]["rank"] = rank
    dist = {c: sum(f["properties"]["priority"] == c for f in feats) for c in CLASSES}
    summary = {"n_zones": len(feats), "n_buildings": len(pts), "priority_counts": dist,
               "top_zones": [f["properties"]["zone_id"] for f in feats[:10]],
               "note": "Uncalibrated decision-support indicator; not a statement about casualties or rescue needs."}
    return {"type": "FeatureCollection", "features": feats, "summary": summary, "config": cfg.to_dict()}
