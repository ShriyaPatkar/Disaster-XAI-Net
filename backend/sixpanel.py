"""
Six-panel tile view, rendered on demand for any exported tile (same figure as the Kaggle notebook
Disaster_XAI_Net_Six_Panel_Figure.ipynb):
 (a) pre  (b) post  (c) xBD ground truth  (d) pipeline (U-Net footprints + fusion severity)
 (e) severity-weighted damage heatmap, fixed scale, only where buildings were detected
 (f) rescue-priority zones, same formula as the map (news term not used inside a single tile)
"""
import io, os, json, math
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly, Rectangle, Patch
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

SEV_IDX = {"no-damage": 0, "minor-damage": 1, "major-damage": 2, "destroyed": 3}
SEV_COL = {"no-damage": "#1a9850", "minor-damage": "#fee08b", "major-damage": "#f46d43", "destroyed": "#a50026"}
PRI_COL = {"Low": "#4575b4", "Medium": "#fdae61", "High": "#f46d43", "Critical": "#a50026"}
HEAT = LinearSegmentedColormap.from_list("dmg", ["#ffffb2", "#fecc5c", "#fd8d3c", "#f03b20", "#bd0026"])
HEAT_SIGMA_M, HEAT_VMAX = 25.0, 1000.0


def metres_per_px(T):
    T = np.asarray(T); lat0 = float(T[2, 1])
    k = 111320.0; c = math.cos(math.radians(lat0))
    dx = math.hypot(T[0, 0] * k * c, T[0, 1] * k); dy = math.hypot(T[1, 0] * k * c, T[1, 1] * k)
    return (dx + dy) / 2


def lonlat_to_px(ring, T):
    T = np.asarray(T); A, b = T[:2].T, T[2]
    return [list(np.linalg.solve(A, np.asarray(p[:2]) - b)) for p in ring]


def _classify(s, thr):
    t1, t2, t3 = thr
    return "Critical" if s >= t3 else "High" if s >= t2 else "Medium" if s >= t1 else "Low"


def zones(blds, size, mpp, cell_m, w, thr, rho_ref=400.0, min_bld=3):
    n = max(1, round(size * mpp / cell_m)); cpx = size / n; area = (cpx * mpp / 1000) ** 2
    out = []
    for i in range(n):
        for j in range(n):
            bs = [b for b in blds if j * cpx <= b["cx"] < (j + 1) * cpx and i * cpx <= b["cy"] < (i + 1) * cpx]
            z = {"x0": j * cpx, "y0": i * cpx, "w": cpx, "n": len(bs)}
            if len(bs) >= min_bld:
                D = float(np.mean([SEV_IDX[b["sev"]] / 3 for b in bs]))
                C = min(1.0, sum(b["p_damaged"] for b in bs) / (area * rho_ref))
                P = w[0] * D + w[1] * C
                z.update(score=P, cls=_classify(P, thr))
            out.append(z)
    return out, cpx * mpp


def render(tile_dir, pred_features, weights=(0.5, 0.35, 0.15), thresholds=(0.20, 0.35, 0.50), cell_m=150.0):
    meta = json.load(open(os.path.join(tile_dir, "meta.json")))
    T = meta["affine"]; mpp = metres_per_px(T)
    pre = Image.open(os.path.join(tile_dir, "pre.jpg")).convert("RGB")
    post = Image.open(os.path.join(tile_dir, "post.jpg")).convert("RGB")
    size = post.size[0]
    s = sum(weights) or 1.0; w = [x / s for x in weights]

    preds = []
    for f in pred_features:
        poly = lonlat_to_px(f["geometry"]["coordinates"][0], T); a = np.asarray(poly)
        p = f["properties"]
        preds.append({"poly": poly, "cx": a[:, 0].mean(), "cy": a[:, 1].mean(), "sev": p.get("severity", "no-damage"),
                      "p_damaged": float(p.get("p_damaged") or 0)})
    gt_path = os.path.join(tile_dir, "gt.json")
    gt = json.load(open(gt_path)) if os.path.exists(gt_path) else None
    Z, cell_real = zones(preds, size, mpp, cell_m, w, thresholds)

    ds = 4; g = size // ds
    dmg = np.zeros((g, g)); occ = np.zeros((g, g))
    for b in preds:
        x, y = min(g - 1, max(0, int(b["cx"] / ds))), min(g - 1, max(0, int(b["cy"] / ds)))
        dmg[y, x] += SEV_IDX[b["sev"]] / 3; occ[y, x] += 1
    sig = HEAT_SIGMA_M / mpp / ds; px_km2 = (mpp * ds / 1000) ** 2
    dmg_d = gaussian_filter(dmg, sig) / px_km2; occ_d = gaussian_filter(occ, sig) / px_km2
    alpha = np.clip(occ_d / (0.15 * occ_d.max() + 1e-9), 0, 1) * 0.85 if occ_d.max() > 0 else np.zeros_like(occ_d)
    rgba = HEAT(np.clip(dmg_d / HEAT_VMAX, 0, 1)); rgba[..., 3] = alpha

    cnt = lambda L: {k: sum(1 for x in L if x["sev"] == k) for k in SEV_IDX}
    fmt = lambda c: "  ".join(f"{k.split('-')[0].capitalize()}:{v}" for k, v in c.items())
    fig, ax = plt.subplots(2, 3, figsize=(18, 13.6))
    for a_ in ax.flat:
        a_.set_xticks([]); a_.set_yticks([]); a_.set_xlim(0, size); a_.set_ylim(size, 0)
    ax[0, 0].imshow(pre); ax[0, 0].set_title("(a) Pre-disaster", fontsize=12, weight="bold")
    ax[0, 1].imshow(post); ax[0, 1].set_title("(b) Post-disaster", fontsize=12, weight="bold")
    ax[0, 2].imshow(post)
    if gt is not None:
        for b in gt:
            if b.get("sev") in SEV_COL and len(b.get("poly", [])) >= 3:
                ax[0, 2].add_patch(MplPoly(b["poly"], closed=True, fc=SEV_COL[b["sev"]], ec=SEV_COL[b["sev"]], alpha=0.55, lw=0.8))
        gtv = [b for b in gt if b.get("sev") in SEV_COL]
        ax[0, 2].set_title(f"(c) Ground truth (xBD), {len(gtv)} buildings\n{fmt(cnt(gtv))}", fontsize=11, weight="bold")
    else:
        ax[0, 2].text(size / 2, size / 2, "ground truth not exported\n(re-run the demo-data export)", ha="center", va="center",
                      color="white", fontsize=12, bbox=dict(fc="#0008", ec="none"))
        ax[0, 2].set_title("(c) Ground truth (xBD)", fontsize=11, weight="bold")
    ax[1, 0].imshow(post)
    for b in preds:
        ax[1, 0].add_patch(MplPoly(b["poly"], closed=True, fc=SEV_COL[b["sev"]], ec=SEV_COL[b["sev"]], alpha=0.55, lw=0.8))
    ax[1, 0].set_title(f"(d) Pipeline: U-Net footprints + fusion severity, {len(preds)} buildings\n{fmt(cnt(preds))}", fontsize=11, weight="bold")
    ax[1, 1].imshow(post); ax[1, 1].imshow(np.zeros((g, g)), cmap="gray", alpha=0.25, extent=(0, size, size, 0))
    ax[1, 1].imshow(rgba, extent=(0, size, size, 0), interpolation="bilinear")
    sm = plt.cm.ScalarMappable(cmap=HEAT, norm=plt.Normalize(0, HEAT_VMAX))
    cax = ax[1, 1].inset_axes([1.02, 0.0, 0.035, 1.0]); cb = fig.colorbar(sm, cax=cax)
    cb.set_label("severity-weighted buildings per km² (fixed scale)", fontsize=9)
    ax[1, 1].set_title(f"(e) Damage heatmap: severity-weighted density (σ = {HEAT_SIGMA_M:.0f} m)\nshown only where buildings were detected",
                       fontsize=11, weight="bold")
    ax[1, 2].imshow(post, alpha=0.75)
    for z in Z:
        x0, y0, wd = z["x0"], z["y0"], z["w"]
        if "cls" in z:
            ax[1, 2].add_patch(Rectangle((x0, y0), wd, wd, fc=PRI_COL[z["cls"]], ec="white", lw=1, alpha=0.55))
            ax[1, 2].text(x0 + wd / 2, y0 + wd / 2, f"{z['cls']}\n{z['score']:.2f}", ha="center", va="center", fontsize=8,
                          color="white", weight="bold")
        else:
            ax[1, 2].add_patch(Rectangle((x0, y0), wd, wd, fc="none", ec="white", lw=0.6, ls=":", hatch="..", alpha=0.6))
    pc = {c: sum(1 for z in Z if z.get("cls") == c) for c in PRI_COL}
    ax[1, 2].set_title(f"(f) Rescue-priority zones (~{cell_real:.0f} m), P = {w[0]:.2f}·D + {w[1]:.2f}·C\n" +
                       "  ".join(f"{k}:{v}" for k, v in pc.items()) + f"  not ranked:{sum(1 for z in Z if 'cls' not in z)}",
                       fontsize=11, weight="bold")
    handles = [Patch(color=c, label=k.replace("-", " ").capitalize()) for k, c in SEV_COL.items()] + \
              [Patch(color=c, alpha=0.6, label=f"Priority {k}") for k, c in PRI_COL.items()] + \
              [Patch(fc="none", ec="grey", hatch="..", label="Not ranked (< 3 buildings)")]
    fig.legend(handles=handles, loc="lower center", ncol=9, fontsize=9, frameon=False)
    fig.suptitle(f"{meta['tile']}   (about {size * mpp:.0f} m across)", fontsize=13)
    fig.subplots_adjust(left=0.01, right=0.97, top=0.92, bottom=0.05, wspace=0.2, hspace=0.2)
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=110); plt.close(fig)
    return buf.getvalue()
