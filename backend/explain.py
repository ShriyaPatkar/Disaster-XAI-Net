"""
Branch-wise Grad-CAM for one building (Stage 2 fusion model), computed on demand.

Same technique as the paper (Section "Explainability"): for the predicted class k, Grad-CAM is
computed SEPARATELY for each fusion branch, attributed to the POST-event input:
  - ConvNeXt branch: output of conv.backbone.stages[1]   (feature map, 8x8 for a 64 px crop)
  - ViT branch:      output of vit.backbone.blocks[-1].norm1, patch tokens reshaped to a 4x4 grid
The two maps are two separate explanations of the same fused prediction, not one combined map.
"""
import io, time, base64
import numpy as np
import torch
from PIL import Image

_JET = np.array([[0, 0, .5], [0, 0, 1], [0, .5, 1], [0, 1, 1], [.5, 1, .5], [1, 1, 0], [1, .5, 0], [1, 0, 0], [.5, 0, 0]])


def _jet(x):
    """x in [0,1] (HxW) -> HxWx3 uint8, jet-like colormap without matplotlib."""
    pos = np.clip(x, 0, 1) * (len(_JET) - 1)
    i = np.floor(pos).astype(int); f = (pos - i)[..., None]
    i2 = np.minimum(i + 1, len(_JET) - 1)
    return (255 * (_JET[i] * (1 - f) + _JET[i2] * f)).astype(np.uint8)


def _png(arr_or_img):
    im = arr_or_img if isinstance(arr_or_img, Image.Image) else Image.fromarray(arr_or_img)
    b = io.BytesIO(); im.save(b, format="PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def _overlay(crop, cam, size, alpha=0.45):
    base = np.asarray(crop.resize((size, size), Image.BILINEAR)).astype(np.float32)
    heat = np.asarray(Image.fromarray((cam * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR)) / 255.0
    return ((1 - alpha) * base + alpha * _jet(heat).astype(np.float32)).clip(0, 255).astype(np.uint8)


def _norm01(c):
    c = np.maximum(c, 0)
    return c / c.max() if c.max() > 1e-12 else np.zeros_like(c)


def gradcam_building(dmg, pre_im, post_im, bbox, display=224):
    """dmg: the loaded deployment_bundle/inference.py module. bbox: (x0,y0,x1,y1) in tile pixels."""
    s = dmg._ensure()
    model = s["models"]["stage2"]
    W, H = post_im.size
    box = dmg.padded_box(*bbox, W, H)
    pre_c, post_c = pre_im.crop(box), post_im.crop(box)
    a = dmg._prep_crop(pre_c).unsqueeze(0).to(s["device"])
    b = dmg._prep_crop(post_c).unsqueeze(0).to(s["device"])

    acts = {"conv": [], "vit": []}
    h1 = model.conv.backbone.stages[1].register_forward_hook(lambda m, i, o: acts["conv"].append(o))
    h2 = model.vit.backbone.blocks[-1].norm1.register_forward_hook(lambda m, i, o: acts["vit"].append(o))
    t0 = time.time()
    try:
        with torch.enable_grad():
            logits = model(a, b)
            probs = torch.softmax(logits.float(), 1)[0].detach().cpu().numpy()
            k = int(probs.argmax())
            # SiameseBranch.forward runs the POST image first, so index 0 = post-event activations
            conv_post, vit_post = acts["conv"][0], acts["vit"][0]
            g_conv, g_vit = torch.autograd.grad(logits[0, k], [conv_post, vit_post])
    finally:
        h1.remove(); h2.remove()

    # ConvNeXt: (1, C, h, w)
    A, G = conv_post[0].detach().float().cpu(), g_conv[0].float().cpu()
    cam_conv = _norm01(torch.relu((G.mean(dim=(1, 2))[:, None, None] * A).sum(0)).numpy())
    # ViT: (1, T, D) -> drop prefix (class/register) tokens -> sqrt(N) x sqrt(N) grid
    npre = int(getattr(model.vit.backbone, "num_prefix_tokens", 1))
    A, G = vit_post[0, npre:].detach().float().cpu(), g_vit[0, npre:].float().cpu()
    n = int(round(A.shape[0] ** 0.5))
    cam_vit = _norm01(torch.relu((A * G.mean(0)[None, :]).sum(1)).numpy().reshape(n, n))

    names = s["damage_names"]
    names = [names[str(i)] if isinstance(names, dict) else names[i] for i in range(len(probs))]
    return {"severity": names[k], "probabilities": {nm: round(float(p), 4) for nm, p in zip(names, probs)},
            "crop_box_xyxy": list(map(int, box)), "grid": {"convnext": list(cam_conv.shape), "vit": [n, n]},
            "images": {"pre": _png(pre_c.resize((display, display), Image.BILINEAR)),
                       "post": _png(post_c.resize((display, display), Image.BILINEAR)),
                       "convnext": _png(_overlay(post_c, cam_conv, display)),
                       "vit": _png(_overlay(post_c, cam_vit, display))},
            "seconds": round(time.time() - t0, 2),
            "note": "Two separate explanations (one per branch) of the same fused prediction, for the predicted class, "
                    "attributed to the post-event crop. The ViT map is a 4x4 token grid upsampled, so it is coarse."}
