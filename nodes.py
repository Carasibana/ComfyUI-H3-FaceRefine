"""Face tracking / cropping / stitching nodes for MiniMax H3 face refinement."""

from __future__ import annotations

import os

import numpy as np
import torch

import comfy.nested_tensor
import folder_paths

# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

_DETECTOR_CACHE: dict[str, object] = {}


def _detector_list() -> list[str]:
    """Face detectors, from Impact subpack's ultralytics_bbox registration if present."""
    names: list[str] = []
    for key in ("ultralytics_bbox", "ultralytics"):
        try:
            names.extend(folder_paths.get_filename_list(key))
        except Exception:
            pass
    seen, out = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out or ["face_yolov8m.pt"]


def _load_detector(name: str):
    if name in _DETECTOR_CACHE:
        return _DETECTOR_CACHE[name]
    path = None
    for key in ("ultralytics_bbox", "ultralytics"):
        try:
            path = folder_paths.get_full_path(key, name)
        except Exception:
            path = None
        if path:
            break
    if path is None:  # fall back to the standard models tree
        base = getattr(folder_paths, "models_dir", "models")
        for sub in ("ultralytics/bbox", "ultralytics", "ultralytics/segm"):
            cand = os.path.join(base, *sub.split("/"), name)
            if os.path.exists(cand):
                path = cand
                break
    if path is None:
        raise FileNotFoundError(
            f"Face detector '{name}' not found in ultralytics_bbox / ultralytics model folders."
        )
    from ultralytics import YOLO

    model = YOLO(path)
    _DETECTOR_CACHE[name] = model
    return model


_REC_CACHE: dict = {}


def _face_recogniser(pack: str = "buffalo_l"):
    """InsightFace recognition model, for identity matching. Cached."""
    if pack in _REC_CACHE:
        return _REC_CACHE[pack]
    import insightface

    # ComfyUI's models/insightface. InsightFace wants the directory CONTAINING a
    # "models" folder, and downloads the pack there on first use if it is absent.
    root = os.path.join(getattr(folder_paths, "models_dir", "models"), "insightface")
    app = insightface.app.FaceAnalysis(
        name=pack, root=root, allowed_modules=["detection", "recognition"],
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    _REC_CACHE[pack] = app
    return app


def _embed_faces(app, bgr: np.ndarray) -> list:
    """[(bbox, normed_embedding), ...] for every face insightface finds."""
    out = []
    for f in app.get(bgr):
        e = getattr(f, "normed_embedding", None)
        if e is None:
            continue
        out.append((f.bbox.tolist(), np.asarray(e, dtype=np.float32)))
    return out


def _best_match(cands: list, ref_emb: np.ndarray):
    """Index of the candidate closest to the reference by cosine similarity, and the score."""
    if not cands or ref_emb is None:
        return None, -1.0
    sims = [float(np.dot(e, ref_emb)) for _, e in cands]
    i = int(np.argmax(sims))
    return i, sims[i]


def _iou(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _continuity_cost(box, last):
    """Distance from the predicted position, with a size-change penalty."""
    cx, cy, sz = (box[0]+box[2])/2.0, (box[1]+box[3])/2.0, box[3]-box[1]
    d = ((cx - last[0]) ** 2 + (cy - last[1]) ** 2) ** 0.5
    return d + abs(sz - last[2]) * 2.0


def _build_clip_anchor(app, images, all_boxes, max_samples=24):
    """Average embedding of the subject, taken from the CLIP ITSELF.

    A stylised reference image sits in a different domain from rendered video frames -
    measured cosine similarity between an illustration reference and a frame of the same
    character was only 0.305, where same-domain photos of one person score 0.5-0.7. So an
    absolute threshold against an external reference is unreliable.

    Anchoring in-domain fixes that: sample frames where ONE face clearly dominates (no
    ambiguity about who the subject is), and average their embeddings.
    """
    B = len(all_boxes)
    step = max(1, B // max_samples)
    embs = []
    for i in range(0, B, step):
        boxes = all_boxes[i]
        if not boxes:
            continue
        heights = sorted((b[3] - b[1] for b in boxes), reverse=True)
        # unambiguous = only one face, or the biggest is clearly the biggest
        if len(heights) > 1 and heights[0] < heights[1] * 1.6:
            continue
        cands = _embed_faces(app, _to_bgr_u8(images[i]))
        if not cands:
            continue
        j = max(range(len(cands)), key=lambda k: cands[k][0][3] - cands[k][0][1])
        embs.append(cands[j][1])
    if not embs:
        return None, 0
    a = np.mean(np.stack(embs), axis=0)
    n = np.linalg.norm(a)
    return (a / n if n > 0 else a), len(embs)


def _track_continuity(all_boxes, start_frame: int, start_idx: int) -> list[int]:
    """Provisional subject track by continuity alone. Index into all_boxes[i], or -1.

    Pure geometry over boxes that are already detected, so it costs nothing, which is
    what makes it usable as a PRE-pass: it says which box is the subject on every frame
    before any embedding work happens.
    """
    B = len(all_boxes)
    track = [-1] * B
    if not (0 <= start_frame < B) or start_idx >= len(all_boxes[start_frame]):
        return track
    track[start_frame] = start_idx
    q = all_boxes[start_frame][start_idx]
    last = ((q[0] + q[2]) / 2.0, (q[1] + q[3]) / 2.0, q[3] - q[1])
    for i in range(start_frame + 1, B):
        boxes = all_boxes[i]
        if not boxes:
            continue
        k = min(range(len(boxes)), key=lambda j: _continuity_cost(boxes[j], last))
        track[i] = k
        q = boxes[k]
        last = ((q[0] + q[2]) / 2.0, (q[1] + q[3]) / 2.0, q[3] - q[1])
    return track


def _anchor_from_track(app, images, all_boxes, track, max_samples=24, min_face=32.0):
    """Average embedding of the TRACKED subject, sampled from frames where the tracked
    box is unambiguous.

    _build_clip_anchor assumes the subject is whoever is biggest. That is the wrong
    assumption the moment the user names a different one through select/select_index, so
    when they do, the anchor is built from their pick instead - otherwise identity
    matching would spend the rest of the clip dragging the crop back onto the dominant
    face the user deliberately did not choose.
    """
    clean = []
    for i, k in enumerate(track):
        if k < 0:
            continue
        boxes = all_boxes[i]
        tb = boxes[k]
        if tb[3] - tb[1] < min_face:          # tiny faces embed badly
            continue
        if any(_iou(tb, q) > 0.05 for j, q in enumerate(boxes) if j != k):
            continue
        clean.append(i)
    if not clean:
        return None, 0
    step = max(1, len(clean) // max_samples)
    embs = []
    for i in clean[::step][:max_samples]:
        cands = _embed_faces(app, _to_bgr_u8(images[i]))
        if not cands:
            continue
        tb = all_boxes[i][track[i]]
        j, best = None, 0.0
        for k, (bb, _) in enumerate(cands):
            v = _iou(bb, tb)
            if v > best:
                best, j = v, k
        if j is None or best < 0.3:           # insightface found other people, not ours
            continue
        embs.append(cands[j][1])
    if not embs:
        return None, 0
    a = np.mean(np.stack(embs), axis=0)
    n = np.linalg.norm(a)
    return (a / n if n > 0 else a), len(embs)


def _to_bgr_u8(img: torch.Tensor) -> np.ndarray:
    """ComfyUI IMAGE frame [H,W,C] float 0..1 -> BGR uint8 for ultralytics."""
    a = (img[..., :3].clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
    return a[..., ::-1].copy()


def _interp_gaps(vals: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fill non-detected frames by linear interpolation; hold at the ends."""
    n = len(vals)
    idx = np.arange(n)
    if not valid.any():
        return np.zeros(n, dtype=np.float64)
    return np.interp(idx, idx[valid], vals[valid])


def _smooth(vals: np.ndarray, window: int, method: str = "gaussian") -> np.ndarray:
    """Smooth a trajectory with reflected edges. window<=1 is a no-op.

    gaussian       - weighted kernel (sigma = window/6). Much better high-frequency
                     rejection than a boxcar, which has sinc sidelobes and leaves
                     residual jitter. Default.
    savgol         - local polynomial fit. Kills jitter while preserving ramps and
                     curves, so a push-in keeps its shape instead of being flattened.
    moving_average - plain boxcar. Kept for comparison.
    """
    if window <= 1 or len(vals) < 3:
        return vals
    window = min(int(window), len(vals))
    if window % 2 == 0:
        window += 1
    if window < 3:
        return vals
    pad = window // 2
    padded = np.pad(vals, pad, mode="reflect")

    if method == "savgol":
        try:
            from scipy.signal import savgol_filter

            polyorder = 2 if window > 3 else 1
            return np.asarray(savgol_filter(padded, window, polyorder))[pad : pad + len(vals)]
        except Exception:
            method = "gaussian"

    if method == "gaussian":
        x = np.arange(window, dtype=np.float64) - pad
        sigma = max(window / 6.0, 0.5)
        kernel = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
        kernel /= kernel.sum()
    else:
        kernel = np.ones(window, dtype=np.float64) / window

    return np.convolve(padded, kernel, mode="valid")[: len(vals)]


def _affine_crop(img: torch.Tensor, box: tuple, cw: int, ch: int) -> torch.Tensor:
    """Sub-pixel crop+resize in one bilinear sample. img [1,H,W,C] -> [1,ch,cw,C].

    Integer slicing quantises the box to whole pixels, and that rounding is by far the
    largest remaining source of frame-to-frame jitter once the trajectory is smoothed
    (measured jerk 0.58 vs 0.06 for the smoothed float trajectory). Sampling at float
    coordinates removes it entirely.
    """
    import torch.nn.functional as F

    x, y, bw, bh = box
    _, H, W, C = img.shape
    src = img[..., :3].movedim(-1, 1).float()
    theta = torch.tensor(
        [[[bw / W, 0.0, (2.0 * x + bw) / W - 1.0],
          [0.0, bh / H, (2.0 * y + bh) / H - 1.0]]],
        dtype=torch.float32, device=src.device,
    )
    grid = F.affine_grid(theta, (1, 3, int(ch), int(cw)), align_corners=False)
    out = F.grid_sample(src, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return out.movedim(1, -1).to(img.dtype)


def _gaussian_blur_mask(mask: torch.Tensor, feather: int) -> torch.Tensor:
    """Separable Gaussian blur on a [1,1,H,W] mask. Mirrors Impact Pack's
    tensor_gaussian_blur_mask / feather_mask (sigma = thickness/3)."""
    import torch.nn.functional as F

    if feather <= 0:
        return mask
    k = 2 * int(feather) + 1
    shortest = min(mask.shape[-2], mask.shape[-1])
    if shortest <= k:
        k = max(3, int(shortest / 2) | 1)
    sigma = max(k / 6.0, 0.5)
    x = torch.arange(k, device=mask.device, dtype=torch.float32) - k // 2
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).to(mask.dtype)
    pad = k // 2
    m = F.conv2d(F.pad(mask, (pad, pad, 0, 0), mode="replicate"), g.view(1, 1, 1, k))
    m = F.conv2d(F.pad(m, (0, 0, pad, pad), mode="replicate"), g.view(1, 1, k, 1))
    return m


def _face_region_mask(ch: int, cw: int, face_rect, dilation: int, feather: int,
                      shape: str, device, dtype) -> torch.Tensor:
    """FaceDetailer-style paste mask: solid over the FACE box inside the larger crop,
    dilated, then Gaussian-blurred. Everything outside keeps its original pixels.

    Impact Pack core.py:1256 builds exactly this - a crop-sized zeros canvas with 1s only
    where the detected face bbox sits - then blurs it. The generous crop exists to give the
    sampler CONTEXT; it is not what gets composited.
    """
    m = torch.zeros((1, 1, int(ch), int(cw)), device=device, dtype=torch.float32)
    fx, fy, fwd, fhd = face_rect
    fx -= dilation; fy -= dilation
    fwd += 2 * dilation; fhd += 2 * dilation

    if shape == "ellipse":
        yy = torch.arange(ch, device=device, dtype=torch.float32).view(-1, 1)
        xx = torch.arange(cw, device=device, dtype=torch.float32).view(1, -1)
        ccx, ccy = fx + fwd / 2.0, fy + fhd / 2.0
        rx, ry = max(fwd / 2.0, 1.0), max(fhd / 2.0, 1.0)
        m[0, 0] = (((xx - ccx) / rx) ** 2 + ((yy - ccy) / ry) ** 2 <= 1.0).float()
    else:
        x0 = max(0, int(round(fx))); y0 = max(0, int(round(fy)))
        x1 = min(int(cw), int(round(fx + fwd))); y1 = min(int(ch), int(round(fy + fhd)))
        if x1 > x0 and y1 > y0:
            m[0, 0, y0:y1, x0:x1] = 1.0

    return _gaussian_blur_mask(m, feather).clamp(0, 1).to(dtype)


def _feather_mask(h: int, w: int, feather: int, device, dtype) -> torch.Tensor:
    """[h,w] mask: 1 in the core, cosine ramp to 0 over `feather` px at every edge."""
    m = torch.ones((h, w), device=device, dtype=dtype)
    f = int(max(0, min(feather, min(h, w) // 2 - 1)))
    if f <= 0:
        return m
    ramp = 0.5 - 0.5 * torch.cos(
        torch.linspace(0, np.pi, f + 2, device=device, dtype=dtype)[1:-1]
    )
    m[:f, :] *= ramp.view(-1, 1)
    m[h - f :, :] *= ramp.flip(0).view(-1, 1)
    m[:, :f] *= ramp.view(1, -1)
    m[:, w - f :] *= ramp.flip(0).view(1, -1)
    return m


# ----------------------------------------------------------------------------
# subject selection + the index preview
# ----------------------------------------------------------------------------

# Ranking metrics. Every one is written so that MORE is FIRST under `descending`, which
# is why "largest" and "most_central" still mean what their names say at the default
# order while the raw coordinate metrics stay literal.
_SELECT_MODES = [
    "largest", "area", "confidence", "most_central",
    "x1", "x2", "y1", "y2", "center_x", "center_y",
]


def _select_metric(box, conf: float, W: int, H: int, mode: str) -> float:
    x0, y0, x1, y1 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    if mode == "area":
        return (x1 - x0) * (y1 - y0)
    if mode == "confidence":
        return float(conf)
    if mode == "most_central":
        # centrality, NOT distance: negated so that higher is nearer the centre and
        # "most_central" + descending keeps picking the most central face.
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        return -(((cx - W / 2.0) ** 2 + (cy - H / 2.0) ** 2) ** 0.5)
    if mode == "x1":
        return x0
    if mode == "x2":
        return x1
    if mode == "y1":
        return y0
    if mode == "y2":
        return y1
    if mode == "center_x":
        return (x0 + x1) / 2.0
    if mode == "center_y":
        return (y0 + y1) / 2.0
    return y1 - y0      # "largest": face HEIGHT, the same metric the tracker uses throughout


def _rank_boxes(boxes, confs, W: int, H: int, mode: str, order: str) -> list[int]:
    """Indices of `boxes`, ranked. Position 0 is what select_index 0 picks.

    `descending` puts the HIGHEST value of the metric first, `ascending` the lowest. Ties
    break on x then y, so two equally-good boxes do not swap ranks between frames.
    """
    sign = -1.0 if order == "descending" else 1.0
    vals = [_select_metric(b, (confs[i] if i < len(confs) else 1.0), W, H, mode)
            for i, b in enumerate(boxes)]
    return sorted(range(len(boxes)),
                  key=lambda k: (sign * vals[k], float(boxes[k][0]), float(boxes[k][1])))


# 5x7 digits, drawn straight into the tensor. A bitmap font rather than PIL because the
# default PIL font is a fixed few pixels tall - unreadable on a 1080p frame - and the
# sized variants are not available on every Pillow this pack has to run against.
_GLYPHS = {
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11111", "00010", "00100", "00010", "00001", "10001", "01110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "11110", "00001", "00001", "10001", "01110"),
    "6": ("00110", "01000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00010", "01100"),
}


def _draw_rect(img: torch.Tensor, x0, y0, x1, y1, colour, thickness: int) -> None:
    """Hollow rectangle on an [H,W,3] frame, clipped to the frame."""
    H, W, _ = img.shape
    xa, xb = max(0, min(int(round(x0)), W)), max(0, min(int(round(x1)), W))
    ya, yb = max(0, min(int(round(y0)), H)), max(0, min(int(round(y1)), H))
    if xb <= xa or yb <= ya:
        return
    t = max(1, min(int(thickness), xb - xa, yb - ya))
    c = torch.tensor(colour, dtype=img.dtype, device=img.device)
    img[ya:ya + t, xa:xb] = c
    img[yb - t:yb, xa:xb] = c
    img[ya:yb, xa:xa + t] = c
    img[ya:yb, xb - t:xb] = c


def _label_size(text: str, scale: int) -> tuple[int, int]:
    pad = 2 * scale
    adv = 6 * scale
    return (max(0, len(text) * adv - scale) + 2 * pad, 7 * scale + 2 * pad)


def _draw_label(img: torch.Tensor, x, y, text: str, scale: int, fg, bg) -> None:
    """Digits on a filled plate, nudged back inside the frame if they would fall off."""
    H, W, _ = img.shape
    bw, bh = _label_size(text, scale)
    x0 = 0 if bw >= W else max(0, min(int(round(x)), W - bw))
    y0 = 0 if bh >= H else max(0, min(int(round(y)), H - bh))
    x1, y1 = min(W, x0 + bw), min(H, y0 + bh)
    if x1 <= x0 or y1 <= y0:
        return
    img[y0:y1, x0:x1] = torch.tensor(bg, dtype=img.dtype, device=img.device)
    fgc = torch.tensor(fg, dtype=img.dtype, device=img.device)
    pad, adv = 2 * scale, 6 * scale
    px0 = x0 + pad
    for ch in text:
        rows = _GLYPHS.get(ch)
        if rows is not None:
            for r, row in enumerate(rows):
                for c, on in enumerate(row):
                    if on != "1":
                        continue
                    gx, gy = px0 + c * scale, y0 + pad + r * scale
                    if gx >= W or gy >= H:
                        continue
                    img[gy:min(gy + scale, H), gx:min(gx + scale, W)] = fgc
        px0 += adv


def _index_preview(images: torch.Tensor, all_boxes, all_confs, mode: str, order: str,
                   limit: int = 32):
    """One card per face index: the first frame in which that index exists, every box
    drawn and numbered with its rank, the box for that index highlighted.

    A rank is a per-frame property, not an identity, so there is no way to read
    select_index off the node itself - this output is how you find out that the person
    you want is number 2.
    """
    B, H, W, _ = images.shape
    max_faces = max((len(b) for b in all_boxes), default=0)
    n = min(max_faces, int(limit))
    thick = max(2, int(round(min(H, W) / 240.0)))
    scale = max(2, int(round(min(H, W) / 140.0)))
    _, lh = _label_size("0", scale)

    cards = []
    for idx in range(n):
        f = next(i for i in range(B) if len(all_boxes[i]) > idx)
        img = images[f, ..., :3].clone()
        ranked = _rank_boxes(all_boxes[f], all_confs[f], W, H, mode, order)
        for rank, bi in enumerate(ranked):
            box = all_boxes[f][bi]
            hit = rank == idx
            col = (0.0, 1.0, 0.2) if hit else (0.15, 0.45, 1.0)
            _draw_rect(img, box[0], box[1], box[2], box[3], col, thick * (2 if hit else 1))
            _draw_label(img, box[0], box[1] - lh, str(rank), scale,
                        (0.0, 0.0, 0.0) if hit else (1.0, 1.0, 1.0), col)
        cards.append(img)
    if not cards:      # no detections anywhere; the caller raises on that separately
        cards.append(images[0, ..., :3].clone())
    return torch.stack(cards, 0), max_faces, n


# ----------------------------------------------------------------------------
# 1. track + crop
# ----------------------------------------------------------------------------


class H3FaceTrackCrop:
    """Detect a face per frame, build a smoothed per-frame crop, emit a constant-size batch.

    The crop SIZE varies per frame so the face fills a constant fraction of every
    crop; every crop is then resized to one canvas size, because H3 generates a
    single fixed WxH for a whole sequence. Result: the face is always large in
    H3's input regardless of how small it was in the source frame.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "detector": (_detector_list(),),
                "confidence": ("FLOAT", {"default": 0.35, "min": 0.05, "max": 0.95, "step": 0.05}),
                "crop_factor": ("FLOAT", {"default": 2.5, "min": 1.2, "max": 8.0, "step": 0.1,
                    "tooltip": "Crop side as a multiple of detected face HEIGHT. 2.5 puts the "
                               "face at ~40% of the crop, comfortably inside H3's good regime. "
                               "Bigger = more context so the seam lands in hair/background, but "
                               "less magnification. 2.0-3.0 is the useful range."}),
                "canvas_width": ("INT", {"default": 512, "min": 128, "max": 1344, "step": 32,
                    "tooltip": "Resolution H3 generates at. 512 is cheap; 768 is H3's native "
                               "short edge and gives the best faces. Ignored when canvas_mode "
                               "is not 'manual'. Cost scales with area: 768 is 2.25x the "
                               "latent tokens of 512."}),
                "canvas_height": ("INT", {"default": 512, "min": 128, "max": 1344, "step": 32}),
                "canvas_mode": (["manual", "auto_no_downscale", "auto_capped_768"],
                    {"default": "manual",
                     "tooltip": "manual: use canvas_width/height as given.\n"
                                "auto_no_downscale: size the canvas from the LARGEST crop in "
                                "the clip so no frame is ever downscaled (magnification never "
                                "drops below 1.0x). Can get expensive on clips that include "
                                "close-ups.\n"
                                "auto_capped_768: same, but clamped to 768 - H3's native short "
                                "edge and a sane VRAM ceiling."}),
                "smooth_window": ("INT", {"default": 21, "min": 1, "max": 201, "step": 2,
                    "tooltip": "Frames of smoothing on the crop CENTRE. 21 at 24fps is ~0.9s. "
                               "Raise if the box still shivers; lower if it lags behind fast "
                               "head movement."}),
                "size_smooth_window": ("INT", {"default": 51, "min": 1, "max": 201, "step": 2,
                    "tooltip": "Frames of smoothing on the crop SIZE. Wants MORE than the "
                               "centre: size jitter makes the crop breathe, which changes the "
                               "resample factor every frame and reads as shimmer. Real zoom "
                               "moves are slow, so heavy smoothing here costs nothing."}),
                "smooth_method": (["gaussian", "savgol", "moving_average"], {"default": "gaussian",
                    "tooltip": "gaussian: best jitter rejection. savgol: preserves the shape of "
                               "a push-in better at large windows. moving_average: the old "
                               "boxcar, leaves residual jitter."}),
                "size_mode": (["max_of_clip", "per_frame"], {"default": "per_frame",
                    "tooltip": "per_frame: constant face-fraction in every crop (correct for "
                               "push-ins). max_of_clip: one size for the whole clip, only "
                               "useful when the shot is genuinely static."}),
            },
            "optional": {
                "identity_reference": ("IMAGE", {
                    "tooltip": "A clear face image of the person to track. When supplied, the "
                               "subject is chosen by FACE IDENTITY rather than by size, so a "
                               "crowd scene locks onto the right person even when someone else "
                               "is briefly larger or nearer.\n\n"
                               "Without it, 'largest' has no notion of WHO it is following - "
                               "it just takes the biggest box each frame, which switches "
                               "subject whenever the framing changes.\n\n"
                               "FOR MULTIPLE PEOPLE: run the pipeline once per subject, each "
                               "with that person's reference here and their own refs on the H3 "
                               "node, and chain them - feed run 1's stitched output in as run "
                               "2's base_images. The composites accumulate."}),
                "identity_track": ("BOOLEAN", {"default": True,
                    "tooltip": "Hold one subject through a crowd. Continuity (nearest box to "
                               "the previous position) decides most frames; the face-identity "
                               "embedding is consulted only when two candidates are similarly "
                               "plausible or their boxes overlap - which is both the accurate "
                               "and the cheap arrangement, since the embedding model then runs "
                               "on a handful of frames instead of all of them. "
                               "The anchor is taken FROM THE CLIP by default (frames where one "
                               "face clearly dominates), because an external stylised reference "
                               "sits in a different domain - measured similarity between an "
                               "illustration and a render of the same character was only 0.305, "
                               "where same-domain faces score 0.5-0.7."}),
                "identity_threshold": ("FLOAT", {"default": 0.28, "min": 0.0, "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Minimum cosine similarity to accept a face as the reference "
                               "person. Below this the frame falls back to continuity (nearest "
                               "to the previous position at a similar size), which is what "
                               "carries tracking through profiles and partial occlusion where "
                               "embeddings become unreliable."}),
                "select": (_SELECT_MODES, {"default": "largest",
                    "tooltip": "How the faces in a frame are RANKED. select_order picks the "
                               "direction and select_index picks one out of the ranking.\n"
                               "largest / area / confidence: face height, box area, detector "
                               "score.\n"
                               "most_central: nearness to the frame centre.\n"
                               "x1 / x2 / y1 / y2 / center_x / center_y: raw box coordinates - "
                               "left edge, right edge, top edge, bottom edge, centre. Use "
                               "x1 with ascending to number faces left to right, y1 with "
                               "ascending to number them top to bottom.\n"
                               "Used only when no identity_reference is connected, and only to "
                               "choose the subject on the first frame that contains the "
                               "requested index - continuity carries it from there."}),
                "fallback_detector": (["none"] + _detector_list(), {"default": "none",
                    "tooltip": "Used only on frames where the FACE detector finds nothing "
                               "(subject turned away). A person/body model such as "
                               "segm\\person_yolov8m-seg.pt gives a real head position from the "
                               "top of the body box, which beats interpolating blindly between "
                               "the last and next face. Set 'none' to interpolate instead."}),
                "fallback_head_frac": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.5, "step": 0.05,
                    "tooltip": "Head centre as a multiple of face height below the top of the "
                               "person box. 0.5 puts it half a face-height down, which is about "
                               "right for a head seen from behind."}),
                # These two sit at the END of the list on purpose: ComfyUI stores widget
                # values positionally, so inserting next to `select` would shift the two
                # fallback_* values in every workflow already saved against this node.
                "select_order": (["descending", "ascending"], {"default": "descending",
                    "tooltip": "Direction of the select ranking. descending puts the HIGHEST "
                               "value of the metric at index 0, ascending the lowest.\n"
                               "largest + descending = biggest face first (the default), "
                               "largest + ascending = smallest first.\n"
                               "most_central + descending = most central first, + ascending = "
                               "furthest from the centre first.\n"
                               "x1 + ascending = leftmost first, x1 + descending = rightmost "
                               "first. Same for y1/y2 top and bottom."}),
                "select_index": ("INT", {"default": 0, "min": 0, "max": 63,
                    "tooltip": "Which face in that ranking to track. 0 is the first, 1 the "
                               "second, and so on.\n"
                               "The subject is locked on the first frame that actually contains "
                               "this index, so a clip that opens on one face and only later "
                               "shows a crowd still finds face 1. Frames before that lock-on are "
                               "interpolated from it and faded out of the composite, the same as "
                               "a detection dropout; the report counts them. Clamped, with a "
                               "warning in the report, if the clip never shows that many faces "
                               "at once. Ignored entirely when identity_reference is connected.\n"
                               "Connect face_index_preview to a PreviewImage to see which "
                               "number is who."}),
            },
        }

    # canvas_w / canvas_h MUST be wired into the H3 node's width/height. With canvas_mode
    # on auto the tracker decides the size, and nothing downstream can know it otherwise -
    # the crop and the AV latent would disagree and H3InjectVideoLatent would refuse.
    # face_index_preview is appended rather than slotted next to `preview` because
    # workflow links are stored by output INDEX - inserting would rewire saved graphs.
    RETURN_TYPES = ("IMAGE", "H3FACEXFORM", "IMAGE", "STRING", "INT", "INT", "IMAGE")
    RETURN_NAMES = ("crops", "transform", "preview", "report", "canvas_w", "canvas_h",
                    "face_index_preview")
    FUNCTION = "run"
    CATEGORY = "MiniMax H3/Face Refine"
    DESCRIPTION = (
        "Per-frame face track -> smoothed, normalised crop -> constant-size batch for H3, "
        "plus the transform needed to paste the result back."
    )

    def run(self, images, detector, confidence, crop_factor, canvas_width, canvas_height,
            canvas_mode, smooth_window, size_smooth_window, smooth_method, size_mode,
            select="largest", fallback_detector="none", fallback_head_frac=0.5,
            identity_reference=None, identity_threshold=0.28, identity_track=True,
            select_order="descending", select_index=0):
        model = _load_detector(detector)
        B, H, W, _ = images.shape

        cx = np.zeros(B); cy = np.zeros(B); sz = np.zeros(B); fw = np.zeros(B)
        valid = np.zeros(B, dtype=bool)       # a real FACE was seen
        via_body = np.zeros(B, dtype=bool)    # head located from a body box instead

        import comfy.model_management as _mm

        # ---- detect once, keep every box -------------------------------------------
        # Detection is the expensive part of this node and the boxes themselves are tiny,
        # so the whole clip is detected up front and cached. Subject selection, the
        # identity anchor and the index preview then all read the same set instead of
        # each re-running the detector over the frames they care about.
        all_boxes: list[list[list[float]]] = []
        all_confs: list[list[float]] = []
        for i in range(B):
            _mm.throw_exception_if_processing_interrupted()
            res = model.predict(_to_bgr_u8(images[i]), conf=confidence, verbose=False)[0]
            if len(res.boxes):
                bx = [[float(v) for v in q] for q in res.boxes.xyxy.tolist()]
                cf = getattr(res.boxes, "conf", None)
                all_boxes.append(bx)
                all_confs.append([float(c) for c in cf.tolist()] if cf is not None
                                 else [1.0] * len(bx))
            else:
                all_boxes.append([])
                all_confs.append([])

        max_faces = max((len(b) for b in all_boxes), default=0)
        if max_faces == 0:
            raise ValueError(
                "No face detected in any frame. Lower `confidence`, or this clip has no "
                "usable face and should be skipped."
            )

        # ---- which face is the subject ---------------------------------------------
        # select + select_order rank the boxes within a frame; select_index takes one out
        # of that ranking. The pick happens ONCE, on the first frame that actually
        # contains the requested index, and continuity carries the subject from there: a
        # rank is a per-frame property, so re-ranking every frame would hop between people
        # the moment two of them cross or change size.
        requested_index = int(select_index)
        select_index = max(0, min(requested_index, max_faces - 1))
        first_face = next(i for i in range(B) if all_boxes[i])
        index_lock = next(i for i in range(B) if len(all_boxes[i]) > select_index)
        selection_default = (select == "largest" and select_order == "descending"
                             and select_index == 0)
        # A user naming a subject outranks the automatic "the biggest face is the subject"
        # guess, so when they have, the identity anchor is built from THEIR pick instead.
        selection_leads = (not selection_default) and identity_reference is None

        # ---- identity anchor -------------------------------------------------------
        # Without one, "largest" has no idea WHO it is following: it takes the biggest box
        # each frame, so in a crowd it hops subject whenever the framing changes. Observed
        # in testing: a single clip switched subject 4 times.
        ref_emb, app = None, None
        n_ident, n_cont, n_conflict = 0, 0, 0
        multi = len(all_boxes[0]) > 1 or (selection_leads and max_faces > 1)

        if identity_track and (multi or identity_reference is not None):
            try:
                app = _face_recogniser()
                if identity_reference is not None:
                    cands = _embed_faces(app, _to_bgr_u8(identity_reference[0]))
                    if cands:
                        j = max(range(len(cands)),
                                key=lambda k: cands[k][0][3] - cands[k][0][1])
                        ref_emb = cands[j][1]
                        print("[H3FaceRefine] identity anchor from the supplied reference")
                if ref_emb is None and selection_leads:
                    # Anchor on the SELECTED subject, followed by continuity. Falling back
                    # to the clip anchor here would re-anchor on the dominant face, i.e.
                    # exactly the person select_index was used to avoid, so a failure here
                    # leaves ref_emb None and the track runs on continuity alone.
                    seed = _rank_boxes(all_boxes[index_lock], all_confs[index_lock], W, H,
                                       select, select_order)[select_index]
                    ref_emb, used = _anchor_from_track(
                        app, images, all_boxes,
                        _track_continuity(all_boxes, index_lock, seed))
                    print(f"[H3FaceRefine] identity anchor built from the selected subject "
                          f"({used} unambiguous frames)" if ref_emb is not None else
                          "[H3FaceRefine] no clean frames to anchor the selected subject - "
                          "tracking by continuity alone")
                elif ref_emb is None:
                    ref_emb, used = _build_clip_anchor(app, images, all_boxes)
                    if ref_emb is not None:
                        print(f"[H3FaceRefine] identity anchor built from the clip itself "
                              f"({used} unambiguous frames)")
            except Exception as exc:
                print(f"[H3FaceRefine] identity matching unavailable ({exc})")

        # Waiting for a frame that contains select_index only makes sense when the RANKING
        # is what picks the subject. A usable identity anchor overrules the ranking at
        # lock-on, so deferring there would discard leading frames identity resolves
        # perfectly - frames where the subject is often alone, i.e. the most reliable ones
        # in the clip.
        ranking_picks = selection_leads or ref_emb is None
        lock_frame = index_lock if ranking_picks else first_face

        last = None   # (cx, cy, size) of the subject on the previous resolved frame

        for i in range(B):
            _mm.throw_exception_if_processing_interrupted()
            boxes = all_boxes[i]
            if not boxes:
                continue
            if last is None and i < lock_frame:
                # Too few faces here to honour select_index. Leaving these frames unresolved
                # hands them to the same interpolation that covers detection dropouts, which
                # beats locking onto whoever happens to be alone in the opening shot.
                continue

            b = None
            if len(boxes) == 1:
                b = boxes[0]
                n_cont += 1
            elif last is None:
                # first resolved frame: identity if we have it, else the ranking rule
                if ref_emb is not None and not selection_leads:
                    cands = _embed_faces(app, _to_bgr_u8(images[i]))
                    k, _ = _best_match(cands, ref_emb)
                    if k is not None:
                        b = cands[k][0]
                        n_ident += 1
                if b is None:
                    # Clamped: when identity was in charge the lock frame is only
                    # guaranteed to hold a face, not to hold select_index of them, and
                    # insightface can come back empty on a frame the face detector liked.
                    ranked = _rank_boxes(boxes, all_confs[i], W, H, select, select_order)
                    b = boxes[ranked[min(select_index, len(ranked) - 1)]]
                    n_cont += 1
            else:
                # Continuity first: the nearest box to where the subject was, penalised for
                # size change. Cheap and correct while people stay separated.
                ranked = sorted(boxes, key=lambda q: _continuity_cost(q, last))
                best, second = ranked[0], ranked[1]
                c0, c1 = _continuity_cost(best, last), _continuity_cost(second, last)

                # AMBIGUOUS when two candidates are similarly plausible, or their boxes
                # overlap - exactly when continuity alone picks the wrong person. Only then
                # is the embedding worth computing.
                conflict = (c1 < c0 * 2.0) or (_iou(best, second) > 0.2)

                if conflict and ref_emb is not None:
                    n_conflict += 1
                    near = [q for q in boxes if _continuity_cost(q, last) < c0 * 3.0] or boxes
                    cands = [c for c in _embed_faces(app, _to_bgr_u8(images[i]))
                             if any(_iou(c[0], q) > 0.3 for q in near)]
                    k, score = _best_match(cands, ref_emb)
                    if k is not None and score >= identity_threshold:
                        b = cands[k][0]
                        n_ident += 1
                if b is None:
                    # No conflict, or the embedding was not confident enough. Embeddings
                    # degrade on profiles and occlusion - precisely where the subject is
                    # hardest to hold - so continuity is the safer default there.
                    b = best
                    n_cont += 1

            last = ((b[0]+b[2])/2.0, (b[1]+b[3])/2.0, b[3]-b[1])
            cx[i] = (b[0] + b[2]) / 2.0
            cy[i] = (b[1] + b[3]) / 2.0
            sz[i] = b[3] - b[1]          # face HEIGHT: more stable than width as the head turns
            fw[i] = b[2] - b[0]          # face WIDTH: needed for the FaceDetailer-style paste mask
            valid[i] = True

        found = int(valid.sum())
        if found == 0:
            raise ValueError(
                "No face detected in any frame. Lower `confidence`, or this clip has no "
                "usable face and should be skipped."
            )

        # Body fallback for frames the face detector missed. Interpolated size feeds the
        # head-position estimate, so size comes from frames where a face WAS measured while
        # position comes from the body actually visible in this frame.
        #
        # Keyed on "no face was detected", NOT on "no subject was resolved": the frames the
        # lock deferral holds back DID have faces, and the body model would re-centre them
        # on the largest body in shot - usually the very person select_index was used to
        # avoid - then mark them via_body, which hides them from both the interpolation and
        # the dropout warning while smoothing that error into the frames after the lock.
        no_face = np.array([not b for b in all_boxes], dtype=bool)
        sz_seed = _interp_gaps(sz, valid)
        if fallback_detector != "none" and no_face.any():
            try:
                bmodel = _load_detector(fallback_detector)
                for i in np.nonzero(no_face)[0]:
                    res = bmodel.predict(_to_bgr_u8(images[i]), conf=confidence, verbose=False)[0]
                    if not len(res.boxes):
                        continue
                    bb = res.boxes.xyxy.tolist()
                    cls = (res.boxes.cls.tolist() if getattr(res.boxes, "cls", None) is not None
                           else [0] * len(bb))
                    people = [q for q, cc in zip(bb, cls) if int(cc) == 0] or bb
                    p = max(people, key=lambda q: (q[2] - q[0]) * (q[3] - q[1]))
                    cx[i] = (p[0] + p[2]) / 2.0
                    cy[i] = p[1] + fallback_head_frac * max(sz_seed[i], 8.0)
                    sz[i] = sz_seed[i]
                    via_body[i] = True
            except Exception as exc:  # never let the fallback kill the run
                print(f"[H3FaceRefine] body fallback '{fallback_detector}' failed: {exc}")

        known = valid | via_body
        raw_cx = _interp_gaps(cx, known)
        raw_cy = _interp_gaps(cy, known)
        raw_sz = _interp_gaps(sz, valid)   # size ALWAYS from real face measurements
        raw_fw = _interp_gaps(fw, valid)
        sm_fw = _smooth(raw_fw, size_smooth_window, smooth_method)
        cx = _smooth(raw_cx, smooth_window, smooth_method)
        cy = _smooth(raw_cy, smooth_window, smooth_method)
        sz = _smooth(raw_sz, size_smooth_window, smooth_method)
        if size_mode == "max_of_clip":
            sz[:] = sz.max()

        # frame-to-frame movement, before vs after: the number that corresponds to
        # visible jitter. Residual is what still moves after smoothing.
        def _jit(a):
            return float(np.abs(np.diff(a)).mean()) if len(a) > 1 else 0.0

        jit_before = (_jit(raw_cx) + _jit(raw_cy)) / 2.0
        jit_after = (_jit(cx) + _jit(cy)) / 2.0
        sz_before, sz_after = _jit(raw_sz), _jit(sz)

        # Size the canvas from the clip itself so no frame is ever downscaled. The largest
        # crop is (largest smoothed face height) * crop_factor, clamped to the frame; matching
        # the canvas to it keeps magnification >= 1.0x everywhere.
        if canvas_mode != "manual":
            need = float(min(sz.max() * crop_factor, H))
            snapped = int(np.ceil(need / 32.0) * 32)
            if canvas_mode == "auto_capped_768":
                snapped = min(snapped, 768)
            snapped = max(128, min(snapped, 1344))
            if snapped != canvas_height:
                print(f"[H3FaceRefine] canvas_mode={canvas_mode}: "
                      f"{canvas_width}x{canvas_height} -> {snapped}x{snapped} "
                      f"(largest crop {need:.0f}px)")
            canvas_width = canvas_height = snapped

        aspect = canvas_width / float(canvas_height)
        boxes: list[tuple[int, int, int, int]] = []
        crops = torch.zeros((B, canvas_height, canvas_width, 3), dtype=images.dtype)
        preview = images[..., :3].clone()

        for i in range(B):
            bh = sz[i] * crop_factor
            bw = bh * aspect
            # keep aspect while fitting inside the frame
            if bw > W:
                bw, bh = float(W), float(W) / aspect
            if bh > H:
                bh, bw = float(H), float(H) * aspect
            # FLOAT box - deliberately not rounded. Integer rounding is the dominant
            # residual jitter once the trajectory is smoothed.
            x = min(max(cx[i] - bw / 2.0, 0.0), max(0.0, W - bw))
            y = min(max(cy[i] - bh / 2.0, 0.0), max(0.0, H - bh))
            box = (float(x), float(y), float(bw), float(bh))
            boxes.append(box)

            crops[i : i + 1] = _affine_crop(
                images[i : i + 1], box, canvas_width, canvas_height
            ).to(crops.dtype)

            # preview: draw the crop rectangle (rounded for drawing only)
            xi, yi = int(round(x)), int(round(y))
            wi, hi = max(4, int(round(bw))), max(4, int(round(bh)))
            xi = min(xi, W - wi); yi = min(yi, H - hi)
            # green = real face, yellow = head located from the body box, red = interpolated
            if valid[i]:
                r, g = 0.0, 1.0
            elif via_body[i]:
                r, g = 1.0, 1.0
            else:
                r, g = 1.0, 0.0
            for (yy0, yy1, xx0, xx1) in (
                (yi, yi + 2, xi, xi + wi), (yi + hi - 2, yi + hi, xi, xi + wi),
                (yi, yi + hi, xi, xi + 2), (yi, yi + hi, xi + wi - 2, xi + wi),
            ):
                preview[i, yy0:yy1, xx0:xx1, 0] = r
                preview[i, yy0:yy1, xx0:xx1, 1] = g
                preview[i, yy0:yy1, xx0:xx1, 2] = 0.0

        # Per-frame confidence weight. When the subject turns away there is no face to
        # refine, and asking H3 to "improve a face" on the back of a head invites it to
        # hallucinate one. Fade the composite out across those runs instead. Smoothed so
        # it ramps over ~half a second rather than popping.
        weights = _smooth(valid.astype(np.float64), max(9, smooth_window // 2), "gaussian")
        weights = np.clip(weights, 0.0, 1.0)

        runs, cur = [], 0
        for v in known:
            if v:
                if cur:
                    runs.append(cur)
                cur = 0
            else:
                cur += 1
        if cur:
            runs.append(cur)
        longest_gap = max(runs) if runs else 0

        mags = [canvas_height / float(b[3]) for b in boxes]
        transform = {
            "boxes": boxes,
            "canvas": (int(canvas_width), int(canvas_height)),
            "src_size": (int(W), int(H)),
            "frames": int(B),
            "weights": [float(w) for w in weights],
            "detected": [bool(v) for v in valid],
            # Face rect per frame in CANVAS pixel coords, centred in the crop. This is what
            # the stitch pastes through - matching FaceDetailer, which builds its paste mask
            # from the face bbox inside the larger crop region, NOT from the crop itself.
            "face_rect": [
                (
                    float(canvas_width) * 0.5 - 0.5 * float(sm_fw[i]) / max(b[2], 1e-6) * canvas_width,
                    float(canvas_height) * 0.5 - 0.5 * float(sz[i]) / max(b[3], 1e-6) * canvas_height,
                    float(sm_fw[i]) / max(b[2], 1e-6) * canvas_width,
                    float(sz[i]) / max(b[3], 1e-6) * canvas_height,
                )
                for i, b in enumerate(boxes)
            ],
            "crop_factor": float(crop_factor),
        }

        # A magnification below 1.0 means the crop is DOWNSCALED into the canvas, i.e. we
        # throw away real detail before handing it to H3 and upscale the result back on
        # stitch. On those frames this pipeline is a net loss versus leaving them alone.
        gapwarn = ""
        if longest_gap >= 12:
            gapwarn = (
                f"\n!! longest dropout is {longest_gap} frames ({longest_gap/24.0:.1f}s). The crop "
                f"box is linearly interpolated across it, so it may drift if the subject moved "
                f"while turned away. Detection weighting fades the composite out there, so those "
                f"frames keep their original pixels - check the preview over that stretch."
            )

        n_down = sum(1 for m in mags if m < 1.0)
        warn = ""
        if n_down:
            need = max(b[3] for b in boxes)
            warn = (
                f"\n!! {n_down}/{B} frames ({n_down/B*100:.0f}%) have magnification < 1.0x - "
                f"their crops are DOWNSCALED into the canvas, losing real detail.\n"
                f"   Fix: raise canvas to >= {need}px (rounded up to a multiple of 32), or lower "
                f"crop_factor, or skip this clip if it is close-up throughout."
            )

        # One card per face index, so the number to type into select_index can be read
        # off a PreviewImage instead of guessed.
        face_index_preview, max_faces, n_cards = _index_preview(
            images, all_boxes, all_confs, select, select_order)

        # Frames that had a face but were held back waiting for select_index. They are
        # interpolated and faded out of the composite like any dropout, so they must be
        # stated rather than left to look like detection failures.
        prelock = sum(1 for i in range(lock_frame) if all_boxes[i])
        idxwarn = ""
        if requested_index > 0 and not ranking_picks:
            idxwarn += (
                f"\n!! select_index={requested_index} was ignored: identity_reference "
                f"decides the subject when one is connected."
            )
        if prelock >= 12:
            idxwarn += (
                f"\n!! {prelock} frames before the lock-on contain fewer than "
                f"{select_index + 1} faces, so they are interpolated from frame {lock_frame} "
                f"and faded out of the composite. Lower select_index if that stretch matters."
            )
        if requested_index != select_index:
            idxwarn = (
                f"\n!! select_index={requested_index} is out of range: at most {max_faces} "
                f"face(s) were ever detected in a single frame, so index {select_index} was "
                f"used instead."
            )
        if n_cards < max_faces:
            idxwarn += (
                f"\n!! face_index_preview shows {n_cards} of {max_faces} indices (capped). "
                f"The higher indices are still selectable."
            )

        box_jit = float(np.mean([abs(boxes[i][0] - boxes[i-1][0]) + abs(boxes[i][1] - boxes[i-1][1])
                                 for i in range(1, len(boxes))])) if len(boxes) > 1 else 0.0
        report = (
            f"subject: select={select} {select_order} index={select_index}"
            f"{'' if ranking_picks else ' (ignored - identity_reference decides)'}, "
            f"locked on at frame {lock_frame} of {B}"
            f"{f', {prelock} earlier frames interpolated' if prelock else ''}\n"
            f"faces: max {max_faces} in one frame; face_index_preview has {n_cards} card(s)\n"
            f"tracking: {n_cont} by continuity, {n_conflict} ambiguous "
            f"({n_ident} resolved by face identity)\n"
            f"frames={B}  face={found} ({found/B*100:.0f}%)  "
            f"body-fallback={int(via_body.sum())}  interpolated={B-int(known.sum())}\n"
            f"face height  min={sz.min():.0f}px  mean={sz.mean():.0f}px  max={sz.max():.0f}px\n"
            f"face fills   ~{100.0/crop_factor:.0f}% of every crop (crop_factor={crop_factor})\n"
            f"crop box     min={min(b[3] for b in boxes)}px  max={max(b[3] for b in boxes)}px\n"
            f"magnification into {canvas_width}x{canvas_height}: "
            f"min={min(mags):.2f}x  mean={sum(mags)/len(mags):.2f}x  max={max(mags):.2f}x\n"
            f"jitter ({smooth_method}) centre {jit_before:.2f} -> {jit_after:.2f} px/frame"
            f"   size {sz_before:.2f} -> {sz_after:.2f} px/frame\n"
            f"box movement {box_jit:.2f} px/frame (sub-pixel float boxes - no integer rounding)\n"
            f"dropout runs: {len(runs)}  longest={longest_gap} frames ({longest_gap/24.0:.1f}s "
            f"at 24fps)  -> composite fades out across these"
            f"{idxwarn}{gapwarn}{warn}"
        )
        # Always print. The report is also returned as an output, but that output is
        # usually left unconnected, and then a run gives no account of how it tracked -
        # which is exactly the information needed to tell whether identity matching did
        # any work on a crowd scene.
        print("[H3FaceRefine] " + report.replace("\n", "\n[H3FaceRefine] "))
        return (crops, transform, preview, report, int(canvas_width), int(canvas_height),
                face_index_preview)


# ----------------------------------------------------------------------------
# 2. stitch back
# ----------------------------------------------------------------------------


class H3FaceStitch:
    """Paste refined crops back using the per-frame transform, with feather + colour match.

    Mirrors Impact Pack's detailer paste - only the face region composites, through a
    dilated then Gaussian-blurred mask - but warps rather than slices: one batched
    `grid_sample` maps each crop back onto the float box it came from, so a trajectory
    smoothed to sub-pixel precision is not re-quantised on the way home.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_images": ("IMAGE",),
                "refined_crops": ("IMAGE",),
                "transform": ("H3FACEXFORM",),
                "paste_region": (["face_only", "face_ellipse", "full_crop"],
                    {"default": "face_only",
                     "tooltip": "WHAT gets composited back. face_only / face_ellipse paste just "
                                "the detected face box (FaceDetailer's behaviour - the wider "
                                "crop exists to give the sampler context, not to be pasted). "
                                "full_crop pastes the whole crop including hair, shoulders and "
                                "background, which risks a visible rectangle if H3 alters them."}),
                "mask_dilation": ("INT", {"default": 16, "min": 0, "max": 256, "step": 2,
                    "tooltip": "Grow the face box before blurring, in canvas px. Impact Pack "
                               "dilates the same way so the blur has room and the blend does "
                               "not eat into the face itself."}),
                "feather": ("INT", {"default": 6, "min": 0, "max": 256, "step": 2,
                    "tooltip": "Gaussian blur radius on the paste mask, in SOURCE pixels. "
                               "Measured against the final frame, not the canvas, so the blend "
                               "is the same physical width whatever this frame's magnification "
                               "happens to be.\n\n"
                               "Canvas-relative feather is a trap: a 75px crop blown up to 512 "
                               "makes a 40px canvas feather only ~6 source px, while a 720px "
                               "crop makes it ~56. The blend ends up TIGHTEST exactly where the "
                               "face is smallest and the composite needs the most help - which "
                               "reads as a hard edge appearing as a shot zooms out."}),
                "colour_match": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Match the refined crop's per-channel mean/std to the region it "
                               "replaces. The crop and the full frame went through independent "
                               "passes, so without this the face can come back subtly brighter "
                               "or differently tinted and read as pasted on."}),
                "blend": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Global opacity of the refined face. Below 1.0 mixes back toward "
                               "the original - useful to dial back over-sharpening."}),
                "undetected_frames": (["fade_out", "skip", "composite_anyway"],
                    {"default": "fade_out",
                     "tooltip": "What to do on frames where no FACE was found (turned away / "
                                "occluded). ALL frames are still sent through H3 either way - "
                                "that is what keeps it temporally consistent - this only "
                                "controls whether the result is pasted back.\n"
                                "fade_out: ramp the composite to zero across the gap (smooth, "
                                "no pop, recommended).\n"
                                "skip: hard cut - those frames keep original pixels exactly.\n"
                                "composite_anyway: paste regardless. Risks H3 hallucinating a "
                                "face onto the back of a head."}),
            },
            "optional": {
                # OPTIONAL, not required - adding a required input breaks every existing
                # workflow and API caller with "Required input is missing".
                "feather_scales_with_crop": ("BOOLEAN", {"default": False,
                    "tooltip": "Old behaviour: treat feather as CANVAS pixels, so the blend "
                               "narrows as the crop shrinks. Leave off."}),
                "masks": ("MASK", {
                    "tooltip": "Optional per-frame paste masks in CANVAS space, e.g. from "
                               "H3 Face Mask (SAM). Overrides paste_region. This is the "
                               "FaceDetailer bbox+SAM path: the mask follows the actual face so "
                               "the blend falls on the jaw and hairline instead of an arbitrary "
                               "rectangle. With a SAM mask use a SMALL feather (4-8); a "
                               "rectangle needs much more."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = "MiniMax H3/Face Refine"
    DESCRIPTION = "Composite H3-refined face crops back into the source frames."

    def run(self, base_images, refined_crops, transform, paste_region, mask_dilation, feather,
            colour_match, blend, undetected_frames="fade_out", masks=None,
            feather_scales_with_crop=False):
        boxes = transform["boxes"]
        if undetected_frames == "composite_anyway":
            weights = None
        elif undetected_frames == "skip":
            weights = [1.0 if d else 0.0 for d in transform.get("detected", [])] or None
        else:
            weights = transform.get("weights")
        B = min(len(boxes), base_images.shape[0], refined_crops.shape[0])
        if base_images.shape[0] != refined_crops.shape[0]:
            print(f"[H3FaceRefine] frame count mismatch: base={base_images.shape[0]} "
                  f"refined={refined_crops.shape[0]} transform={len(boxes)} -> using {B}")

        import torch.nn.functional as F

        cw, ch = transform["canvas"]
        W, H = transform["src_size"]
        face_rects = transform.get("face_rect")

        # ---- GPU, batched. The previous version was a per-frame Python loop on CPU
        # tensors: measured at ~1 core of 24 busy with the GPU idle, ~8 minutes for 362
        # frames. Every operation here (affine warp, blur, blend) is a tensor op, so it
        # belongs on the GPU and can be batched over frames.
        import comfy.model_management as mm

        try:
            dev = mm.get_torch_device()
        except Exception:
            dev = base_images.device
        dt = base_images.dtype
        out = base_images[..., :3].clone()

        # chunked so the warped batch does not blow VRAM: N x H x W x 3 at once
        per_frame_mb = (H * W * 3 * 4) / 2 ** 20
        chunk = max(1, min(32, int(1024 / max(per_frame_mb, 1e-6))))

        for c0 in range(0, B, chunk):
            mm.throw_exception_if_processing_interrupted()
            c1 = min(c0 + chunk, B)
            n = c1 - c0

            # feather is given in SOURCE pixels; the mask is built in canvas space, so
            # convert using this chunk's magnification (canvas / crop height). Without
            # this the blend is ~10x tighter on distant frames than on close ones.
            if feather_scales_with_crop:
                f_can = int(feather)
            else:
                bh_mid = float(boxes[(c0 + c1 - 1) // 2][3])
                f_can = int(round(feather * (ch / max(bh_mid, 1.0))))
                f_can = max(1, min(f_can, ch // 3))

            # --- batched paste-mask in canvas space ---
            if masks is not None:
                mk = masks[c0:c1].to(dev).float()
                if mk.shape[-2:] != (ch, cw):
                    mk = F.interpolate(mk.unsqueeze(1), size=(ch, cw),
                                       mode="bilinear", align_corners=False)
                else:
                    mk = mk.unsqueeze(1)
                if mask_dilation > 0:
                    k = 2 * int(mask_dilation) + 1
                    mk = F.max_pool2d(mk, k, stride=1, padding=k // 2)
                mask_can = _gaussian_blur_mask(mk, f_can).clamp(0, 1)
            elif paste_region == "full_crop":
                one = _feather_mask(ch, cw, f_can, dev, torch.float32)
                mask_can = one.view(1, 1, ch, cw).expand(n, 1, ch, cw)
            else:
                mask_can = torch.cat([
                    _face_region_mask(
                        ch, cw,
                        face_rects[i] if face_rects and i < len(face_rects)
                        else (cw * 0.25, ch * 0.25, cw * 0.5, ch * 0.5),
                        int(mask_dilation), f_can,
                        "ellipse" if paste_region == "face_ellipse" else "rect",
                        dev, torch.float32)
                    for i in range(c0, c1)], dim=0)

            # --- one affine grid for the whole chunk ---
            th = torch.empty((n, 2, 3), dtype=torch.float32, device=dev)
            for j, i in enumerate(range(c0, c1)):
                x, y, bw, bh = (float(v) for v in boxes[i])
                th[j, 0, 0] = W / bw; th[j, 0, 1] = 0.0
                th[j, 0, 2] = (W - 2.0 * x) / bw - 1.0
                th[j, 1, 0] = 0.0;    th[j, 1, 1] = H / bh
                th[j, 1, 2] = (H - 2.0 * y) / bh - 1.0
            grid = F.affine_grid(th, (n, 3, int(H), int(W)), align_corners=False)

            patch_can = refined_crops[c0:c1, ..., :3].to(dev).movedim(-1, 1).float()
            patch = F.grid_sample(patch_can, grid, mode="bilinear",
                                  padding_mode="zeros", align_corners=False)
            m = F.grid_sample(mask_can.to(dev), grid, mode="bilinear",
                              padding_mode="zeros", align_corners=False).clamp(0, 1)

            patch = patch.movedim(1, -1)                 # [n,H,W,3]
            m = m.movedim(1, -1)                         # [n,H,W,1]
            base = out[c0:c1].to(dev).float()

            # --- weighted colour match, no boolean gather/scatter ---
            if colour_match > 0.0:
                wsum = m.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)
                bmu = (base * m).sum(dim=(1, 2), keepdim=True) / wsum
                pmu = (patch * m).sum(dim=(1, 2), keepdim=True) / wsum
                bsd = (((base - bmu) ** 2 * m).sum(dim=(1, 2), keepdim=True)
                       / wsum).sqrt().clamp_min(1e-6)
                psd = (((patch - pmu) ** 2 * m).sum(dim=(1, 2), keepdim=True)
                       / wsum).sqrt().clamp_min(1e-6)
                adj = (patch - pmu) * (bsd / psd) + bmu
                patch = patch + (adj - patch) * float(colour_match)
                patch = patch.clamp(0, 1)

            # --- per-frame opacity (blend x detection weight) ---
            wv = torch.full((n, 1, 1, 1), float(blend), device=dev, dtype=torch.float32)
            if weights is not None:
                for j, i in enumerate(range(c0, c1)):
                    if i < len(weights):
                        wv[j] *= float(weights[i])
            mm_ = m * wv

            out[c0:c1] = ((1.0 - mm_) * base + mm_ * patch).to(out.device, dt)

        return (out,)

# ----------------------------------------------------------------------------
# 3. inject real video into the AV latent
# ----------------------------------------------------------------------------


class H3InjectVideoLatent:
    """Replace the VIDEO stream of an H3 AV latent with real encoded frames (img2img seed).

    H3's own nodes always build a zeros latent - references are conditioning that is
    re-injected each step, never a starting point - so there is no stock video-to-video
    path. This encodes real frames into the video stream while leaving the audio stream
    intact, which turns SamplerCustomAdvanced + truncated sigmas into ordinary img2img.

    Pair with MiniMaxH3NativeAudioLock for the audio stream, and set strength with
    BasicScheduler's `denoise` - NOT with SplitSigmas. H3's flow-matching shift (12 by
    default) puts even the last split point of a short schedule at an effective sigma
    around 0.8, which rewrites the frame. `denoise` instead builds a longer full-range
    schedule and keeps only its lowest sigmas, so steps and strength stay independent.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "av_latent": ("LATENT",),
                "images": ("IMAGE",),
                "vae": ("VAE",),
            },
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("av_latent", "report")
    FUNCTION = "run"
    CATEGORY = "MiniMax H3/Face Refine"
    DESCRIPTION = "Encode real frames into the video stream of an H3 joint AV latent."

    def run(self, av_latent, images, vae):
        samples = av_latent.get("samples")
        if samples is None:
            raise KeyError('LATENT is missing "samples".')
        is_nested = isinstance(samples, comfy.nested_tensor.NestedTensor) or getattr(
            samples, "is_nested", False
        )
        if not is_nested:
            raise ValueError(
                "Expected a MiniMax H3 joint AV latent (NestedTensor). Feed the LATENT output "
                "of MiniMaxH3ReferenceToVideo / EmptyMiniMaxH3LatentAV."
            )

        members = list(samples.unbind())
        video_tmpl = members[0]

        encoded = vae.encode(images[..., :3])
        if encoded.ndim == 4:  # [B,C,H,W] -> [1,C,T,H,W]
            encoded = encoded.unsqueeze(0).movedim(1, 2)

        tgt_t, tgt_h, tgt_w = video_tmpl.shape[-3], video_tmpl.shape[-2], video_tmpl.shape[-1]
        got_t, got_h, got_w = encoded.shape[-3], encoded.shape[-2], encoded.shape[-1]
        if (got_h, got_w) != (tgt_h, tgt_w):
            raise ValueError(
                f"Spatial latent mismatch: encoded {got_h}x{got_w} but the AV latent expects "
                f"{tgt_h}x{tgt_w}. The crop canvas and the H3 node's width/height must match "
                f"(both are pixels/16)."
            )
        note = ""
        if got_t != tgt_t:
            # H3 packs 17 pixel frames -> 5 latent frames; a frame count off the 17k+5
            # grid lands here. Trim or pad rather than fail, but say so loudly.
            if got_t > tgt_t:
                encoded = encoded[..., :tgt_t, :, :]
            else:
                pad = video_tmpl[..., : tgt_t - got_t, :, :].to(encoded.device, encoded.dtype)
                encoded = torch.cat([encoded, pad], dim=-3)
            note = (f"  WARNING temporal mismatch: encoded t={got_t} vs latent t={tgt_t} "
                    f"-> {'trimmed' if got_t > tgt_t else 'padded'}. Frame count is probably "
                    f"off H3's 17k+5 grid.\n")

        members[0] = encoded.to(video_tmpl.device, video_tmpl.dtype)
        out = dict(av_latent)
        out["samples"] = comfy.nested_tensor.NestedTensor(tuple(members))

        report = (
            f"injected video latent {tuple(encoded.shape)} into AV latent "
            f"(streams={len(members)})\n{note}"
            f"frames_in={images.shape[0]}  {images.shape[2]}x{images.shape[1]}px"
        )
        return (out, report)


# ----------------------------------------------------------------------------


class H3PerFrameDenoise:
    """Scale denoise strength per frame, inversely to how big the face is.

    The sampler builds ONE sigma schedule for a whole clip, so every frame normally gets
    the same denoise. That is a poor fit for a shot where the subject walks from distant to
    close: the tiny-face frames have no detail to preserve and want a strong pass so H3
    SYNTHESISES a face, while the large-face frames have real detail and want a gentle one
    so it is not rewritten. One value cannot serve both.

    ComfyUI's noise_mask scales denoising per latent position, so varying it along the
    temporal axis gives per-frame strength out of a single sampling pass. Place this AFTER
    MiniMaxH3NativeAudioLock - it preserves that node's audio-side zeros, which are what
    keep the audio clean and drive lipsync.

    Granularity is one latent frame, i.e. 17 pixel frames per 5 latents (~3.4 frames).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "av_latent": ("LATENT",),
                "transform": ("H3FACEXFORM",),
                "strength_small_face": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Denoise multiplier where the face is SMALLEST. 1.0 = the full "
                               "denoise set on BasicScheduler."}),
                "strength_large_face": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Denoise multiplier where the face is LARGEST. Lower preserves "
                               "the detail those frames already have."}),
                "scale_mode": (["absolute_px", "relative_to_clip"],
                    {"default": "absolute_px",
                     "tooltip": "absolute_px: strength is set by real face size in SOURCE "
                                "pixels via face_px_small/large. Safe across a batch - a clip "
                                "that never has a small face gets the baseline throughout. "
                                "relative_to_clip: normalise to this clip's own min/max, so "
                                "its smallest face always gets the full boost regardless of "
                                "actual size. Use when tuning a single clip to its extremes."}),
                "face_px_small": ("FLOAT", {"default": 30.0, "min": 4.0, "max": 400.0,
                    "step": 1.0,
                    "tooltip": "Face height (SOURCE px) at or below which the full "
                               "strength_small_face is applied. Genuinely tiny faces only."}),
                "face_px_large": ("FLOAT", {"default": 120.0, "min": 8.0, "max": 800.0,
                    "step": 1.0,
                    "tooltip": "Face height (SOURCE px) at or above which strength_large_face "
                               "is applied. Calibrated so a clip whose smallest face is ~90px "
                               "gets only a mild boost - that size was already fine at the "
                               "baseline denoise - and anything past 120px gets none."}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.2, "max": 4.0, "step": 0.1,
                    "tooltip": "Curve on the interpolation. >1 keeps strength high until the "
                               "face is genuinely large; <1 drops it off early."}),
                "smooth_frames": ("INT", {"default": 9, "min": 1, "max": 61, "step": 2,
                    "tooltip": "Smooth the strength curve over time. An abrupt change in "
                               "denoise between neighbouring frames is visible as a texture "
                               "pop, so this wants to be generous."}),
            },
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("av_latent", "report")
    FUNCTION = "run"
    CATEGORY = "MiniMax H3/Face Refine"
    DESCRIPTION = "Per-frame denoise strength, scaled inversely to face size."

    def run(self, av_latent, transform, strength_small_face, strength_large_face,
            face_px_small, face_px_large, gamma, smooth_frames,
            scale_mode="absolute_px"):
        import torch.nn.functional as F

        samples = av_latent.get("samples")
        if samples is None or not (
                isinstance(samples, comfy.nested_tensor.NestedTensor)
                or getattr(samples, "is_nested", False)):
            raise ValueError("Expected a MiniMax H3 joint AV latent (NestedTensor).")

        members = list(samples.unbind())
        video = members[0]
        latent_t = video.shape[-3]

        boxes = transform["boxes"]
        cf = float(transform.get("crop_factor", 3.0)) or 3.0
        # source face height per frame = crop height / crop_factor
        face = np.array([b[3] / cf for b in boxes], dtype=np.float64)
        if face.size == 0:
            raise ValueError("transform has no boxes")

        if scale_mode == "relative_to_clip":
            # Normalise to THIS clip's own range: its smallest face always gets the full
            # boost, whatever size that actually is. Useful when you want a clip worked to
            # its own extremes, but across a batch it over-treats clips whose "smallest"
            # face is already large.
            lo, hi = float(face.min()), float(face.max())
        else:
            # Absolute pixel thresholds. A clip that never has a genuinely small face sits
            # at the baseline throughout, which is what makes one setting safe batch-wide.
            lo, hi = float(face_px_small), float(face_px_large)
        if hi - lo < 1e-6:
            t = np.zeros_like(face)
        else:
            t = np.clip((face - lo) / (hi - lo), 0.0, 1.0)
        t = np.clip(t, 0.0, 1.0) ** float(gamma)
        strength = strength_small_face + (strength_large_face - strength_small_face) * t
        strength = _smooth(strength, int(smooth_frames), "gaussian")
        strength = np.clip(strength, 0.0, 1.0)

        # per pixel-frame -> per latent-frame
        s = torch.from_numpy(strength).float().view(1, 1, -1)
        s = F.interpolate(s, size=int(latent_t), mode="linear", align_corners=True)
        s = s.view(1, 1, int(latent_t), 1, 1).to(video.device, torch.float32)

        vmask = s.expand(video.shape[0], 1, latent_t, video.shape[-2], video.shape[-1])
        vmask = vmask.expand(-1, video.shape[1], -1, -1, -1).contiguous()

        prev = av_latent.get("noise_mask")
        if prev is not None and (isinstance(prev, comfy.nested_tensor.NestedTensor)
                                 or getattr(prev, "is_nested", False)):
            # keep the audio side exactly as NativeAudioLock left it
            pm = list(prev.unbind())
            pm[0] = vmask.to(pm[0].dtype)
            new_mask = comfy.nested_tensor.NestedTensor(tuple(pm))
        else:
            audio_zero = torch.zeros_like(members[1]) if len(members) > 1 else None
            new_mask = comfy.nested_tensor.NestedTensor(
                (vmask.to(video.dtype),) + ((audio_zero,) if audio_zero is not None else ()))

        out = dict(av_latent)
        out["noise_mask"] = new_mask
        report = (
            f"per-frame denoise: face {face.min():.0f}-{face.max():.0f}px, ramp "
            f"{lo:.0f}-{hi:.0f}px ({scale_mode})  ->  strength "
            f"{strength.max():.2f} (smallest) .. {strength.min():.2f} (largest)\n"
            f"mean {strength.mean():.2f} over {len(strength)} frames, "
            f"{latent_t} latent steps, gamma={gamma}"
        )
        print("[H3FaceRefine] " + report)
        return (out, report)


class H3FaceMaskSAM:
    """True face-shaped paste masks via SAM, computed on the stabilised crops.

    Impact Pack's best-quality path is bbox + SAM: the bbox seeds a point/box prompt and
    SAM returns a mask that follows the actual face, so the blend falls on the jaw and
    hairline rather than on an arbitrary rectangle.

    Runs on the INPUT crops, exactly as FaceDetailer computes its mask from the source
    image before enhancement and then pastes the enhanced patch through it.

    Video-specific addition: SAM is run per frame and the resulting mask stack is
    temporally smoothed. Per-frame segmentation wobbles by a few pixels, and an unsmoothed
    mask boundary flickers - which is exactly the artefact this whole pipeline exists to
    avoid. Smoothing is cheap and meaningful here precisely because the crops are
    face-stabilised, so the masks are already roughly aligned.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "crops": ("IMAGE", {
                    "tooltip": "Wire the INPUT crops here - the 'crops' output of H3 Face "
                               "Track + Crop - NOT the refined/decoded result.\n\n"
                               "This matches FaceDetailer: make_sam_mask() runs on the SOURCE "
                               "image and the resulting mask is what the enhanced patch is "
                               "later pasted through. Generation never feeds back into the "
                               "mask.\n\n"
                               "Masking the generated result instead is actively wrong: if the "
                               "model nudges the face inward, the mask traces the NEW, smaller "
                               "silhouette and the ORIGINAL face pokes out past it - most "
                               "visibly the nose on profile shots. Masking the input covers "
                               "where the face actually is in the footage being replaced.\n\n"
                               "It is also cheaper: no dependency on the sampler, so SAM need "
                               "not be resident alongside the video model."}),
                "sam_model": ("SAM_MODEL",),
                "transform": ("H3FACEXFORM",),
                "threshold": ("FLOAT", {"default": 0.93, "min": 0.0, "max": 1.0, "step": 0.01}),
                "dilation": ("INT", {"default": 0, "min": 0, "max": 128, "step": 2,
                    "tooltip": "Mirrors FaceDetailer's sam_dilation default of 0. "
                               "SAM masks are accurate, so they rarely need growing."}),
                "temporal_smooth": ("INT", {"default": 5, "min": 1, "max": 31, "step": 2,
                    "tooltip": "Frames of averaging across the mask stack. 1 disables it and "
                               "you will likely see the mask edge shimmer."}),
            },
        }

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("masks", "report")
    FUNCTION = "run"
    CATEGORY = "MiniMax H3/Face Refine"
    DESCRIPTION = "Per-frame SAM face masks on the stabilised crops, temporally smoothed."

    def run(self, crops, sam_model, transform, threshold, dilation, temporal_smooth):
        import torch.nn.functional as F

        sam_obj = sam_model if not hasattr(sam_model, "sam_wrapper") else sam_model.sam_wrapper
        face_rects = transform.get("face_rect") or []
        B, ch, cw, _ = crops.shape
        masks = torch.zeros((B, ch, cw), dtype=torch.float32)
        ok = 0

        import comfy.model_management as mm
        import comfy.utils as _cu

        # REQUIRED. SAMLoader's AUTO device_mode leaves the model on CPU and only moves it
        # to VRAM when prepare_device() is called - Impact's own make_sam_mask does this
        # before its work. Without it every predict() runs the ViT image encoder on CPU,
        # which is ~10-50x slower and was the cause of multi-minute mask passes.
        if hasattr(sam_obj, "prepare_device"):
            sam_obj.prepare_device()

        pbar = _cu.ProgressBar(B)
        try:
            for i in range(B):
                # SAM runs per frame and can take minutes on a long clip. Without these two
                # lines the node is one uninterruptible block: ComfyUI only honours cancel
                # BETWEEN nodes, so a wedged run needs a restart to clear.
                mm.throw_exception_if_processing_interrupted()
                pbar.update(1)
                if i % 25 == 0:
                    print(f"[H3FaceRefine] SAM mask {i}/{B}")
                fr = face_rects[i] if i < len(face_rects) else (cw*0.25, ch*0.25, cw*0.5, ch*0.5)
                fx, fy, fwd, fhd = fr
                bbox = [max(0, int(fx)), max(0, int(fy)),
                        min(cw, int(fx + fwd)), min(ch, int(fy + fhd))]
                pts = [(int(fx + fwd / 2), int(fy + fhd / 2))]
                img = (crops[i, ..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                try:
                    det = sam_obj.predict(img, pts, [1], bbox, threshold)
                except Exception:
                    det = None
                if det:
                    m = det[0] if not isinstance(det, torch.Tensor) else det
                    m = torch.as_tensor(np.asarray(m), dtype=torch.float32).squeeze()
                    if m.shape[-2:] == (ch, cw):
                        masks[i] = (m > 0.5).float()
                        ok += 1
        finally:
            if hasattr(sam_obj, "release_device"):
                try:
                    sam_obj.release_device()
                except Exception:
                    pass

        # frames SAM failed on fall back to the face rect so they are never left empty
        for i in range(B):
            if masks[i].max() <= 0:
                fx, fy, fwd, fhd = (face_rects[i] if i < len(face_rects)
                                    else (cw*0.25, ch*0.25, cw*0.5, ch*0.5))
                x0, y0 = max(0, int(fx)), max(0, int(fy))
                x1, y1 = min(cw, int(fx + fwd)), min(ch, int(fy + fhd))
                if x1 > x0 and y1 > y0:
                    masks[i, y0:y1, x0:x1] = 1.0

        if dilation > 0:
            k = 2 * int(dilation) + 1
            masks = F.max_pool2d(masks.unsqueeze(1), k, stride=1, padding=k // 2).squeeze(1)

        if temporal_smooth > 1 and B > 2:
            w = min(int(temporal_smooth) | 1, B if B % 2 else B - 1)
            if w >= 3:
                pad = w // 2
                # replicate-pad needs a 3D tensor when padding only the last dim, so go
                # straight to [pixels, 1, frames] rather than via a 4D intermediate
                t = masks.permute(1, 2, 0).reshape(-1, 1, B).contiguous()
                t = F.pad(t, (pad, pad), mode="replicate")
                kern = torch.ones(1, 1, w, dtype=t.dtype, device=t.device) / w
                sm = F.conv1d(t, kern)
                masks = sm.reshape(ch, cw, B).permute(2, 0, 1).contiguous()

        report = (f"SAM masks: {ok}/{B} frames segmented "
                  f"({B-ok} fell back to the face rect)\n"
                  f"dilation={dilation}  temporal_smooth={temporal_smooth}\n"
                  f"mean coverage {float(masks.mean())*100:.1f}% of canvas")
        print("[H3FaceRefine] " + report)
        return (masks, report)


class H3FaceTransformInfo:
    """Print the per-frame transform - sanity-check tracking before spending GPU time."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"transform": ("H3FACEXFORM",),
                             "max_rows": ("INT", {"default": 12, "min": 1, "max": 400})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "MiniMax H3/Face Refine"

    def run(self, transform, max_rows):
        boxes = transform["boxes"]
        cw, ch = transform["canvas"]
        lines = [f"frames={transform['frames']}  canvas={cw}x{ch}  src={transform['src_size']}",
                 f"{'frame':>6} {'x':>6} {'y':>6} {'w':>6} {'h':>6} {'mag':>6}"]
        step = max(1, len(boxes) // max_rows)
        for i in range(0, len(boxes), step):
            x, y, w, h = boxes[i]
            lines.append(f"{i:>6} {x:>6} {y:>6} {w:>6} {h:>6} {ch/h:>5.2f}x")
        txt = "\n".join(lines)
        print("[H3FaceRefine]\n" + txt)
        return (txt,)


NODE_CLASS_MAPPINGS = {
    "H3FaceTrackCrop": H3FaceTrackCrop,
    "H3FaceStitch": H3FaceStitch,
    "H3InjectVideoLatent": H3InjectVideoLatent,
    "H3PerFrameDenoise": H3PerFrameDenoise,
    "H3FaceMaskSAM": H3FaceMaskSAM,
    "H3FaceTransformInfo": H3FaceTransformInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3FaceTrackCrop": "H3 Face Track + Crop",
    "H3FaceStitch": "H3 Face Stitch Back",
    "H3InjectVideoLatent": "H3 Inject Video Latent (img2img)",
    "H3PerFrameDenoise": "H3 Per-Frame Denoise",
    "H3FaceMaskSAM": "H3 Face Mask (SAM)",
    "H3FaceTransformInfo": "H3 Face Transform Info",
}
