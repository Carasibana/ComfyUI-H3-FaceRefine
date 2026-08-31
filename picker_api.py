"""Backend route for the H3 Load Video + Face Select picker.

The picker runs before the graph does, so it cannot read anything the node
produced. It repeats the node's own load/detect/cut pass and returns one card per
shot: the frame the shot locks on, every face outlined and numbered.
"""

import asyncio
import base64
import io
import json
import os
import traceback

import numpy as np
import torch

from . import nodes as N


# Both routes below take a path from the request and read the file it names. `video`
# accepts any path on purpose - source footage rarely lives in ComfyUI's input folder -
# so by default that is what they honour, which is also what VideoHelperSuite does. Set
# H3_FACEREFINE_STRICT_PATHS to confine them to ComfyUI's own directory instead, for a
# server reachable by anything other than the person sitting at it.
_STRICT_ENV = "H3_FACEREFINE_STRICT_PATHS"


def _path_allowed(path: str) -> bool:
    if not os.environ.get(_STRICT_ENV):
        return True
    try:
        import folder_paths
        base = os.path.abspath(getattr(folder_paths, "base_path", ".") or ".")
    except Exception:
        base = os.path.abspath(".")
    try:
        return os.path.commonpath([base, os.path.abspath(path)]) == base
    except ValueError:
        return False        # different drive on Windows: commonpath raises
    except Exception:
        return False


def _strict_refusal(path: str):
    return {"error": (
        f"{_STRICT_ENV} is set, so this only reads files inside ComfyUI's own folder. "
        f"{os.path.basename(path)} is outside it. Upload the video with Browse, or unset "
        f"{_STRICT_ENV}.")}


def _shot_cards(frames, all_boxes, all_confs, segs, W, H, mode, long_edge=640):
    """Per shot: the frame it locks on, that frame with every face numbered, and the
    boxes normalised 0-1 so the browser can lay click targets over the image."""
    from PIL import Image

    thick = max(2, int(round(min(H, W) / 240.0)))
    scale = max(2, int(round(min(H, W) / 140.0)))
    _, lh = N._label_size("0", scale)

    out = []
    for a, b in segs:
        frame = next((i for i in range(a, b) if all_boxes[i]), -1)
        if frame < 0:
            out.append({"segment": [int(a), int(b)], "frame": -1, "faces": 0,
                        "jpg": None, "boxes": []})
            continue
        img = frames[frame][..., :3].clone()
        ranked = N._rank_boxes(all_boxes[frame], all_confs[frame], W, H, mode)
        norm = []
        for rank, bi in enumerate(ranked):
            box = all_boxes[frame][bi]
            col = (0.15, 0.45, 1.0)
            N._draw_rect(img, box[0], box[1], box[2], box[3], col, thick)
            N._draw_label(img, box[0], box[1] - lh, str(rank), scale, (1.0, 1.0, 1.0), col)
            norm.append([box[0] / W, box[1] / H, box[2] / W, box[3] / H])

        arr = (img.clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
        pil = Image.fromarray(arr)
        if max(pil.size) > long_edge:
            r = long_edge / float(max(pil.size))
            pil = pil.resize((max(1, int(pil.width * r)), max(1, int(pil.height * r))))
        buf = io.BytesIO()
        pil.convert("RGB").save(buf, format="JPEG", quality=88, optimize=True)
        out.append({"segment": [int(a), int(b)], "frame": int(frame),
                    "faces": len(ranked), "boxes": norm,
                    "jpg": base64.b64encode(buf.getvalue()).decode("ascii")})
    return out


# Last scan, keyed on everything that would change the answer. Scanning runs the
# detector over the whole clip, so clicking the button twice on an unchanged node
# should not pay for it twice.
_CACHE = {"key": None, "data": None}

# Only what changes the DETECTION. `select` changes which face is numbered 0, not which
# faces exist, so it re-renders cards from this and never re-scans.
_KEYED = ("detector", "confidence", "cut_detection", "cut_threshold",
          "skip_first_frames", "frame_load_cap", "select_every_nth")


def _cache_key(path, params):
    try:
        stat = os.stat(path)
        stamp = (int(stat.st_mtime), int(stat.st_size))
    except OSError:
        stamp = (0, 0)
    return json.dumps([path, stamp] + [str(params.get(k)) for k in _KEYED], sort_keys=True)


def _progress(stage, done=0, total=0, detail=""):
    """Push a scan update to the browser over ComfyUI's own websocket.

    The scan is one blocking request, so without this the dialog can only show a
    spinner and an elapsed count - it cannot say which stage is running or how far in.
    """
    try:
        from server import PromptServer
        PromptServer.instance.send_sync("h3_facerefine/progress", {
            "stage": stage, "done": int(done), "total": int(total), "detail": detail})
    except Exception:
        pass                      # progress is cosmetic; never let it break a scan


class _RenderBusy(RuntimeError):
    """Raised instead of scanning while the GPU is busy."""


def _render_running():
    """True while ComfyUI is executing a prompt.

    The scan runs the detector on the GPU from an HTTP thread, outside the queue, so it
    can start mid-render. Ultralytics abandons NMS when it runs over its time budget and
    returns whatever it had, which means a scan run against a busy GPU can quietly find
    fewer faces than are there - and a selection made from it would be wrong.
    """
    try:
        from server import PromptServer
        running, _pending = PromptServer.instance.prompt_queue.get_current_queue()
        return bool(running)
    except Exception:
        return False


def _scan(params):
    """Blocking. Runs on a worker thread so the server keeps serving."""
    path = N._resolve_video(params.get("video"))
    if not _path_allowed(path):
        raise ValueError(_strict_refusal(path)["error"])

    key = _cache_key(path, params)
    if _CACHE["key"] == key and _CACHE["data"] is not None:
        return _cards_from(_CACHE["data"], params, cached=True)

    _progress("Loading the video", 0, 0, os.path.basename(path))
    if _render_running() and not params.get("force"):
        raise _RenderBusy(
            "A render is running. Scanning now would share the GPU with it and could "
            "miss faces, so the result would not be safe to pick from. Try again once "
            "the queue is clear.")

    images, _audio, fps = N._load_video_components(path)
    images, _audio, fps = N._trim_batch(
        images, _audio, fps,
        int(params.get("skip_first_frames") or 0),
        int(params.get("frame_load_cap") or 0),
        int(params.get("select_every_nth") or 1))
    B, H, W, _ = images.shape

    model = N._load_detector(str(params.get("detector")))
    conf = float(params.get("confidence") or 0.35)

    cuts, cut_det, cut_tc = [], None, None
    if str(params.get("cut_detection", "none")) != "none":
        try:
            cut_det, cut_tc = N._make_cut_detector(float(params.get("cut_threshold") or 3.0))
        except Exception:
            cut_det = None

    det_name = os.path.basename(str(params.get("detector") or "")) or "the detector"
    work = (f"faces ({det_name})" if cut_det is None
            else f"faces ({det_name}) and cuts (PySceneDetect)")
    step = max(1, B // 50)          # ~50 updates, not one per frame

    all_boxes, all_confs = [], []
    for i in range(B):
        if i % step == 0:
            _progress(f"Scanning for {work}", i, B, f"frame {i} of {B}")
        bgr = N._to_bgr_u8(images[i])
        if cut_det is not None:
            for t in cut_det.process_frame(cut_tc(i, fps=24.0), bgr):
                cuts.append(int(t.get_frames()))
        r = model.predict(bgr, conf=conf, verbose=False)[0]
        if len(r.boxes):
            bx = [[float(v) for v in q] for q in r.boxes.xyxy.tolist()]
            cf = getattr(r.boxes, "conf", None)
            all_boxes.append(bx)
            all_confs.append([float(c) for c in cf.tolist()] if cf is not None
                             else [1.0] * len(bx))
        else:
            all_boxes.append([])
            all_confs.append([])

    if max((len(b) for b in all_boxes), default=0) == 0:
        raise ValueError("No face detected in any frame. Lower confidence, or use a "
                         "detector that suits this material.")

    _progress(f"Scanning for {work}", B, B, f"frame {B} of {B}")
    segs = N._segments(B, cuts)
    # Only the frame each shot locks on is kept. Holding the whole clip to re-render
    # from would cost gigabytes; these are a handful of frames and are all a card needs.
    keep = {}
    for a, b in segs:
        f = next((i for i in range(a, b) if all_boxes[i]), -1)
        if f >= 0:
            keep[f] = images[f].clone()
    det = {"frames": keep, "all_boxes": all_boxes, "all_confs": all_confs,
           "segs": segs, "B": int(B), "W": int(W), "H": int(H), "fps": float(fps),
           "path": path}
    _CACHE["key"], _CACHE["data"] = key, det
    return _cards_from(det, params, cached=False)


def _cards_from(det, params, cached):
    """Render the shot cards from a cached detection.

    Separate from the scan because `select` changes only which face is numbered 0, and
    re-detecting a whole clip to answer that would be a full pass for nothing.
    """
    mode = N._review_select(str(params.get("select") or "largest_face"))
    _progress("Building the shot previews", 0, 0, f"{len(det['segs'])} shot(s)")
    cards = _shot_cards(det["frames"], det["all_boxes"], det["all_confs"],
                        det["segs"], det["W"], det["H"], mode)
    return {"frames": det["B"], "width": det["W"], "height": det["H"],
            "fps": det["fps"], "source": os.path.basename(det["path"]),
            "path": det["path"], "shots": cards, "cached": bool(cached)}


def _preview_xy(params):
    """One frame, with the closest_to_xy guides on it. Blocking; worker thread.

    Seeks to the single frame rather than decoding the clip, because this answers "where
    does my point land" while the number is being typed - a full pass per keystroke would
    make it useless. frame_index counts LOADED frames, so trimming is undone here to reach
    the right frame of the file.
    """
    import cv2
    from PIL import Image

    if _render_running() and not params.get("force"):
        raise _RenderBusy(
            "A render is running. Detecting now would share the GPU with it and could "
            "miss faces, so the face this shows as nearest may not be the one the graph "
            "picks. Try again once the queue is clear.")

    path = N._resolve_video(params.get("video"))
    if not _path_allowed(path):
        raise ValueError(_strict_refusal(path)["error"])
    skip = int(params.get("skip_first_frames") or 0)
    nth = max(1, int(params.get("select_every_nth") or 1))
    cap_n = int(params.get("frame_load_cap") or 0)
    want = max(0, int(params.get("frame_index") or 0))
    if cap_n:
        want = min(want, cap_n - 1)
    src = skip + want * nth

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open {os.path.basename(path)}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total and src >= total:
        src = total - 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, src)
    ok, bgr = cap.read()
    cap.release()
    if not ok or bgr is None:
        raise ValueError(f"No frame {want} in this video"
                         + (f" (it has {total})" if total else ""))

    img = torch.from_numpy(bgr[..., ::-1].copy()).float() / 255.0      # BGR -> RGB [H,W,3]
    H, W, _ = img.shape
    model = N._load_detector(str(params.get("detector")))
    r = model.predict(bgr, conf=float(params.get("confidence") or 0.35), verbose=False)[0]
    boxes = [[float(v) for v in q] for q in r.boxes.xyxy.tolist()] if len(r.boxes) else []
    cf = getattr(r.boxes, "conf", None)
    confs = ([float(c) for c in cf.tolist()] if cf is not None and len(r.boxes)
             else [1.0] * len(boxes))

    # The ranking substitutes the frame centre for a missing point, so the marker has
    # to as well or it would be drawn somewhere the ranking never measured.
    tx = W / 2.0 if params.get("X") is None else float(params.get("X"))
    ty = H / 2.0 if params.get("Y") is None else float(params.get("Y"))
    thick = max(2, int(round(min(H, W) / 240.0)))
    scale = max(2, int(round(min(H, W) / 140.0)))
    _, lh = N._label_size("0", scale)
    ranked = N._rank_boxes(boxes, confs, W, H, "closest_to_xy", tx, ty) if boxes else []
    N._draw_crosshair(img, tx, ty, (1.0, 0.75, 0.0),
                      max(3, int(round(min(H, W) / 200.0))))
    for rank, bi in enumerate(ranked):
        b = boxes[bi]
        hit = rank == 0
        col = (0.0, 1.0, 0.2) if hit else (0.15, 0.45, 1.0)
        N._draw_rect(img, b[0], b[1], b[2], b[3], col, thick * (2 if hit else 1))
        N._draw_label(img, b[0], b[1] - lh, str(rank), scale,
                      (0.0, 0.0, 0.0) if hit else (1.0, 1.0, 1.0), col)

    arr = (img.clamp(0, 1).numpy() * 255.0).astype(np.uint8)
    pil = Image.fromarray(arr)
    if max(pil.size) > 640:
        rr = 640.0 / float(max(pil.size))
        pil = pil.resize((max(1, int(pil.width * rr)), max(1, int(pil.height * rr))))
    buf = io.BytesIO()
    pil.convert("RGB").save(buf, format="JPEG", quality=88, optimize=True)
    return {"jpg": base64.b64encode(buf.getvalue()).decode("ascii"),
            "faces": len(boxes), "frame": int(want), "source_frame": int(src),
            "width": int(W), "height": int(H),
            "chose": (int(ranked[0]) if ranked else -1)}


def register():
    """Register the route. Safe to call when ComfyUI's server is absent."""
    try:
        from aiohttp import web
        from server import PromptServer
    except Exception:
        return

    routes = getattr(getattr(PromptServer, "instance", None), "routes", None)
    if routes is None:
        return

    @routes.get("/h3_facerefine/videos")
    async def _videos_route(_request):
        """Video files in ComfyUI's input folder, as full paths, for the Browse list."""
        try:
            import folder_paths
            root = folder_paths.get_input_directory()
            names = N._input_video_list()
            if names == [N._NO_VIDEO]:
                names = []
            return web.json_response({"root": root, "videos": [
                {"name": n, "path": os.path.join(root, n)} for n in names]})
        except Exception as exc:
            return web.json_response({"error": str(exc), "videos": []})

    @routes.get("/h3_facerefine/preview")
    async def _preview_route(request):
        """Serve a video for the node's preview player.

        ComfyUI's own /view only reaches its input folder, and `video` accepts any
        path, so a pasted path would have nothing to preview. FileResponse handles
        range requests, so the player can seek.
        """
        raw = request.query.get("path", "")
        try:
            path = N._resolve_video(raw)
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=404)
        if os.path.splitext(path)[1].lower() not in N._VIDEO_EXTS:
            return web.json_response({"error": "not a video file"}, status=400)
        if not _path_allowed(path):
            return web.json_response(_strict_refusal(path), status=403)
        return web.FileResponse(path)

    @routes.post("/h3_facerefine/preview_xy")
    async def _preview_xy_route(request):
        try:
            params = await request.json()
        except Exception:
            return web.json_response({"error": "bad request body"}, status=400)
        loop = asyncio.get_event_loop()
        try:
            data = await loop.run_in_executor(None, _preview_xy, params)
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=200)
        return web.json_response(data)

    @routes.post("/h3_facerefine/scan")
    async def _scan_route(request):
        try:
            params = await request.json()
        except Exception:
            return web.json_response({"error": "bad request body"}, status=400)
        loop = asyncio.get_event_loop()
        try:
            data = await loop.run_in_executor(None, _scan, params)
        except _RenderBusy as exc:
            return web.json_response({"error": str(exc), "busy": True}, status=200)
        except Exception as exc:
            traceback.print_exc()
            return web.json_response({"error": str(exc)}, status=200)
        return web.json_response(data)
