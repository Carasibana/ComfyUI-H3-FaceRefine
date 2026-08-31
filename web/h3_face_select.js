import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE = "H3FaceSelect";
const TRACKER = "H3FaceTrackCrop";
const THUMB = 64;
// confirmed_pick entry for "the subject is not in this shot at all"
const ABSENT = -1;

// ComfyUI also runs in a desktop shell, where the browser's own dialogs are not the
// host's and window.alert blocks everything behind it. Core never calls it: its upload
// widget reports a failed POST with toast.addAlert(status + " - " + statusText), so this
// does the same, and falls back to the console rather than to a browser dialog.
function notify(msg) {
    try {
        const em = app.extensionManager;
        if (em && em.toast && em.toast.addAlert) {
            em.toast.addAlert(String(msg));
            return;
        }
    } catch (e) { /* fall through to the console */ }
    console.error("[H3 FaceRefine] " + msg);
}

const w = (node, name) => (node.widgets || []).find((x) => x.name === name);

// Values that were relabelled while unreleased. A widget holding a value its combo
// no longer offers fails validation outright, or is reset to the first entry - which
// silently changes what a saved workflow does.
const RETIRED = {
    select: {
        confidence: "detector_score",
        most_central: "centre_most",
        identity: "identity_reference",
        // 1.0.0 offered only largest and most_central, with no select_order to give
        // a direction, so migrateWidgetValues never fires on those workflows. Both
        // can only have meant the descending sense. Anything that DID carry an order
        // was already rewritten by DIRECTED above, so it never reaches this table.
        largest: "largest_face",
        area: "largest_face",
    },
    cut_detection: { auto: "auto (pyscenedetect)" },
};

// The positional modes carried their direction in the retired select_order widget, so
// one old pair collapses onto one new name. Both have to be read to write one value.
const DIRECTED = {
    largest:  { descending: "largest_face", ascending: "smallest_face" },
    area:     { descending: "largest_face", ascending: "smallest_face" },
    x1:       { descending: "right_most",   ascending: "left_most" },
    x2:       { descending: "right_most",   ascending: "left_most" },
    center_x: { descending: "right_most",   ascending: "left_most" },
    y1:       { descending: "bottom_most",  ascending: "top_most" },
    y2:       { descending: "bottom_most",  ascending: "top_most" },
    center_y: { descending: "bottom_most",  ascending: "top_most" },
};
const ORDERS = ["descending", "ascending"];

// Runs BEFORE litegraph applies widgets_values, because select_order sat in the middle
// of the list: dropping it shifts every value after it onto the wrong widget. Rewriting
// the saved array first is what keeps an old workflow meaning what it meant.
function migrateWidgetValues(info) {
    const vals = info && info.widgets_values;
    if (!Array.isArray(vals)) return;
    const oi = vals.findIndex((v) => ORDERS.includes(v));
    if (oi < 0) return;                       // already migrated, or never had one
    // Search BACKWARDS from the order value for a mode that needs rewriting. Scanning
    // forwards for any old select value matched canvas_mode instead, whose default is
    // also "manual" and which sits earlier in the tracker's list. Only the directional
    // modes need the order to resolve, so only those are looked for.
    for (let i = oi - 1; i >= 0; i--) {
        const dir = DIRECTED[String(vals[i])];
        if (dir) { vals[i] = dir[String(vals[oi])] || dir.descending; break; }
    }
    vals.splice(oi, 1);
    // Old files carry trailing nulls and blanks from widgets that should never have
    // been serialised. They used to fall off the end harmlessly; now that the node has
    // three MORE widgets than it did, they would land on X, Y and frame_index instead
    // of their defaults - and an empty string there is not a number.
    while (vals.length &&
           (vals[vals.length - 1] === null ||
            vals[vals.length - 1] === undefined ||
            vals[vals.length - 1] === "")) {
        vals.pop();
    }
}

function migrateValues(node) {
    for (const [name, map] of Object.entries(RETIRED)) {
        const wd = w(node, name);
        if (wd && Object.prototype.hasOwnProperty.call(map, wd.value)) {
            wd.value = map[wd.value];
        }
    }
}

function styleOnce() {
    if (document.getElementById("h3fr-style")) return;
    const s = document.createElement("style");
    s.id = "h3fr-style";
    s.textContent = [
        ".h3fr-back{position:fixed;inset:0;background:rgba(0,0,0,.72);z-index:10000;",
        "  display:flex;align-items:center;justify-content:center;font-family:sans-serif}",
        ".h3fr-box{background:#222;color:#ddd;border:1px solid #444;border-radius:8px;",
        "  max-width:min(1100px,94vw);max-height:92vh;display:flex;flex-direction:column}",
        ".h3fr-head{padding:12px 16px;border-bottom:1px solid #444;font-size:14px}",
        ".h3fr-body{padding:12px 16px;overflow:auto;flex:1}",
        ".h3fr-foot{padding:12px 16px;border-top:1px solid #444;display:flex;gap:8px;",
        "  justify-content:flex-end;align-items:center}",
        ".h3fr-msg{flex:1;font-size:12px;color:#999}",
        ".h3fr-shot{margin-bottom:18px}",
        ".h3fr-shot h4{margin:0 0 6px;font-size:13px;font-weight:600;color:#bbb}",
        ".h3fr-imgwrap{position:relative;display:inline-block;line-height:0}",
        ".h3fr-imgwrap img{max-width:100%;border-radius:4px;display:block}",
        ".h3fr-hit{position:absolute;border:2px solid transparent;border-radius:3px;cursor:pointer}",
        ".h3fr-hit:hover{border-color:#7ab7ff}",
        ".h3fr-hit.sel{border-color:#2ecc40;background:rgba(46,204,64,.18)}",
        ".h3fr-nums{margin-top:6px;display:flex;flex-wrap:wrap;gap:6px}",
        ".h3fr-nums button{background:#333;color:#ddd;border:1px solid #555;border-radius:4px;",
        "  padding:3px 10px;cursor:pointer;font-size:12px}",
        ".h3fr-nums button.sel{background:#2ecc40;border-color:#2ecc40;color:#111;font-weight:700}",
        ".h3fr-nums button.none{margin-left:10px;font-style:italic}",
        ".h3fr-nums button.none.sel{background:#b8862b;border-color:#b8862b;color:#111}",
        ".h3fr-same-cell.absent canvas{border-color:#b8862b}",
        ".h3fr-xyhost{overflow:hidden;border-radius:4px;position:relative}",
        ".h3fr-xymsg{padding:8px 10px;color:#ddd;font:12px sans-serif;",
        "  background:#1e1e1e;border:1px solid #444;border-radius:4px;",
        "  box-sizing:border-box;height:100%;overflow:hidden}",
        ".h3fr-xyhost img{width:100%;height:auto;display:block;border-radius:4px}",
        ".h3fr-xytag{position:absolute;left:0;right:0;bottom:0;padding:3px 6px;",
        "  background:rgba(0,0,0,.62);color:#eee;font:11px sans-serif}",
        ".h3fr-foot button{background:#333;color:#ddd;border:1px solid #555;border-radius:4px;",
        "  padding:6px 14px;cursor:pointer}",
        ".h3fr-foot button.go{background:#2d6cdf;border-color:#2d6cdf;color:#fff}",
        ".h3fr-list{display:flex;flex-direction:column;gap:2px;min-width:420px}",
        ".h3fr-list button{text-align:left;background:#2a2a2a;color:#ddd;border:1px solid #3a3a3a;",
        "  border-radius:3px;padding:6px 10px;cursor:pointer;font-size:12px;font-family:monospace}",
        ".h3fr-list button:hover{background:#333;border-color:#2d6cdf}",
        ".h3fr-strip{display:flex;flex-wrap:wrap;gap:6px;padding:4px 2px;",
        "  align-items:flex-start;justify-content:center;cursor:pointer}",
        ".h3fr-thumb{position:relative;width:64px;height:64px;border-radius:4px;",
        "  overflow:hidden;border:1px solid #555;background:#2a2a2a;flex:0 0 auto}",
        ".h3fr-thumb:hover{border-color:#7ab7ff}",
        ".h3fr-thumb img{width:100%;height:100%;object-fit:cover;display:block}",
        ".h3fr-thumb.set{border-color:#2ecc40}",
        ".h3fr-tag{position:absolute;left:0;bottom:0;background:rgba(0,0,0,.7);color:#eee;",
        "  font:10px/1.4 sans-serif;padding:0 4px;border-top-right-radius:3px}",
        ".h3fr-vid{width:100%;height:100%;background:#111;border:1px solid #444;",
        "  border-radius:4px;display:block;object-fit:contain;box-sizing:border-box}",
        ".h3fr-vidhost{overflow:hidden;box-sizing:border-box}",
        ".h3fr-striphost{overflow:hidden;box-sizing:border-box}",
        ".h3fr-prog{height:6px;background:#333;border-radius:3px;overflow:hidden;",
        "  margin-top:12px}",
        ".h3fr-prog i{display:block;height:100%;width:32%;background:#2d6cdf;",
        "  border-radius:3px;animation:h3frslide 1.15s ease-in-out infinite}",
        ".h3fr-prog.h3fr-known i{animation:none;transition:width .18s linear}",
        "@keyframes h3frslide{0%{transform:translateX(-110%)}",
        "  100%{transform:translateX(360%)}}",
        ".h3fr-same{position:sticky;top:0;z-index:2;background:#262626;",
        "  border:1px solid #3a3a3a;border-radius:6px;padding:8px 10px;margin-bottom:14px}",
        ".h3fr-same p{margin:0 0 8px;font:12px sans-serif;color:#bbb}",
        ".h3fr-same b{color:#eee}",
        ".h3fr-same-row{display:flex;gap:8px;flex-wrap:wrap}",
        ".h3fr-same-cell{text-align:center;font:10px sans-serif;color:#999}",
        ".h3fr-same-cell canvas{width:56px;height:56px;border-radius:4px;display:block;",
        "  border:1px solid #555;background:#2a2a2a}",
        ".h3fr-same-cell.set canvas{border-color:#2ecc40}",
        ".h3fr-vid-none{width:100%;min-height:70px;background:#1e1e1e;border:1px dashed #555;",
        "  border-radius:4px;color:#888;font:12px sans-serif;display:flex;",
        "  align-items:center;justify-content:center;text-align:center;padding:8px;",
        "  box-sizing:border-box;cursor:pointer}",
    ].join("\n");
    document.head.appendChild(s);
}

function dialog(title) {
    styleOnce();
    const back = document.createElement("div");
    back.className = "h3fr-back";
    const box = document.createElement("div");
    box.className = "h3fr-box";
    const head = document.createElement("div");
    head.className = "h3fr-head";
    head.textContent = title;
    const body = document.createElement("div");
    body.className = "h3fr-body";
    const foot = document.createElement("div");
    foot.className = "h3fr-foot";
    const msg = document.createElement("span");
    msg.className = "h3fr-msg";
    foot.appendChild(msg);
    box.appendChild(head);
    box.appendChild(body);
    box.appendChild(foot);
    back.appendChild(box);
    document.body.appendChild(back);
    const close = () => back.remove();
    back.addEventListener("click", (e) => { if (e.target === back) close(); });
    return { body, foot, msg, head, close };
}

function button(label, cls, fn) {
    const b = document.createElement("button");
    b.textContent = label;
    if (cls) b.className = cls;
    b.onclick = fn;
    return b;
}

/* --------------------------------------------------------------- thumbnails */

// Drawn rather than shipped as an asset: a blank face outline with a question mark,
// so an unpicked shot reads as "nothing chosen" rather than as an empty box.
function placeholderThumb() {
    const c = document.createElement("canvas");
    c.width = THUMB;
    c.height = THUMB;
    const x = c.getContext("2d");
    x.fillStyle = "#2a2a2a";
    x.fillRect(0, 0, THUMB, THUMB);
    x.strokeStyle = "#6a6a6a";
    x.lineWidth = 2;
    x.beginPath();
    x.ellipse(THUMB / 2, THUMB * 0.44, THUMB * 0.26, THUMB * 0.32, 0, 0, Math.PI * 2);
    x.stroke();
    x.beginPath();
    x.arc(THUMB / 2, THUMB * 1.02, THUMB * 0.36, Math.PI * 1.18, Math.PI * 1.82);
    x.stroke();
    x.fillStyle = "#9a9a9a";
    x.font = "bold " + Math.round(THUMB * 0.4) + "px sans-serif";
    x.textAlign = "center";
    x.textBaseline = "middle";
    x.fillText("?", THUMB / 2, THUMB * 0.45);
    return c.toDataURL("image/png");
}

// Marked absent: a crossed-out box, distinct from the question mark that means
// "not chosen yet".
function absentThumb() {
    const c = document.createElement("canvas");
    c.width = THUMB;
    c.height = THUMB;
    const x = c.getContext("2d");
    x.fillStyle = "#2a2a2a";
    x.fillRect(0, 0, THUMB, THUMB);
    x.strokeStyle = "#b8862b";
    x.lineWidth = 3;
    const m = THUMB * 0.28;
    x.beginPath();
    x.moveTo(m, m); x.lineTo(THUMB - m, THUMB - m);
    x.moveTo(THUMB - m, m); x.lineTo(m, THUMB - m);
    x.stroke();
    return c.toDataURL("image/png");
}

// Cropped from the card the picker already fetched, so confirming costs no request.
function cropFace(imgEl, box) {
    const c = document.createElement("canvas");
    c.width = THUMB;
    c.height = THUMB;
    const x = c.getContext("2d");
    const W = imgEl.naturalWidth;
    const H = imgEl.naturalHeight;
    let sx = box[0] * W;
    let sy = box[1] * H;
    let sw = (box[2] - box[0]) * W;
    let sh = (box[3] - box[1]) * H;
    const pad = Math.max(sw, sh) * 0.18;
    sx -= pad; sy -= pad; sw += pad * 2; sh += pad * 2;
    const side = Math.max(sw, sh);
    sx -= (side - sw) / 2;
    sy -= (side - sh) / 2;
    sx = Math.max(0, Math.min(sx, W - 1));
    sy = Math.max(0, Math.min(sy, H - 1));
    const s = Math.max(1, Math.min(side, W - sx, H - sy));
    x.fillStyle = "#000";
    x.fillRect(0, 0, THUMB, THUMB);
    try {
        x.drawImage(imgEl, sx, sy, s, s, 0, 0, THUMB, THUMB);
    } catch (e) { /* nothing to draw */ }
    return c.toDataURL("image/jpeg", 0.82);
}

function renderStrip(node) {
    const el = node.__h3frStrip;
    if (!el) return;
    const thumbs = (node.properties && node.properties.h3fr_thumbs) || [];
    const picks = String((w(node, "confirmed_pick") || {}).value || "")
        .split(",").filter((s) => s !== "");
    el.innerHTML = "";
    el.title = "Click to choose the subject";
    const n = Math.max(thumbs.length, picks.length, 1);
    for (let i = 0; i < n; i++) {
        const isAbsent = String(picks[i]) === String(ABSENT);
        const d = document.createElement("div");
        d.className = "h3fr-thumb" + (thumbs[i] ? " set" : "");
        const img = document.createElement("img");
        img.src = isAbsent ? absentThumb() : (thumbs[i] || placeholderThumb());
        d.appendChild(img);
        const tag = document.createElement("span");
        tag.className = "h3fr-tag";
        if (n > 1) {
            tag.textContent = "shot " + (i + 1) +
                (isAbsent ? " · absent"
                          : (picks[i] !== undefined ? " · #" + picks[i] : ""));
        } else {
            tag.textContent = picks[0] === undefined ? "none"
                            : isAbsent ? "not in this shot"
                            : "face #" + picks[0];
        }
        d.appendChild(tag);
        el.appendChild(d);
    }
    node.__h3frThumbCount = n;
    reflow(node);
}

// Every URL goes through api.apiURL so it survives ComfyUI being served under a
// sub-path. apiURL rather than fetchApi because these are also <video> and <img>
// src attributes, which cannot go through fetch at all. ComfyUI answers custom
// routes under both / and /api, so the prefix is safe on the pack's own routes.
function previewSrc(value) {
    const v = String(value || "").trim().replace(/^["']|["']$/g, "");
    if (!v) return null;
    // A bare name lives in ComfyUI's input folder and /view can serve it; anything
    // with a path separator or a drive letter needs our own route. Windows paths
    // carry neither a forward slash nor a leading slash, so both have to be tested.
    if (!/[\\\/]/.test(v) && !/^[a-zA-Z]:/.test(v)) {
        return api.apiURL("/view?filename=" + encodeURIComponent(v)
                          + "&type=input&rand=" + Math.random());
    }
    return api.apiURL("/h3_facerefine/preview?path=" + encodeURIComponent(v));
}

const VID_MIN = 120;
const VID_AR = 9 / 16;      // until the clip says otherwise

// Derived from the node's CURRENT width every time it is asked, rather than measured
// once when the clip loads: a cached height keeps its old value when the node is
// resized, so the box stops matching the width and the player letterboxes inside it.
function videoBoxHeight(node) {
    const wide = (node.size ? node.size[0] : 300) - 20;
    const ar = node.__h3frVideoAR || VID_AR;
    return Math.max(VID_MIN, Math.round(wide * ar));
}

function renderPreview(node) {
    const host = node.__h3frVideo;
    if (!host) return;
    host.style.height = videoBoxHeight(node) + "px";
    const src = previewSrc((w(node, "video") || {}).value);
    host.innerHTML = "";
    if (!src) {
        const empty = document.createElement("div");
        empty.className = "h3fr-vid-none";
        empty.textContent = "No video. Use Browse to choose one, or paste a path into video.";
        empty.onclick = () => chooseFile(node);
        host.appendChild(empty);
        return;
    }
    const v = document.createElement("video");
    v.className = "h3fr-vid";
    v.onloadedmetadata = () => {
        // Size the box to the clip's aspect, then hold the player to that box, so it
        // cannot spill over the widget below it.
        node.__h3frVideoAR =
            v.videoHeight && v.videoWidth ? v.videoHeight / v.videoWidth : VID_AR;
        host.style.height = videoBoxHeight(node) + "px";
        reflow(node);
    };
    v.src = src;
    v.controls = true;
    v.muted = true;
    v.preload = "metadata";
    v.onerror = () => {
        host.innerHTML = "";
        const bad = document.createElement("div");
        bad.className = "h3fr-vid-none";
        bad.textContent = "Cannot preview this file. Check the path.";
        host.appendChild(bad);
    };
    host.appendChild(v);
}

// DOM widgets do not resize the node on their own, so the video player and the
// thumbnail strip end up drawn over each other. Re-measure after anything that
// changes their height, and only ever grow - a node the user made taller stays.
function reflow(node) {
    if (!node.computeSize) return;
    const want = node.computeSize();
    const cur = node.size || [0, 0];
    node.setSize([Math.max(cur[0], want[0]), Math.max(cur[1], want[1])]);
    node.setDirtyCanvas(true, true);
}

/* ------------------------------------------------- which widgets are relevant */

const MANUAL = "manual";
const IDENTITY = "identity_reference";

function showWidget(node, widget, show) {
    if (!widget) return;
    if (widget.__h3frType === undefined) {
        widget.__h3frType = widget.type;
        widget.__h3frCompute = widget.computeSize;
    }
    widget.hidden = !show;
    widget.type = show ? widget.__h3frType : "h3frhidden";
    widget.computeSize = show
        ? widget.__h3frCompute
        : () => [0, -4];
    if (widget.element) widget.element.style.display = show ? "" : "none";
}

// select answers "which face". A rule needs select_index, closest_to_xy needs its
// point and frame, manual needs the reviewed answer and the picker. Showing all of
// them at once is what made them look unrelated.
// cut_threshold does nothing while cut detection is off.
function applyCutMode(node) {
    const on = String((w(node, "cut_detection") || {}).value || "none") !== "none";
    showWidget(node, w(node, "cut_threshold"), on);
    reflow(node);
}

// Widgets the tracker stops consulting once face_pick supplies the boxes, the cuts
// and the chosen subject. Greying is a signal, not enforcement - the value is still
// serialised and still reaches Python, which guards itself.
const linked = (node, name) =>
    (node.inputs || []).some((i) => i.name === name && i.link != null);

function setInert(node, name, inert) {
    const wd = w(node, name);
    if (!wd) return;
    wd.disabled = !!inert;
}

function applyPickOverride(node) {
    const pick = linked(node, "face_pick");
    // insightface finds the reference face with its own detector; the crop-based
    // backends run the tracker's detector at its own confidence.
    const cropBackend = ["clip_vision", "ccip"]
        .includes(String((w(node, "identity_model") || {}).value || "insightface"));
    const refDetects = pick && linked(node, "identity_reference") && cropBackend;
    const fallback =
        String((w(node, "fallback_detector") || {}).value || "none") !== "none";

    for (const n of ["select", "select_index", "X", "Y", "frame_index",
                     "cut_detection", "cut_threshold"]) {
        setInert(node, n, pick);
    }
    setInert(node, "detector", pick && !refDetects);
    setInert(node, "confidence", pick && !refDetects && !fallback);
    node.updateComputedDisabled?.();
    node.setDirtyCanvas?.(true, true);
}

// `confidence` gates the detector, never identity matching - two different things
// that both read as "how sure are we". Both nodes carry one.
function labelConfidence(node) {
    const cw = w(node, "confidence");
    if (cw) cw.label = "detector_confidence";
}

function watchPickOverride(node) {
    for (const n of ["identity_model", "fallback_detector"]) {
        const wd = w(node, n);
        if (!wd) continue;
        const prev = wd.callback;
        wd.callback = function () {
            const out = prev ? prev.apply(this, arguments) : undefined;
            applyPickOverride(node);
            return out;
        };
    }
    const conn = node.onConnectionsChange;
    node.onConnectionsChange = function () {
        const out = conn ? conn.apply(this, arguments) : undefined;
        applyPickOverride(node);
        return out;
    };
    labelConfidence(node);
    setTimeout(() => applyPickOverride(node), 0);
}

function watchCutMode(node) {
    const cw = w(node, "cut_detection");
    if (!cw) return;
    const prev = cw.callback;
    cw.callback = function () {
        const out = prev ? prev.apply(this, arguments) : undefined;
        applyCutMode(node);
        return out;
    };
    setTimeout(() => applyCutMode(node), 0);
}

// Each mode answers "which face" a different way, so each brings its own controls:
// a rule ranks and indexes, identity matches a reference, manual asks you.
// X, Y and frame_index answer to closest_to_xy alone. BOTH nodes carry them, and the
// tracker has none of the rest of the mode machinery, so this is the part they share.
function applyXYMode(node) {
    const byXY = String((w(node, "select") || {}).value || "") === "closest_to_xy";
    for (const n of ["X", "Y", "frame_index"]) showWidget(node, w(node, n), byXY);
    reflow(node);
    return byXY;
}

function watchSelect(node, fn) {
    const sw = w(node, "select");
    if (!sw) return;
    const prev = sw.callback;
    sw.callback = function () {
        const r = prev ? prev.apply(this, arguments) : undefined;
        fn(node);
        return r;
    };
    setTimeout(() => fn(node), 0);
}

function applyMode(node) {
    applyCutMode(node);
    const sel = String((w(node, "select") || {}).value || "");
    const manual = sel === MANUAL;
    const byId = sel === IDENTITY;
    const byRule = !manual && !byId;
    const byXY = applyXYMode(node);
    showWidget(node, w(node, "select_index"), byRule);
    showWidget(node, w(node, "confirmed_pick"), manual);
    showWidget(node, w(node, "identity_model"), byId);
    showWidget(node, w(node, "identity_threshold"), byId);
    const pick = (node.widgets || []).find((x) => x.__h3frPick);
    showWidget(node, pick, manual);
    const xyBtn = (node.widgets || []).find((x) => x.__h3frXYBtn);
    showWidget(node, xyBtn, byXY);
    const xyHost = (node.widgets || []).find((x) => x.name === "h3fr_xy");
    showWidget(node, xyHost, byXY);
    if (node.__h3frXY) node.__h3frXY.style.display = byXY ? "" : "none";
    const strip = (node.widgets || []).find((x) => x.name === "h3fr_preview");
    showWidget(node, strip, manual);
    if (node.__h3frStrip) node.__h3frStrip.style.display = manual ? "" : "none";
    reflow(node);
}

// closest_to_xy is typed, not clicked, so the only question is "where did that land".
// Shown on the node beside the numbers rather than in a dialog: a dialog would put the
// picture somewhere other than the fields being edited, and would look like a second way
// of choosing a face by hand, which manual already is.
async function previewCoordinates(node) {
    const host = node.__h3frXY;
    if (!host) return;
    const names = ["video", "detector", "confidence", "X", "Y", "frame_index",
                   "skip_first_frames", "frame_load_cap", "select_every_nth"];
    const params = {};
    for (const n of names) {
        const x = w(node, n);
        if (x) params[n] = x.value;
    }
    // The box is closed to zero height until there is something to put in it, so every
    // write has to open it and re-lay the node out. Without that, "reading..." and any
    // error land inside a 0px overflow:hidden div and the button looks like it did
    // nothing at all.
    const say = (text) => {
        host.innerHTML = "";
        const p = document.createElement("div");
        p.className = "h3fr-xymsg";
        p.textContent = text;
        host.appendChild(p);
        node.__h3frXYState = "msg";
        reflow(node);
    };
    say("Reading frame " + (params.frame_index ?? 0) + "…");
    try {
        const r = await fetch(api.apiURL("/h3_facerefine/preview_xy"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(params),
        });
        const d = await r.json();
        if (d.error) { say(d.error); notify(d.error); return; }
        host.innerHTML = "";
        const img = document.createElement("img");
        img.src = "data:image/jpeg;base64," + d.jpg;
        host.appendChild(img);
        const tag = document.createElement("div");
        tag.className = "h3fr-xytag";
        tag.textContent = d.faces
            ? "frame " + d.frame + " · " + d.faces + " face(s) · nearest is face " + d.chose
            : "frame " + d.frame + " · no face detected here";
        host.appendChild(tag);
        node.__h3frXYState = "img";
        reflow(node);
    } catch (e) {
        say(String(e));
        notify(e);
    }
}

function refreshButton(node) {
    const b = (node.widgets || []).find((x) => x.__h3frPick);
    if (!b) return;
    const v = String((w(node, "confirmed_pick") || {}).value || "").trim();
    if (!v) { b.name = "Pick faces"; node.setDirtyCanvas(true, true); return; }
    // "Faces selected: 0" reads as a COUNT - none selected. The hash marks each one as
    // an identifier, and keeps the actual indices, which a shot count would throw away.
    const parts = v.split(",").map((x) => x.trim()).filter((x) => x !== "");
    const named = parts.map((x) =>
        String(x) === String(ABSENT) ? "absent" : "#" + x).join(", ");
    b.name = (parts.length === 1 ? "Face selected: " : "Faces selected: ")
             + named + "  —  change";
    node.setDirtyCanvas(true, true);
}

function setPicks(node, value, thumbs) {
    const cw = w(node, "confirmed_pick");
    if (cw) {
        cw.value = value;
        if (cw.callback) cw.callback(value);
    }
    node.properties = node.properties || {};
    node.properties.h3fr_thumbs = thumbs || [];
    refreshButton(node);
    renderStrip(node);
}

// A reviewed answer is one face index per shot, held positionally. Re-cutting the
// clip or re-detecting it leaves those entries describing shots that are not there,
// so the selection is dropped and the button goes back to asking for one.
// A reviewed pick is an index into one frame's left-to-right order. Anything that
// changes WHICH faces are found renumbers that order, so the saved index would point
// at a different person - confidence included, since it sets how many are detected.
const INVALIDATES = ["video", "detector", "confidence", "cut_detection",
                     "skip_first_frames", "frame_load_cap", "select_every_nth"];

function dropPicks(node) {
    if (node.__h3frLoading) return;
    if (!String((w(node, "confirmed_pick") || {}).value || "").trim()) return;
    setPicks(node, "", []);
}

function watchInvalidators(node) {
    const chain = (name, when) => {
        const wd = w(node, name);
        if (!wd) return;
        const prev = wd.callback;
        wd.callback = function () {
            const out = prev ? prev.apply(this, arguments) : undefined;
            if (!when || when()) dropPicks(node);
            return out;
        };
    };
    INVALIDATES.forEach((n) => chain(n));
    // cut_threshold only feeds the scan while cut detection is actually running.
    chain("cut_threshold",
          () => String((w(node, "cut_detection") || {}).value || "none") !== "none");
}

/* ------------------------------------------------------------------- browse */

// A browser cannot hand back a real path, so choosing a file off disk means
// uploading it into ComfyUI's input folder - which is what the native loaders do.
function chooseFile(node) {
    const inp = document.createElement("input");
    inp.type = "file";
    inp.accept = "video/*";
    inp.style.display = "none";
    document.body.appendChild(inp);
    inp.onchange = async () => {
        const file = inp.files && inp.files[0];
        inp.remove();
        if (!file) return;
        const vw = w(node, "video");
        const was = vw ? vw.value : "";
        if (vw) {
            vw.value = "uploading " + file.name + "…";
            node.setDirtyCanvas(true, true);
        }
        try {
            const body = new FormData();
            body.append("image", file);
            body.append("type", "input");
            body.append("overwrite", "true");
            const r = await fetch(api.apiURL("/upload/image"), { method: "POST", body });
            if (!r.ok) throw new Error("HTTP " + r.status);
            const data = await r.json();
            const name = data.subfolder ? data.subfolder + "/" + data.name : data.name;
            if (vw) {
                vw.value = name;
                if (vw.callback) vw.callback(name);
            }
            setPicks(node, "", []);          // a new video invalidates the old selection
            renderPreview(node);
        } catch (e) {
            if (vw) vw.value = was;
            notify("Upload failed: " + e);
        }
        node.setDirtyCanvas(true, true);
    };
    inp.click();
}

/* --------------------------------------------------------------------- pick */

async function openPicker(node, force) {
    const names = ["video", "detector", "confidence", "select", "X", "Y",
                   "frame_index", "cut_detection", "cut_threshold",
                   "skip_first_frames", "frame_load_cap", "select_every_nth"];
    const params = {};
    for (const n of names) {
        const x = w(node, n);
        if (x) params[n] = x.value;
    }

    const d = dialog("Reading the video…");
    const note = document.createElement("div");
    note.style.cssText = "font:12px sans-serif;color:#aaa";
    note.textContent = "The result is reused until the video or the detection settings "
        + "change.";
    const bar = document.createElement("div");
    bar.className = "h3fr-prog";
    bar.appendChild(document.createElement("i"));
    d.body.appendChild(note);
    d.body.appendChild(bar);
    d.foot.appendChild(button("Cancel", "", d.close));

    const t0 = Date.now();
    const tick = setInterval(() => {
        if (!d.body.isConnected) { clearInterval(tick); return; }
        d.msg.textContent = ((Date.now() - t0) / 1000).toFixed(0) + "s elapsed";
    }, 250);

    // The scan is one blocking request, so it reports its stage over the websocket.
    const onProgress = ({ detail: p }) => {
        if (!p || !d.body.isConnected) return;
        d.head.textContent = p.stage;
        if (p.detail) note.textContent = p.detail;
        if (p.total > 0) {
            bar.classList.add("h3fr-known");
            bar.firstChild.style.width =
                Math.max(2, Math.round((p.done / p.total) * 100)) + "%";
        } else {
            bar.classList.remove("h3fr-known");
            bar.firstChild.style.width = "";
        }
    };
    api.addEventListener("h3_facerefine/progress", onProgress);
    const stopProgress = () =>
        api.removeEventListener("h3_facerefine/progress", onProgress);

    let data;
    try {
        const r = await fetch(api.apiURL("/h3_facerefine/scan"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(force ? { ...params, force: true } : params),
        });
        data = await r.json();
    } catch (e) {
        clearInterval(tick); stopProgress();
        d.body.textContent = "Request failed: " + e;
        d.msg.textContent = "";
        return;
    }
    clearInterval(tick); stopProgress();
    if (!d.body.isConnected) return;
    if (data.error) {
        d.head.textContent = data.busy ? "Waiting on the render"
                                       : "Could not read the video";
        d.body.textContent = data.error;
        d.msg.textContent = "";
        if (data.busy) {
            // sharing the GPU is a judgement call about the card, not a rule
            d.foot.appendChild(button("Scan anyway", "", () => {
                d.close();
                openPicker(node, true);
            }));
        }
        return;
    }

    const shots = data.shots || [];
    d.head.textContent = data.source + " — " + data.frames + " frames, " +
        shots.length + " shot(s)";
    d.body.innerHTML = "";
    d.foot.innerHTML = "";
    const msg = document.createElement("span");
    msg.className = "h3fr-msg";
    msg.textContent = data.cached ? "reused the previous scan" : "";
    d.foot.appendChild(msg);

    const cw = w(node, "confirmed_pick");
    const prev = String(cw ? cw.value || "" : "").split(",").map((x) => parseInt(x, 10));
    const chosen = shots.map((s, i) => {
        if (prev[i] === ABSENT) return ABSENT;
        return Number.isInteger(prev[i]) && prev[i] >= 0 && prev[i] < s.faces ? prev[i] : 0;
    });
    const imgs = [];
    const noneBtns = [];

    // A rank is per-frame, so the same person is usually a DIFFERENT number after a
    // cut. This row shows the face chosen in each shot side by side, so you can see
    // at a glance whether you have picked one person or several.
    let sameRow = null;
    if (shots.length > 1) {
        const same = document.createElement("div");
        same.className = "h3fr-same";
        const p = document.createElement("p");
        p.innerHTML = "Pick <b>the same person</b> in every shot. A cut renumbers the " +
            "faces, so that person is usually a different number in each one. " +
            "These are your picks:";
        sameRow = document.createElement("div");
        sameRow.className = "h3fr-same-row";
        same.appendChild(p);
        same.appendChild(sameRow);
        d.body.appendChild(same);
    }

    const updateSame = () => {
        if (!sameRow) return;
        sameRow.innerHTML = "";
        shots.forEach((shot, i) => {
            const cell = document.createElement("div");
            cell.className = "h3fr-same-cell";
            const c = document.createElement("canvas");
            c.width = 56; c.height = 56;
            const x = c.getContext("2d");
            const img = imgs[i];
            const bx = chosen[i] === ABSENT ? null : (shot.boxes && shot.boxes[chosen[i]]);
            if (chosen[i] === ABSENT) {
                x.fillStyle = "#2a2a2a"; x.fillRect(0, 0, 56, 56);
                x.strokeStyle = "#b8862b"; x.lineWidth = 2;
                x.beginPath(); x.moveTo(14, 14); x.lineTo(42, 42);
                x.moveTo(42, 14); x.lineTo(14, 42); x.stroke();
                cell.classList.add("absent");
            } else if (img && img.complete && bx) {
                const W = img.naturalWidth, H = img.naturalHeight;
                let sx = bx[0] * W, sy = bx[1] * H;
                let sw = (bx[2] - bx[0]) * W, sh = (bx[3] - bx[1]) * H;
                const pad = Math.max(sw, sh) * 0.18;
                sx -= pad; sy -= pad; sw += pad * 2; sh += pad * 2;
                const side = Math.max(sw, sh);
                sx = Math.max(0, Math.min(sx - (side - sw) / 2, W - 1));
                sy = Math.max(0, Math.min(sy - (side - sh) / 2, H - 1));
                const sd = Math.max(1, Math.min(side, W - sx, H - sy));
                try { x.drawImage(img, sx, sy, sd, sd, 0, 0, 56, 56); } catch (e) { /* */ }
                cell.classList.add("set");
            } else {
                x.fillStyle = "#2a2a2a"; x.fillRect(0, 0, 56, 56);
                x.fillStyle = "#888"; x.font = "20px sans-serif";
                x.textAlign = "center"; x.textBaseline = "middle";
                x.fillText("?", 28, 28);
            }
            const lab = document.createElement("div");
            lab.textContent = "shot " + (i + 1) +
                (chosen[i] === ABSENT ? " · absent" : " · " + chosen[i]);
            cell.appendChild(c);
            cell.appendChild(lab);
            sameRow.appendChild(cell);
        });
    };

    shots.forEach((shot, si) => {
        const wrap = document.createElement("div");
        wrap.className = "h3fr-shot";
        const a = shot.segment[0];
        const b = shot.segment[1];
        const h = document.createElement("h4");
        if (shots.length > 1) {
            h.textContent = "Shot " + (si + 1) + " — frames " + a + "-" + (b - 1) +
                (shot.frame < 0 ? " — no face here" : ", showing frame " + shot.frame);
        } else {
            h.textContent = shot.frame < 0 ? "No face found" : "Frame " + shot.frame;
        }
        wrap.appendChild(h);

        if (shot.jpg) {
            const iw = document.createElement("div");
            iw.className = "h3fr-imgwrap";
            const img = document.createElement("img");
            img.src = "data:image/jpeg;base64," + shot.jpg;
            iw.appendChild(img);
            imgs[si] = img;

            const nums = document.createElement("div");
            nums.className = "h3fr-nums";
            const hits = [];
            const btns = [];
            const mark = () => {
                hits.forEach((el, i) => el.classList.toggle("sel", i === chosen[si]));
                btns.forEach((el, i) => el.classList.toggle("sel", i === chosen[si]));
                if (noneBtns[si]) {
                    noneBtns[si].classList.toggle("sel", chosen[si] === ABSENT);
                }
            };
            (shot.boxes || []).forEach((bx, i) => {
                const hit = document.createElement("div");
                hit.className = "h3fr-hit";
                hit.style.left = bx[0] * 100 + "%";
                hit.style.top = bx[1] * 100 + "%";
                hit.style.width = (bx[2] - bx[0]) * 100 + "%";
                hit.style.height = (bx[3] - bx[1]) * 100 + "%";
                hit.title = "face " + i;
                hit.onclick = () => { chosen[si] = i; mark(); updateSame(); };
                iw.appendChild(hit);
                hits.push(hit);
                const nb = button(String(i), "", () => {
                    chosen[si] = i; mark(); updateSame();
                });
                nums.appendChild(nb);
                btns.push(nb);
            });
            const none = button("not in this shot", "none", () => {
                chosen[si] = ABSENT; mark(); updateSame();
            });
            none.title = "The subject does not appear here. These frames keep their " +
                "original pixels instead of being refined.";
            nums.appendChild(none);
            noneBtns[si] = none;
            wrap.appendChild(iw);
            wrap.appendChild(nums);
            mark();
            img.onload = updateSame;
        }
        d.body.appendChild(wrap);
    });
    updateSame();

    d.foot.appendChild(button("Clear", "", () => {
        setPicks(node, "", []);
        d.close();
    }));
    d.foot.appendChild(button("Cancel", "", d.close));
    d.foot.appendChild(button("Use these", "go", () => {
        const thumbs = shots.map((s, i) => {
            if (chosen[i] === ABSENT) return null;
            if (imgs[i] && s.boxes && s.boxes[chosen[i]]) {
                return cropFace(imgs[i], s.boxes[chosen[i]]);
            }
            return null;
        });
        setPicks(node, chosen.join(","), thumbs);
        d.close();
    }));
}

/* ----------------------------------------------------------------- register */

app.registerExtension({
    name: "h3.facerefine.picker",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        // The tracker has the same cut inputs but none of the picker, so it only gets
        // the show/hide.
        if (nodeData.name === TRACKER) {
            const madeT = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = madeT ? madeT.apply(this, arguments) : undefined;
                watchCutMode(this);
                watchSelect(this, applyXYMode);
                watchPickOverride(this);
                return r;
            };
            const cfgT = nodeType.prototype.configure;
            nodeType.prototype.configure = function (info) {
                migrateWidgetValues(info);
                return cfgT ? cfgT.apply(this, arguments) : undefined;
            };
            const confT = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                const r = confT ? confT.apply(this, arguments) : undefined;
                migrateValues(this);
                setTimeout(() => {
                    applyCutMode(this); applyXYMode(this); applyPickOverride(this);
                }, 0);
                return r;
            };
            return;
        }
        if (nodeData.name !== NODE) return;

        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = created ? created.apply(this, arguments) : undefined;

            const browse = this.addWidget("button", "Browse…", null, () => chooseFile(this));
            const pick = this.addWidget("button", "Pick faces", null, () => openPicker(this));
            const xyBtn = this.addWidget("button", "Preview coordinates", null,
                                         () => previewCoordinates(this));
            // Buttons hold no value: serialising them would take slots in
            // widgets_values and shift every saved setting after them.
            if (xyBtn) xyBtn.__h3frXYBtn = true;
            [browse, pick, xyBtn].forEach((b) => {
                if (!b) return;
                b.serialize = false;
                b.serializeValue = () => undefined;
            });
            if (pick) pick.__h3frPick = true;

            const self = this;

            if (this.addDOMWidget) {
                styleOnce();

                const el = document.createElement("div");
                el.className = "h3fr-strip h3fr-striphost";
                el.onclick = () => openPicker(self);
                const dw = this.addDOMWidget("h3fr_preview", "div", el, {
                    serialize: false,
                    hideOnZoom: false,
                });
                if (dw) {
                    // enough rows for however many shots there are, so the strip never
                    // spills over the widget below it
                    dw.computeSize = () => {
                        const wide = (self.size ? self.size[0] : 300) - 16;
                        const per = Math.max(1, Math.floor(wide / (THUMB + 8)));
                        const rows = Math.ceil((self.__h3frThumbCount || 1) / per);
                        const h = rows * (THUMB + 8) + 10;
                        el.style.height = h + "px";
                        return [self.size ? self.size[0] : 300, h];
                    };
                }
                this.__h3frStrip = el;

                const xy = document.createElement("div");
                xy.className = "h3fr-xyhost";
                const xyw = this.addDOMWidget("h3fr_xy", "div", xy, {
                    serialize: false,
                    hideOnZoom: false,
                });
                if (xyw) {
                    // Collapsed until asked for, so the node does not carry a tall empty
                    // panel in every other mode.
                    xyw.computeSize = () => {
                        const wide = self.size ? self.size[0] : 300;
                        const on = String((w(self, "select") || {}).value || "")
                                   === "closest_to_xy";
                        const st = on ? self.__h3frXYState : null;
                        const h = st === "img" ? Math.round(wide * 0.62) + 22
                                : st === "msg" ? 34
                                : 0;
                        xy.style.height = h ? h + "px" : "0px";
                        return [wide, h];
                    };
                }
                this.__h3frXY = xy;

                const vid = document.createElement("div");
                vid.className = "h3fr-vidhost";
                const vwid = this.addDOMWidget("h3fr_clip", "div", vid, {
                    serialize: false,
                    hideOnZoom: false,
                });
                if (vwid) {
                    // The host is resized here too, not just measured: computeSize is what
                    // runs while the node is being dragged, so it is the only place that
                    // sees every new width.
                    vwid.computeSize = () => {
                        const h = videoBoxHeight(self);
                        vid.style.height = h + "px";
                        return [self.size ? self.size[0] : 300, h + 8];
                    };
                }
                this.__h3frVideo = vid;
            }

            watchCutMode(this);
            watchInvalidators(this);
            labelConfidence(this);

            // select decides which widgets are even relevant.
            const selWidget = w(this, "select");
            if (selWidget) {
                const prevSel = selWidget.callback;
                selWidget.callback = function () {
                    const out = prevSel ? prevSel.apply(this, arguments) : undefined;
                    applyMode(self);
                    return out;
                };
            }

            // Typing or pasting a path should update the player too.
            const vidWidget = w(this, "video");
            if (vidWidget) {
                const prevCb = vidWidget.callback;
                vidWidget.callback = function () {
                    const out = prevCb ? prevCb.apply(this, arguments) : undefined;
                    renderPreview(self);
                    return out;
                };
            }

            setTimeout(() => {
                refreshButton(self);
                renderStrip(self);
                renderPreview(self);
                applyMode(self);
            }, 0);
            return r;
        };

        const cfgS = nodeType.prototype.configure;
        nodeType.prototype.configure = function (info) {
            migrateWidgetValues(info);
            return cfgS ? cfgS.apply(this, arguments) : undefined;
        };
        const configure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            this.__h3frLoading = true;
            const r = configure ? configure.apply(this, arguments) : undefined;
            migrateValues(this);
            setTimeout(() => {
                this.__h3frLoading = false;
                refreshButton(this);
                renderStrip(this);
                renderPreview(this);
                applyMode(this);
            }, 0);
            return r;
        };
    },
});
