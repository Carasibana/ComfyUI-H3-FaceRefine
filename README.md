# ComfyUI-H3-FaceRefine

**A ComfyUI custom node set to refine and improve the quality of small faces in MiniMax H3 video.**

MiniMax H3 renders faces poorly when the head occupies a small fraction of the frame. This is a property
of head-size-in-frame, not of output resolution, so it persists at 720p and above. These nodes
detect the face on every frame, crop to it so it fills a canvas, let H3 re-generate it, and
composite the result back into the original video.

Modelled on [Impact Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack)'s **FaceDetailer**,
adapted from stills to video.

---

## Results

Source on the left, refined on the right:

![source vs refined, side by side](screenshots/COMPARISON.gif)

Single frame at full resolution:

| Source | Refined |
|---|---|
| ![source frame](screenshots/INPUT_00001.png) | ![refined frame](screenshots/REFINED_00001.png) |

The reason it works is what H3 is handed. Instead of a distant head a few dozen pixels tall, it
gets the face tracked and normalized to fill the canvas:

| Crop in: what H3 sees | Crop out: what H3 returns |
|---|---|
| ![input crop](screenshots/CROPS_INPUT_00001.gif) | ![refined crop](screenshots/CROPS_00001.gif) |
| [full-res still](screenshots/CROPS_INPUT_00001.png) | [full-res still](screenshots/CROPS_00001.png) |

These two are what to watch for temporal behaviour: the box has to sit still on a moving subject,
or the refined face boils.

---

## Installation

Clone into `ComfyUI/custom_nodes/`:

```bash
git clone https://github.com/Carasibana/ComfyUI-H3-FaceRefine.git
```

Restart ComfyUI. The nodes appear under **MiniMax H3/Face Refine**.

### Requirements

**For the nodes themselves:**

| | |
|---|---|
| ComfyUI with MiniMax H3 support | H3's own nodes are **core** (`comfy_extras/nodes_minimax_h3.py`), not an add-on, you just need a build recent enough to have them |
| a face detector | e.g. `face_yolov8m.pt` in `models/ultralytics/bbox/`. The one thing you must supply yourself |

Python packages (`ultralytics`, `scipy`, `insightface`) install automatically from
`requirements.txt` / `pyproject.toml`.

> **A note on onnxruntime.** `insightface` needs it, and this pack deliberately does not pin
> a variant. If you install `onnxruntime-gpu` *alongside* an existing `onnxruntime`, the
> CPU-only package shadows it, `CUDAExecutionProvider` disappears, and identity matching
> silently runs on CPU. Install one or the other, not both. Check with:
> `python -c "import onnxruntime; print(onnxruntime.get_available_providers())"`

**Additionally, to run the example workflows:**

| | |
|---|---|
| [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) | video load and save |
| [ComfyUI-H3-NativeAudioLock](https://github.com/Shrek3OnVH5/MiniMax-H3-NativeAudio-MusicVideo-Workflow) | supplies `MiniMaxH3NativeAudioLock`, which drives lipsync. It ships inside that repository under `custom_nodes/ComfyUI-H3-NativeAudioLock`, so copy that folder into your own `ComfyUI/custom_nodes/` |
| [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) | only if loading H3 as GGUF |
| [ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack) | only for the SAM workflow (supplies `SAMLoader`) |
| a SAM model in `models/sams/` | as above |

Model lookups go through ComfyUI's `folder_paths`, so anything registered in
`extra_model_paths.yaml` is found automatically.

### Models

| Model | Goes in | Source |
|---|---|---|
| `face_yolov8m.pt` | `models/ultralytics/bbox/` | [Bingsu/adetailer](https://huggingface.co/Bingsu/adetailer/blob/main/face_yolov8m.pt) |
| `person_yolov8m-seg.pt` *(optional, for `fallback_detector`)* | `models/ultralytics/segm/` | [Bingsu/adetailer](https://huggingface.co/Bingsu/adetailer/blob/main/person_yolov8m-seg.pt) |
| MiniMax H3 diffusion model | `models/diffusion_models/` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) |
| Qwen3-VL text encoder | `models/text_encoders/` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) |
| `minimax_h3_video_vae_fp16.safetensors` | `models/vae/` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/vae) |
| `minimax_h3_audio_vae_fp32.safetensors` | `models/vae/` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/vae) |
| `minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy.safetensors` *(optional, big speed-up)* | `models/loras/` | [Kijai/MiniMax-H3_comfy](https://huggingface.co/Kijai/MiniMax-H3_comfy/tree/main/loras), distilled by [lightx2v](https://huggingface.co/lightx2v/Minimax-h3-Turbo) |
| `sam_vit_b_01ec64.pth` *(SAM workflow only)* | `models/sams/` | [Meta segment-anything](https://github.com/facebookresearch/segment-anything#model-checkpoints) |
| InsightFace `buffalo_l` *(crowd tracking)* | `models/insightface/` | downloaded automatically on first use |

H3's original weights are [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3); the
Comfy-Org repository above is the repackaged form ComfyUI expects. The example workflows load the
model and text encoder as **GGUF** through [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF),
which is what lets H3 fit in 12 GB. Swap those two loaders for the stock `UNETLoader` and
`CLIPLoader` if you have VRAM for full-precision weights.

Only the face detector is genuinely required by the nodes themselves. Everything else is needed
because the pipeline runs H3, so if you are already generating H3 video you will have it already.

---

## Quick start

Two example workflows are included, both annotated in-graph:

| Template | Paste mask | Extra dependency |
|---|---|---|
| **H3 Face Refine** | dilated + blurred rect | none |
| **H3 Face Refine SAM** | true face-shaped (SAM) | Impact Pack |

Both are installed as ComfyUI **workflow templates**. Open them from
**Workflow → Browse Templates** once the pack is installed, no file hunting required.
They also live in [`example_workflows/`](example_workflows) if you would rather load the JSON
directly.

**Start with the rect one.** Load it, then set:

1. **Source clip**: point the loader at your video.
2. **Ref image(s)**: the same character references you generated the clip with.
3. **Vocals**: an isolated vocal track, for lipsync.
4. **Prompt**: the clip's own prompt.
5. **`length`** on the H3 node: your clip's exact frame count.

Everything else can stay at the shipped values for a first run.

```
source clip ───┬──────────────────────────────────────────────────────┐
               ▼                                                      │
      H3 Face Track + Crop ──┬── crops ─────────────────┐             │
               │             │                          ▼             │
               │             │     MiniMaxH3ReferenceToVideo          │
               │             │        (refs; W/H <- canvas_w/h)       │
               │             │                          │             │
               │             └──────────► H3 Inject Video Latent      │
               │                                        │             │
               │                 MiniMaxH3NativeAudioLock <- vocals   │
               │                                        │             │
               │                        H3 Per-Frame Denoise          │
               │                                        │             │
               │              SamplerCustomAdvanced ─► VAEDecode      │
               │                                        │             │
               └── transform ─────────► H3 Face Stitch Back ◄─────────┘
                                                │
                                  original audio ──► save
```

The `transform` output is the spine of the whole thing: it records where every crop came from, so
the stitch can put each refined face back exactly where it belongs.

---

# Node reference

## H3 Face Track + Crop

<img src="screenshots/H3%20Face%20Track%20%2B%20Crop.png" alt="H3 Face Track + Crop" width="380">

Detects the face on every frame, fills gaps where detection fails, smooths the trajectory, and
emits a constant-size batch of crops plus the `transform` needed to paste results back.

**Inputs**

| Input | Default | What it does |
|---|---|---|
| `images` | - | The clip to refine. Connect your video loader. |
| `detector` | first found | Face detection model from `models/ultralytics/`. `face_yolov8m.pt` is a good default. |
| `confidence` | `0.35` | Detection threshold. Lower catches more profiles and partial faces at the cost of false positives. |
| `crop_factor` | `2.5` | Crop side as a multiple of face **height**. 2.5 puts the face at ~40% of the crop. Bigger gives more context so the seam lands in hair and background, but less magnification. **2.0-3.0 is the useful range.** |
| `canvas_width` / `canvas_height` | `512` | Resolution H3 generates at. Ignored unless `canvas_mode` is `manual`. 768 is H3's native short edge and gives the best faces, at 2.25× the cost of 512. |
| `canvas_mode` | `manual` | `manual` uses the two values above. `auto_no_downscale` sizes from the largest crop so no frame is ever downscaled. `auto_capped_768` does the same but clamps to 768, a sane VRAM ceiling and the best default. |
| `smooth_window` | `21` | Frames of smoothing on the crop **centre**. 21 at 24 fps is ~0.9 s. Raise if the box shivers, lower if it lags fast head movement. |
| `size_smooth_window` | `51` | Frames of smoothing on the crop **size**. Deliberately larger than the centre window, because size jitter makes the crop breathe, which changes the resample factor every frame and reads as shimmer. |
| `smooth_method` | `gaussian` | `gaussian` rejects jitter best. `savgol` preserves the shape of a push-in at large windows. `moving_average` is a plain boxcar and leaves residual jitter. |
| `size_mode` | `per_frame` | `per_frame` holds the face at a constant fraction of every crop, which is correct for push-ins. `max_of_clip` uses one size throughout, only useful when the shot is static. |
| `identity_reference` *(opt)* | - | A clear face image of the person to track. Picks the subject by identity rather than size. |
| `identity_track` *(opt)* | `True` | Hold one subject through a crowd. Continuity decides most frames; the identity embedding is consulted only when two candidates are similarly plausible or their boxes overlap. |
| `identity_threshold` *(opt)* | `0.28` | Minimum cosine similarity to accept a face as the reference person. Below it, the frame falls back to continuity, which is what carries tracking through profiles and occlusion. |
| `select` *(opt)* | `largest` | How the faces in a frame are **ranked**. `largest`, `area`, `confidence`, `most_central`, or the raw box coordinates `x1`, `x2`, `y1`, `y2`, `center_x`, `center_y`. See [Choosing which face](#choosing-which-face). |
| `fallback_detector` *(opt)* | `none` | Used only on frames where the face detector finds nothing. A person/body model gives a real head position from the top of the body box, which beats interpolating blindly. |
| `fallback_head_frac` *(opt)* | `0.5` | Head centre as a multiple of face height below the top of the person box. 0.5 suits a head seen from behind. |
| `select_order` *(opt)* | `descending` | Direction of the ranking. `descending` puts the **highest** value of the metric at index 0, `ascending` the lowest. |
| `select_index` *(opt)* | `0` | Which face in that ranking to track. `0` is the first, `1` the second, and so on. |

**Outputs**

| Output | Goes to |
|---|---|
| `crops` | `H3 Inject Video Latent` → `images`, and `H3 Face Mask (SAM)` → `crops` |
| `transform` | `H3 Face Stitch Back`, `H3 Per-Frame Denoise`, `H3 Face Mask (SAM)` |
| `preview` | Optional: a debug view of the tracked boxes |
| `report` | Text summary: detections, gaps, magnification warnings |
| `canvas_w` / `canvas_h` | **Must** be wired to the H3 node's `width` / `height` |
| `face_index_preview` | Optional: one numbered frame per face index. This is how you find out which number to put in `select_index` |

> **Wire the canvas, don't type it.** In the `auto_*` modes this node chooses the size. If the H3
> node's `width`/`height` disagree, the latent shapes differ and injection refuses.

> Watch `report` for `magnification < 1.0x`. That means the crop is being *downscaled* into the
> canvas and real detail is being discarded. Raise the canvas, or skip clips that are close-up
> throughout since they have nothing to gain.

### Choosing which face

`select`, `select_order` and `select_index` work as one rule: **rank the faces in a frame, then take
the *n*th**. They apply only when no `identity_reference` is connected.

| `select` | ranks by |
|---|---|
| `largest` | face **height** — the same measurement the tracker uses everywhere else |
| `area` | box area, so a wide face outranks a tall thin one |
| `confidence` | the detector's own score |
| `most_central` | nearness to the frame centre |
| `x1` / `x2` | left edge / right edge |
| `y1` / `y2` | top edge / bottom edge |
| `center_x` / `center_y` | box centre |

`select_order` is the direction, and it applies to every metric including `largest`:

| | `descending` (default) | `ascending` |
|---|---|---|
| `largest` | biggest face first | **smallest face first** |
| `most_central` | most central first | furthest from the centre first |
| `x1` | rightmost first | **leftmost first** |
| `y1` | furthest down the frame first | **topmost first** |

So `x1` + `ascending` numbers a line-up left to right, `y1` + `ascending` numbers it top to bottom,
and `largest` + `descending` + index 0 is the original behaviour, unchanged.

`select_index` then picks out of that ranking: `0` is the first, `1` the second. Connect
**`face_index_preview`** to a `PreviewImage` to see which number is who. It emits one frame per
index — the first frame in the clip where that index exists — with every detected face outlined and
numbered, and that index's face highlighted in green. It shows at most 32 indices; higher ones are
still selectable, and the report says when it capped.

Three things worth knowing:

- **A rank is a per-frame property, not an identity.** The pick is made **once**, on the first frame
  that actually contains the requested index, and continuity carries the subject from there. Ranking
  every frame independently would hop between people the moment two of them cross or change size.
  The report prints the frame it locked on at.
- **Anything before that frame is interpolated.** If you ask for index 2 and the clip opens on a
  single face, those opening frames have no index 2 to lock onto, so they are filled in from the
  lock-on frame and faded out of the composite, exactly like a detection dropout. The report counts
  them, and warns if there are more than a dozen.
- **The identity anchor follows your pick.** With no reference image the node normally assumes the
  biggest face is the subject — which used to override `most_central` in a crowd. Now, once you
  select a different subject, the anchor is rebuilt from *your* pick, so identity matching holds
  that person rather than dragging the crop back onto the dominant face.

A connected `identity_reference` outranks all of this: it decides the subject on its own, `select`
and `select_index` are ignored (the report says so), and no frames are held back.

If the clip never shows as many faces at once as you asked for, `select_index` is clamped to what
exists and the report says so.

**Multiple people:** run the pipeline once per subject, each with that person's `identity_reference`
and their own refs on the H3 node, then chain them, feeding run 1's stitched output in as run 2's
`base_images`. The composites accumulate.

---

## H3 Inject Video Latent (img2img)

<img src="screenshots/H3%20Inject%20Video%20Latent%20(img2img).png" alt="H3 Inject Video Latent (img2img)" width="380">

Encodes real frames into the **video** stream of H3's joint audio-video latent, leaving the audio
stream intact. This is the missing img2img path: H3's stock nodes always build a zeros latent, because
references are conditioning re-injected each step, never a starting point. Without this there
is no video-to-video.

**Inputs**

| Input | What to connect |
|---|---|
| `av_latent` | The `LATENT` output of `MiniMaxH3ReferenceToVideo` |
| `images` | `crops` from **H3 Face Track + Crop** |
| `vae` | The **video** VAE |

**Outputs:** `av_latent` (onwards to `MiniMaxH3NativeAudioLock`) and `report`.

It has no widgets. Strength is set downstream by `BasicScheduler`'s `denoise`, **not** by
`SplitSigmas`. See [Denoise](#denoise) below.

---

## H3 Per-Frame Denoise

<img src="screenshots/H3%20Per-Frame%20Denoise.png" alt="H3 Per-Frame Denoise" width="380">

Varies denoise strength along the temporal axis via the latent's noise mask, so one sampling pass
covers a shot that goes from distant to close.

A single denoise cannot serve a whole clip: a tiny face has no detail to preserve and wants a
strong pass so the model *synthesizes*, while a large face has real detail and wants a gentle pass
so it is not rewritten. This node scales the base denoise per frame by measured face size.

**Inputs**

| Input | Default | What it does |
|---|---|---|
| `av_latent` | - | From `MiniMaxH3NativeAudioLock` |
| `transform` | - | From **H3 Face Track + Crop**. This is where face sizes come from |
| `strength_small_face` | `1.0` | Multiplier where the face is smallest. `1.0` = the full denoise set on `BasicScheduler`. |
| `strength_large_face` | `0.35` | Multiplier where the face is largest. Lower preserves the detail those frames already have. |
| `scale_mode` | `absolute_px` | `absolute_px` keys off real face size in source pixels, which is safe across a batch, since a clip that never has a small face just sits at the baseline. `relative_to_clip` normalizes to that clip's own min/max, so its smallest face always gets the full boost. Use the latter when tuning one clip to its extremes. |
| `face_px_small` | `30.0` | Face height (source px) at or below which full `strength_small_face` applies. |
| `face_px_large` | `120.0` | Face height (source px) at or above which `strength_large_face` applies. |
| `gamma` | `1.0` | Curve on the interpolation. `>1` keeps strength high until the face is genuinely large; `<1` drops it off early. |
| `smooth_frames` | `9` | Smooths the strength curve over time. An abrupt denoise change between neighbouring frames shows up as a texture pop, so be generous. |

**Outputs:** `av_latent` (to `SamplerCustomAdvanced` → `latent_image`) and `report`.

Base denoise and these multipliers are tuned **together**. The example workflows ship a base of
`0.45` which this node scales down on large-face frames. If you bypass this node, drop the base a
long way or every large face gets rewritten.

---

### Denoise

Denoise values do **not** transfer from SDXL-family models. H3 is flow matching with a large sigma
shift:

```
sigma = shift * t / (1 + (shift - 1) * t)
```

At the default shift of 12, `0.25`, an ordinary FaceDetailer value, lands at an effective sigma
of **0.800** and rewrites the frame.

| denoise | effective sigma (shift 12) |
|---|---|
| 0.02 | 0.197 |
| 0.05 | 0.387 |
| 0.15 | ~0.66 |
| 0.25 | 0.800 |

`steps` and `denoise` are **independent**: `BasicScheduler` builds a `steps/denoise`-long
full-range schedule and keeps the lowest `steps+1` sigmas, so 4 steps with a turbo LoRA is both
fast and gentle. Do **not** use `SplitSigmas`. On a 4-step schedule at shift 12 even the last
split point is already sigma 0.800.

Push the ceiling too high and the head drifts relative to the body, a content problem no mask can
hide.

---

## H3 Face Mask (SAM)

<img src="screenshots/H3%20Face%20Mask%20(SAM).png" alt="H3 Face Mask (SAM)" width="380">

Produces true face-shaped paste masks instead of a rectangle, temporally smoothed. Optional, and
requires Impact Pack for `SAMLoader`.

**Inputs**

| Input | Default | What it does |
|---|---|---|
| `crops` | - | **The tracker's `crops`, not the decoded result.** See the warning below. |
| `sam_model` | - | From Impact Pack's `SAMLoader` |
| `transform` | - | From **H3 Face Track + Crop** |
| `threshold` | `0.93` | SAM confidence threshold for accepting mask pixels. |
| `dilation` | `0` | Grow the mask. SAM masks are accurate, so they rarely need it. |
| `temporal_smooth` | `5` | Frames of averaging across the mask stack. `1` disables it, and the mask edge will shimmer. |

**Outputs:** `masks` (to `H3 Face Stitch Back` → `masks`) and `report`.

> **Mask the input, never the output.** Wire the tracker's crops here. This matches FaceDetailer,
> which computes its mask from the source image; generation never feeds back into the mask. Mask
> the *generated* result instead and, if the model nudges the face inward, the mask traces the new
> smaller silhouette while the original face pokes out past it, most visibly the nose on profile
> shots. It is also cheaper: no dependency on the sampler, so SAM need not be resident alongside
> the video model.

**Is it worth it?** Often not. A rect mask frequently beats SAM here. SAM traces the face tightly,
so any drift in the refined face lands right on the silhouette, whereas a slightly looser rect puts
the seam in hair and background where it reads far less. Try the rect workflow first.

> If you use SAM masks, drop `feather` on the stitch node to **4-8**. A rect needs far more.

---

## H3 Face Stitch Back

<img src="screenshots/H3%20Face%20Stitch%20Back.png" alt="H3 Face Stitch Back" width="380">

Warps each refined crop back onto the exact float box it came from, colour-matches it, feathers the
edge and composites. A single batched `grid_sample` does the warp, so a trajectory smoothed to
sub-pixel precision is not re-quantized on the way home.

**Inputs**

| Input | Default | What it does |
|---|---|---|
| `base_images` | - | The **original** frames, the same clip fed to the tracker |
| `refined_crops` | - | `VAEDecode` output |
| `transform` | - | From **H3 Face Track + Crop** |
| `paste_region` | `face_only` | What actually composites. `face_only` / `face_ellipse` paste just the detected face box; `full_crop` pastes hair, shoulders and background too and risks a visible rectangle. |
| `mask_dilation` | `16` | Grow the face box before blurring, in canvas px, so the blur has room and the blend does not eat into the face. |
| `feather` | `6` | Gaussian blur radius on the paste mask, in **source pixels**, measured against the final frame, so the blend is the same physical width at any magnification. Use ~24 with a rect mask, 4-8 with SAM. |
| `colour_match` | `1.0` | Match the crop's per-channel mean/std to the region it replaces. The crop and the frame went through independent passes, so without this the face can come back subtly brighter and read as pasted on. |
| `blend` | `1.0` | Global opacity of the refined face. Below 1.0 mixes back toward the original, useful to dial back over-sharpening. |
| `undetected_frames` | `fade_out` | What to do where no face was found. **All** frames still go through H3 either way, which is what keeps it temporally consistent. This only controls pasting. `fade_out` ramps the composite to zero across the gap; `skip` hard-cuts to original pixels; `composite_anyway` risks H3 hallucinating a face onto the back of a head. |
| `feather_scales_with_crop` *(opt)* | `False` | Legacy: treat `feather` as canvas pixels so the blend narrows as the crop shrinks. Leave off. |
| `masks` *(opt)* | - | Per-frame masks from **H3 Face Mask (SAM)**. Overrides `paste_region`. |

**Outputs:** `images`, the finished frames. Send to your video save node.

> **Only the face region composites.** The wide crop exists to give the sampler *context*. It is
> not what gets pasted. Pasting the whole crop covers roughly 88% of the canvas versus 16% for the
> face box, and any change the model made to hair or background returns as a rectangle.

---

## H3 Face Transform Info

<img src="screenshots/H3%20Face%20Transform%20Info.png" alt="H3 Face Transform Info" width="380">

A debug node. Prints the per-frame boxes so you can sanity-check tracking before spending time on a
sampling pass.

| Input | Default | What it does |
|---|---|---|
| `transform` | - | From **H3 Face Track + Crop** |
| `max_rows` | `12` | How many frames to print. |

**Outputs:** `info` (a string). It is an output node, so it displays in the graph directly.

Use it when the stitch looks misaligned, or to confirm gap-filling behaved on a clip where the
subject turns away.

---

## Lipsync

H3 is a **joint** audio-video model. `MiniMaxH3NativeAudioLock` encodes real audio into the audio
stream of the AV latent, sets `noise_mask` to ones for video and zeros for audio so only video
denoises, and the video branch cross-attends to that fixed audio. That is what shapes the mouth.

Feed it an isolated **vocals** track for a cleaner signal. The **original** audio goes separately
into the save node. Two distinct audio paths, easy to confuse.

---

## Gotchas

- **Frame count must sit on H3's 17k+5 grid** (5, 22, 39 … 175, 226, 362). Clips generated by H3
  already do.
- **Cost is `canvas² × frames`.** Auto canvas sizing considers face size only, so on a long clip it
  can pick a canvas that exceeds VRAM and falls back to streaming weights from system RAM, an
  order-of-magnitude slowdown rather than a clean error.
- **`SAMLoader`'s `AUTO` device mode leaves the model on CPU** until `prepare_device()` is called.
  This pack calls it; if you write your own SAM path, missing it makes mask passes 10-50× slower
  with the GPU idle at half power.

---

## Credits

The compositing approach is taken directly from
**[ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack)** by
**[ltdrdata](https://github.com/ltdrdata)**, specifically `FaceDetailer` and its detailer paste.
The dilate-then-blur face-region mask, the crop-for-context-but-paste-only-the-face principle, and
the bbox+SAM masking path are all its design; this pack adapts them to a per-frame video pipeline.
If you find this useful, that project is why.

The lipsync path in the example workflows depends on `MiniMaxH3NativeAudioLock`, from
**[MiniMax-H3-NativeAudio-MusicVideo-Workflow](https://github.com/Shrek3OnVH5/MiniMax-H3-NativeAudio-MusicVideo-Workflow)**
by **[Shrek3OnVH5](https://github.com/Shrek3OnVH5)**. It is not redistributed here, so install it
from that repository.

Also builds on:

- **[MiniMax H3](https://github.com/MiniMax-AI)**, the joint audio-video model being refined
- **[ComfyUI](https://github.com/comfyanonymous/ComfyUI)** by comfyanonymous
- **[ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)** by Kosinkadink
- **[ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)** by city96
- **[Ultralytics YOLO](https://github.com/ultralytics/ultralytics)** for detection, and
  **[InsightFace](https://github.com/deepinsight/insightface)** for identity embeddings

---

## Made with these nodes

[![More Than Words Can Ever Steal](https://img.youtube.com/vi/18iiffk-QWE/maxresdefault.jpg)](https://www.youtube.com/watch?v=18iiffk-QWE)

**[More Than Words Can Ever Steal](https://www.youtube.com/watch?v=18iiffk-QWE)**, a music video by
[Carasibana](https://www.youtube.com/@Carasibana-Music), generated with MiniMax H3. Every shot
where the subject sits at any distance from camera was face-refined with this pack.

---

## Licence

MIT. See [LICENSE](LICENSE).

---

<sub>Built with [Claude Code](https://claude.com/claude-code).</sub>
