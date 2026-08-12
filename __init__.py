"""ComfyUI-H3-FaceRefine

Per-frame face tracking, normalised cropping, AV-latent injection and feathered
stitch-back, so MiniMax H3 can re-generate small/distant faces at a useful scale
and be composited back into the source video.

Why this exists
---------------
H3 renders faces poorly when the head occupies a small fraction of the frame.
That is a property of head-size-in-frame, not of output resolution, so no
post-process upscaler fixes it. The workaround is to crop to the face, hand H3
a sequence where the face is large, refine at low denoise so the result stays
frame-aligned, then paste back.

The crop must be PER FRAME. A single fixed crop degenerates to a wide shot on a
push-in, which hands H3 exactly the small face it fails on.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
