"""Local clipping: ffmpeg subclip + face-aware vertical reframe.

Two stages per highlight:
  1. Cut the source video to [start, end] with ffmpeg (re-encoded, audio kept).
  2. Reframe the cut to the target aspect ratio, tracking faces with OpenCV's
     DNN-based YuNet detector (cv2.FaceDetectorYN — ships in the main cv2
     module since OpenCV 5.x, no contrib needed, and is far more stable
     frame-to-frame than the Haar cascade the original repo used).

     Two reframe modes (--reframe-mode):
       blur (default) — no cropping. The full frame is scaled to fit the
         canvas width and centred; the empty top/bottom bars are filled
         with a zoomed, blurred copy of the same frame (the OpusClip/CapCut
         look). Nothing outside the frame is ever lost.
       crop — the original approach: slide a crop window across the frame,
         centred on the tracked face. Loses whatever falls outside the
         window; kept as an option for single-speaker footage where a
         tight crop reads better than padding.
"""
import os
import subprocess
import time
from typing import Dict, List, Optional, Tuple

from ..config import LOCAL_OUTPUT_DIR

_YUNET_MODEL_PATH = os.path.join(os.path.dirname(__file__), "data", "face_detection_yunet.onnx")

# --- face tracking tuning -----------------------------------------------
DEBOUNCE_FRAMES = 3     # a new detection must repeat this many times (in
                        # roughly the same spot) before it becomes the target
DEBOUNCE_RADIUS = 60    # px: detections within this radius count as "the same" candidate
MAX_SPEED_PX = 18       # px/frame max movement of the tracked centre — turns
                        # any remaining jump into a fast pan instead of a cut
SMOOTHING = 0.15        # exponential pull toward the *confirmed* target
FG_ZOOM = 1.35          # blur mode: how much to zoom the foreground into the
                        # tracked face before fitting it to the canvas width.
                        # 1.0 = full uncropped frame (further away, more context).
                        # Higher = tighter on the speaker (less context, more "focused").


def _safe_remove(path: str, retries: int = 6, delay: float = 0.3) -> None:
    """os.remove with retries. On Windows, a file handle opened by
    cv2.VideoCapture (or held briefly by antivirus/indexing) can outlive the
    Python object that opened it by a few hundred ms, so an immediate
    os.remove() right after cap.release() can raise WinError 32 (file in
    use) even though nothing is really still using it. Not an issue on
    Linux/Mac, where this basically never retries."""
    if not os.path.exists(path):
        return
    last_error: Optional[Exception] = None
    for attempt in range(retries):
        try:
            os.remove(path)
            return
        except PermissionError as e:
            last_error = e
            time.sleep(delay)
    print(f"[clip/local] warning: could not delete temp file {path}: {last_error}", flush=True)


def _ratio(aspect_ratio: str) -> float:
    """Parse '9:16' → 9/16, '1:1' → 1.0."""
    try:
        w, h = aspect_ratio.split(":")
        return float(w) / float(h)
    except (ValueError, ZeroDivisionError):
        return 9.0 / 16.0


def _cut_subclip(source_path: str, start: float, end: float, out_path: str) -> str:
    """ffmpeg -ss start -to end → re-encoded mp4 with audio."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", source_path,
        "-ss", f"{start:.3f}",
        "-to", f"{end:.3f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    return out_path


def _load_face_detector(input_size: Tuple[int, int]):
    """Create the YuNet detector, or None if model unavailable (falls back
    to center-crop without face tracking)."""
    if not os.path.exists(_YUNET_MODEL_PATH) or os.path.getsize(_YUNET_MODEL_PATH) < 10_000:
        print("[clip] YuNet model not found, falling back to center-crop (no face tracking)", flush=True)
        return None
    import cv2  # type: ignore
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except Exception:
        pass
    return cv2.FaceDetectorYN.create(_YUNET_MODEL_PATH, "", input_size, 0.6, 0.3, 5000)


def _detect_face_center(detector, frame) -> Optional[Tuple[int, int]]:
    """Largest detected face's centre point, or None if nobody was found."""
    _, faces = detector.detect(frame)
    if faces is None or len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])[:4]
    return (int(x + w / 2), int(y + h / 2))


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


class _FaceTracker:
    """Turns noisy per-frame detections into a stable crop/pan centre.

    Two independent safeguards against the "jumpy tracking" the original
    Haar-based version had:
      - Debounce: a new detection only becomes the *confirmed* target once
        it repeats DEBOUNCE_FRAMES times close to the same spot. A single
        stray detection (or one dropped frame) can't yank the frame.
      - Speed clamp: however hard we then chase the confirmed target, the
        rendered centre can move at most MAX_SPEED_PX per frame — any
        remaining jump becomes a fast pan, never an instant cut.
    """

    def __init__(self, default_center: Tuple[int, int]):
        self.current: Tuple[float, float] = default_center
        self.confirmed_target: Tuple[float, float] = default_center
        self._candidate: Optional[Tuple[int, int]] = None
        self._candidate_count = 0

    def update(self, detection: Optional[Tuple[int, int]]) -> Tuple[int, int]:
        if detection is not None:
            if self._candidate is not None and _dist(detection, self._candidate) <= DEBOUNCE_RADIUS:
                self._candidate_count += 1
            else:
                self._candidate = detection
                self._candidate_count = 1
            if self._candidate_count >= DEBOUNCE_FRAMES:
                self.confirmed_target = self._candidate

        tx, ty = self.confirmed_target
        cx, cy = self.current
        nx, ny = cx + (tx - cx) * SMOOTHING, cy + (ty - cy) * SMOOTHING

        dx, dy = nx - cx, ny - cy
        moved = (dx ** 2 + dy ** 2) ** 0.5
        if moved > MAX_SPEED_PX:
            scale = MAX_SPEED_PX / moved
            nx, ny = cx + dx * scale, cy + dy * scale

        self.current = (nx, ny)
        return (int(nx), int(ny))


def _crop_canvas_size(src_w: int, src_h: int, target_ratio: float) -> Tuple[int, int]:
    """Largest crop window that fits inside the source frame at target_ratio."""
    if target_ratio < src_w / src_h:
        crop_h = src_h
        crop_w = int(crop_h * target_ratio)
    else:
        crop_w = src_w
        crop_h = int(crop_w / target_ratio)
    return max(2, crop_w - (crop_w % 2)), max(2, crop_h - (crop_h % 2))


def _blur_canvas_size(target_ratio: float, base_width: int = 1080) -> Tuple[int, int]:
    """Fixed, platform-recommended canvas for blur-pad mode (1080x1920 for
    9:16) — independent of source resolution, so output is predictable."""
    h = int(round(base_width / target_ratio))
    return base_width, h - (h % 2)


def _compose_crop_frame(frame, canvas_w: int, canvas_h: int, center: Tuple[int, int], src_w: int, src_h: int):
    cx, cy = center
    x0 = max(0, min(src_w - canvas_w, cx - canvas_w // 2))
    y0 = max(0, min(src_h - canvas_h, cy - canvas_h // 2))
    return frame[y0:y0 + canvas_h, x0:x0 + canvas_w]


def _compose_blur_frame(frame, canvas_w: int, canvas_h: int, center: Tuple[int, int]):
    """Background: a zoomed + blurred copy of the FULL frame fills the canvas
    (visible in the top/bottom bars). Foreground: a moderate crop centred on
    the tracked face (FG_ZOOM), fit to the canvas width — tighter than the
    raw frame so the speaker reads clearly, but never as tight as
    --reframe-mode crop, so context doesn't disappear."""
    import cv2  # type: ignore

    src_h, src_w = frame.shape[:2]

    # Background: scale to COVER the whole canvas (edges may spill off), blur it.
    cover_scale = max(canvas_w / src_w, canvas_h / src_h)
    bg_w, bg_h = int(src_w * cover_scale) + 2, int(src_h * cover_scale) + 2
    bg = cv2.resize(frame, (bg_w, bg_h), interpolation=cv2.INTER_LINEAR)
    bx, by = (bg_w - canvas_w) // 2, (bg_h - canvas_h) // 2
    bg = bg[by:by + canvas_h, bx:bx + canvas_w]
    bg = cv2.GaussianBlur(bg, (0, 0), sigmaX=25)
    bg = (bg.astype("float32") * 0.55).astype("uint8")  # darken so the foreground pops

    # Foreground: moderate horizontal crop centred on the tracked face, then
    # fit to canvas width. A narrower crop_w (relative to src_w) fills more
    # of the canvas height once scaled up, i.e. reads as "more zoomed in".
    fg_crop_w = max(1, int(src_w / FG_ZOOM))
    cx, _cy = center
    x0 = max(0, min(src_w - fg_crop_w, cx - fg_crop_w // 2))
    fg_crop = frame[:, x0:x0 + fg_crop_w]

    fit_scale = canvas_w / fg_crop_w
    fg_w, fg_h = canvas_w, int(round(src_h * fit_scale))
    fg = cv2.resize(fg_crop, (fg_w, fg_h), interpolation=cv2.INTER_LINEAR)

    canvas = bg
    y_off = (canvas_h - fg_h) // 2
    if y_off >= 0:
        canvas[y_off:y_off + fg_h, 0:fg_w] = fg
    else:
        # Foreground taller than the canvas (narrow crop + tall source) — centre-crop it.
        crop_y = -y_off
        canvas[:, :] = fg[crop_y:crop_y + canvas_h, 0:fg_w]
    return canvas


def _reframe_vertical(in_path: str, out_path: str, aspect_ratio: str, reframe_mode: str = "blur") -> str:
    """Reframe the cut clip to the target aspect ratio, tracking faces."""
    try:
        import cv2  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "opencv-contrib-python is required for --mode local. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e
    if reframe_mode not in ("blur", "crop"):
        raise ValueError(f"reframe_mode must be 'blur' or 'crop', got {reframe_mode!r}")

    target_ratio = _ratio(aspect_ratio)
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {in_path}")

    try:
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        detector = _load_face_detector((src_w, src_h))
        use_face_tracking = detector is not None
        if not use_face_tracking:
            print("[clip] center-crop mode (no face tracking)", flush=True)
        if reframe_mode == "blur":
            canvas_w, canvas_h = _blur_canvas_size(target_ratio)
        else:
            canvas_w, canvas_h = _crop_canvas_size(src_w, src_h, target_ratio)

        silent_path = out_path + ".silent.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(silent_path, fourcc, fps, (canvas_w, canvas_h))
        tracker = _FaceTracker(default_center=(src_w // 2, src_h // 2))

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if use_face_tracking:
                    center = tracker.update(_detect_face_center(detector, frame))
                else:
                    center = (src_w // 2, src_h // 2)
                if reframe_mode == "blur":
                    out_frame = _compose_blur_frame(frame, canvas_w, canvas_h, center)
                else:
                    out_frame = _compose_crop_frame(frame, canvas_w, canvas_h, center, src_w, src_h)
                writer.write(out_frame)
        finally:
            writer.release()
    finally:
        cap.release()

    # Mux audio from the cut clip back onto the silent reframed video.
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", silent_path,
        "-i", in_path,
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "128k",
        "-map", "0:v:0", "-map", "1:a:0?",
        "-shortest",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    _safe_remove(silent_path)
    return out_path


def _reframe_ffmpeg(in_path: str, out_path: str, target_ratio: float) -> str:
    """Fast reframe using pure ffmpeg (no OpenCV). Scales to fill a
    1080-wide vertical canvas with black padding bars. Sub-second per clip."""
    t_h = int(round(1080 / target_ratio))
    t_h = t_h if t_h % 2 == 0 else t_h + 1
    scale_h = f"scale=1080:-2"
    pad = f"pad=1080:{t_h}:(ow-iw)/2:(oh-ih)/2:color=black"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", in_path,
        "-vf", f"{scale_h},{pad}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-c:a", "aac", "-b:a", "128k",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    return out_path


def crop_clip_local(
    source_path: str,
    start_time: float,
    end_time: float,
    aspect_ratio: str,
    out_path: str,
    reframe_mode: str = "blur",
) -> str:
    """Cut + reframe one highlight, returning the local mp4 path.

    reframe_mode:
      "blur" (default) — OpenCV face-aware blur padding (slow but polished).
      "crop" — OpenCV face-tracked crop (slow).
      "ffmpeg" — Pure ffmpeg scale+pad (fast, sub-second per clip).
    """
    cut_path = out_path + ".cut.mp4"
    try:
        _cut_subclip(source_path, start_time, end_time, cut_path)
        if reframe_mode == "ffmpeg":
            target_ratio = _ratio(aspect_ratio) if isinstance(aspect_ratio, str) else aspect_ratio
            _reframe_ffmpeg(cut_path, out_path, target_ratio)
        else:
            _reframe_vertical(cut_path, out_path, aspect_ratio, reframe_mode=reframe_mode)
    finally:
        _safe_remove(cut_path)
    return out_path


def crop_highlights_local(
    source_path: str,
    highlights: List[Dict],
    aspect_ratio: str = "9:16",
    out_dir: Optional[str] = None,
    reframe_mode: str = "blur",
) -> List[Dict]:
    out_dir = out_dir or LOCAL_OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    results: List[Dict] = []
    for i, h in enumerate(highlights, 1):
        out_path = os.path.join(out_dir, f"short_{i:02d}.mp4")
        print(f"[clip/local] {i}/{len(highlights)}: {h.get('title', '(untitled)')}", flush=True)
        try:
            crop_clip_local(
                source_path,
                float(h["start_time"]),
                float(h["end_time"]),
                aspect_ratio,
                out_path,
                reframe_mode=reframe_mode,
            )
            results.append({**h, "clip_url": out_path})
        except Exception as e:
            print(f"[clip/local] {i} failed: {e}", flush=True)
            results.append({**h, "clip_url": None, "error": str(e)})
    return results
