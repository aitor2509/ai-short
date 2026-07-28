"""End-to-end orchestrator.

Three modes:
  * mode="api"   (default) — MuAPI does download / transcribe / LLM / autocrop.
                              Fast, no local deps, pay-per-call.
  * mode="local"            — yt-dlp + faster-whisper + OpenAI or Gemini + ffmpeg/opencv.
                              Self-hosted, LLM_PROVIDER selects OpenAI or Gemini.
  * mode="editor"           — Optimized for bulk clip generation. Downloads, runs signals
                              FIRST (before transcription), transcribes ONLY candidate
                              windows, then LLM + clip. ~10 min/video instead of ~67 min.
"""
import os
import json
from typing import Dict, List, Optional

from .clipper import crop_highlights
from .downloader import download_youtube
from .highlights import call_muapi_llm, get_highlights
from .transcriber import transcribe


def _run_local(
    youtube_url: str,
    num_clips: int,
    aspect_ratio: str,
    download_format: str,
    language: Optional[str],
    reframe_mode: str = "blur",
    min_duration: float = 15.0,
    max_duration: float = 90.0,
) -> Dict:
    from .local.clipper import crop_highlights_local
    from .local.downloader import download_youtube_local
    from .local.llm import call_local_llm
    from .local.transcriber import transcribe_local

    source_path = download_youtube_local(youtube_url, fmt=download_format)

    transcript = transcribe_local(source_path, language=language)
    if not transcript["segments"]:
        raise RuntimeError(
            "Whisper produced no segments. The video may have no detectable speech."
        )

    candidate_windows = None
    try:
        from .local.downloader import _extract_youtube_video_id
        from .local.signals import get_candidate_windows

        video_id = _extract_youtube_video_id(youtube_url)
        if video_id:
            candidate_windows = get_candidate_windows(
                video_id, source_path, duration=transcript.get("duration", 0)
            )
    except Exception as e:
        # Pre-filter is purely an optimization — if it breaks for any reason
        # (network, parsing, missing deps), fall back to the full scan below
        # instead of failing the whole run over a cost-saving step.
        print(f"[pipeline/local] pre-filter signals unavailable, doing a full scan instead: {e}", flush=True)

    highlights_result = get_highlights(
        transcript, num_clips=num_clips, llm_fn=call_local_llm,
        min_duration=min_duration, max_duration=max_duration,
        candidate_windows=candidate_windows,
    )
    all_highlights: List[Dict] = highlights_result.get("highlights", [])
    if not all_highlights:
        raise RuntimeError("Highlight generator returned zero clips.")

    top = sorted(all_highlights, key=lambda h: int(h.get("score", 0)), reverse=True)[:num_clips]
    print(f"[pipeline/local] cropping {len(top)} of {len(all_highlights)} candidates", flush=True)

    shorts = crop_highlights_local(source_path, top, aspect_ratio=aspect_ratio, reframe_mode=reframe_mode)

    return {
        "mode": "local",
        "source_video_url": source_path,
        "transcript": transcript,
        "highlights": all_highlights,
        "shorts": shorts,
    }


def _run_api(
    youtube_url: str,
    num_clips: int,
    aspect_ratio: str,
    download_format: str,
    language: Optional[str],
    min_duration: float = 15.0,
    max_duration: float = 90.0,
) -> Dict:
    source_url = download_youtube(youtube_url, fmt=download_format)

    transcript = transcribe(source_url, language=language)
    if not transcript["segments"]:
        raise RuntimeError(
            "Whisper produced no segments. The video may have no detectable speech."
        )

    highlights_result = get_highlights(
        transcript, num_clips=num_clips, llm_fn=call_muapi_llm,
        min_duration=min_duration, max_duration=max_duration,
    )
    all_highlights: List[Dict] = highlights_result.get("highlights", [])
    if not all_highlights:
        raise RuntimeError("Highlight generator returned zero clips.")

    top = sorted(all_highlights, key=lambda h: int(h.get("score", 0)), reverse=True)[:num_clips]
    print(f"[pipeline] cropping {len(top)} of {len(all_highlights)} candidates", flush=True)

    shorts = crop_highlights(source_url, top, aspect_ratio=aspect_ratio)

    return {
        "mode": "api",
        "source_video_url": source_url,
        "transcript": transcript,
        "highlights": all_highlights,
        "shorts": shorts,
    }


def _run_editor(
    youtube_url: str,
    num_clips: int = 50,
    aspect_ratio: str = "9:16",
    download_format: str = "720",
    language: Optional[str] = None,
    reframe_mode: str = "ffmpeg",
    min_duration: float = 15.0,
    max_duration: float = 90.0,
    out_dir: Optional[str] = None,
) -> Dict:
    """Editor mode: optimized bulk clip generation.

    Flow:
      1. Download video (yt-dlp)
      2. Run pre-filter signals FIRST (heatmap, audio energy, scene changes,
         motion, comments) — NO transcription needed for signals, they're fast
      3. Extract + transcribe ONLY the candidate windows (~25 min audio instead
         of 2h = ~2-3 min whisper CPU instead of ~45 min)
      4. Send candidate windows + their transcript to Gemini LLM (1 call)
      5. Clip all winners

    Returns the same shape as _run_local / _run_api.
    """
    from .local.clipper import crop_highlights_local
    from .local.downloader import download_youtube_local, _extract_youtube_video_id
    from .local.llm import call_local_llm
    from .local.signals import get_candidate_windows
    from .local.transcriber import transcribe_windows
    from .config import LOCAL_OUTPUT_DIR

    source_path = download_youtube_local(youtube_url, fmt=download_format)
    out_dir = out_dir or LOCAL_OUTPUT_DIR

    video_id = _extract_youtube_video_id(youtube_url)

    import subprocess
    duration = 0.0
    if video_id:
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                 "-of", "csv=p=0", source_path],
                capture_output=True, text=True, check=True,
            )
            duration = float(result.stdout.strip())
        except Exception:
            pass

    candidate_windows = get_candidate_windows(
        video_id, source_path, duration=duration,
        top_n=max(num_clips * 2, 50),
    )
    if not candidate_windows:
        print("[pipeline/editor] no candidate windows from signals, transcribing full video", flush=True)
        from .local.transcriber import transcribe_local
        transcript = transcribe_local(source_path, language=language)
    else:
        print(
            f"[pipeline/editor] signals found {len(candidate_windows)} candidate windows, "
            f"transcribing ONLY those segments...",
            flush=True,
        )
        transcript = transcribe_windows(source_path, candidate_windows, language=language)

    if not transcript["segments"]:
        raise RuntimeError("No transcript segments produced.")

    highlights_result = get_highlights(
        transcript, num_clips=num_clips, llm_fn=call_local_llm,
        min_duration=min_duration, max_duration=max_duration,
        candidate_windows=candidate_windows if candidate_windows else None,
    )
    all_highlights: List[Dict] = highlights_result.get("highlights", [])
    if not all_highlights:
        raise RuntimeError("Highlight generator returned zero clips.")

    top = sorted(all_highlights, key=lambda h: int(h.get("score", 0)), reverse=True)[:num_clips]
    print(f"[pipeline/editor] cropping {len(top)} of {len(all_highlights)} candidates", flush=True)

    shorts = crop_highlights_local(source_path, top, aspect_ratio=aspect_ratio, reframe_mode=reframe_mode, out_dir=out_dir)

    return {
        "mode": "editor",
        "source_video_url": source_path,
        "transcript": transcript,
        "highlights": all_highlights,
        "shorts": shorts,
    }


def generate_shorts(
    youtube_url: str,
    num_clips: int = 3,
    aspect_ratio: str = "9:16",
    download_format: str = "720",
    language: Optional[str] = None,
    mode: str = "api",
    reframe_mode: str = "blur",
    min_duration: float = 15.0,
    max_duration: float = 90.0,
    out_dir: Optional[str] = None,
) -> Dict:
    """Run the full pipeline and return a structured result.

    Args:
        youtube_url: source URL.
        num_clips: how many shorts to render.
        aspect_ratio: e.g. "9:16", "1:1".
        download_format: source resolution ("360" / "480" / "720" / "1080").
        language: ISO-639-1 to force Whisper language detection.
        mode: "api" (default, MuAPI), "local" (yt-dlp + faster-whisper +
            OpenAI or Gemini + ffmpeg), or "editor" (optimized bulk:
            signals → transcribe windows → LLM → clip, ~10 min/video).
        reframe_mode: "blur" (default, no cropping — pads with a blurred
            zoomed background) or "crop" (slides a crop window on the
            tracked face). Only applies to mode="local".
        min_duration / max_duration: hard clip-length bounds in seconds
            (default 15-90). The LLM is prompted with this range and the
            result is also clamped/filtered afterwards, so it's a real
            guarantee, not just a suggestion.

    Returns:
        {
          "mode": "api" | "local" | "editor",
          "source_video_url": str,   # hosted URL (api) or local path (local)
          "transcript": {...},
          "highlights": [...],       # all candidates ranked
          "shorts": [...],           # top `num_clips` with clip_url / local path
        }
    """
    mode = (mode or "api").lower()
    if mode == "local":
        return _run_local(
            youtube_url, num_clips, aspect_ratio, download_format, language, reframe_mode,
            min_duration=min_duration, max_duration=max_duration,
        )
    if mode == "api":
        return _run_api(
            youtube_url, num_clips, aspect_ratio, download_format, language,
            min_duration=min_duration, max_duration=max_duration,
        )
    if mode == "editor":
        return _run_editor(
            youtube_url, num_clips=num_clips, aspect_ratio=aspect_ratio,
            download_format=download_format, language=language,
            reframe_mode=reframe_mode, min_duration=min_duration,
            max_duration=max_duration, out_dir=out_dir,
        )
    raise ValueError(f"Unknown mode: {mode!r}. Use 'api', 'local' or 'editor'.")
