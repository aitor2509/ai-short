"""Free, LLM-free signals to pre-filter candidate windows before ever
calling an LLM. Goal: cut a 2-hour video down to 5-10 candidate windows
using zero tokens, then send ONLY those windows' transcript text to the
LLM for the final verdict — instead of the whole video in ~19min chunks.

Three independent signals, each optional and each fails silently (returns
an empty list) rather than raising — losing one signal just means the
final ranking relies more on the others, never crashes the pipeline:

  1. YouTube's "Most Replayed" heatmap — real audience rewatch behavior,
     scraped from the watch page (not an official API). Best signal when
     available, but NOT always available: needs ~50k+ views, and some
     videos don't get one at all. Most fragile of the three — YouTube can
     change its internal page structure at any time.
  2. Audio energy peaks — RMS volume spikes (laughs, reactions, exclamations),
     computed locally with ffmpeg + numpy, no external calls at all.
  3. Comments mentioning a timestamp ("el minuto 12:34...") — via the
     YouTube Data API v3 (API key only, no OAuth), weighted by like count.
"""
import json
import os
import re
import subprocess
import wave
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import YOUTUBE_API_KEY

# ---------------------------------------------------------------------------
# Signal 1 — YouTube "Most Replayed" heatmap (scraped, unofficial)
# ---------------------------------------------------------------------------

_YT_INITIAL_DATA_MARKERS = ("var ytInitialData = ", 'ytInitialData"] = ')


def _extract_balanced_json(text: str, start_idx: int) -> Optional[str]:
    """text[start_idx] must be '{'. Walks forward counting brace depth
    (respecting quoted strings) to find the matching closing brace — more
    robust than a regex when the JSON itself contains '};' substrings,
    which a naive greedy/lazy regex can truncate on."""
    if start_idx >= len(text) or text[start_idx] != "{":
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start_idx, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start_idx:i + 1]
    return None


def _find_heatmap_markers(node) -> List[Dict]:
    """Recursively search the parsed ytInitialData for heatmap marker
    dicts. Deliberately NOT hardcoded to one exact nested path
    (frameworkUpdates -> entityBatchUpdate -> ... -> macroMarkersListEntity
    -> markersList -> markers) because YouTube's exact structure shifts
    between A/B test groups and app versions — a recursive search for
    "any dict that looks like a heatmap marker" survives those shifts
    better than a brittle fixed path.
    """
    found: List[Dict] = []
    if isinstance(node, dict):
        has_intensity = "intensityScoreNormalized" in node
        start_key = "startMillis" if "startMillis" in node else (
            "visibleTimeRangeStartMillis" if "visibleTimeRangeStartMillis" in node else None
        )
        if has_intensity and start_key:
            found.append(node)
        else:
            for v in node.values():
                found.extend(_find_heatmap_markers(v))
    elif isinstance(node, list):
        for item in node:
            found.extend(_find_heatmap_markers(item))
    return found


def get_most_replayed_heatmap(video_id: str, timeout: float = 15.0) -> List[Tuple[float, float, float]]:
    """Returns [(start_sec, end_sec, intensity_0_to_1), ...] sorted by time,
    or [] if unavailable/blocked/changed structure — never raises."""
    try:
        import urllib.request

        req = urllib.request.Request(
            f"https://www.youtube.com/watch?v={video_id}",
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            html = resp.read().decode("utf-8", errors="ignore")

        marker_pos = None
        for marker in _YT_INITIAL_DATA_MARKERS:
            idx = html.find(marker)
            if idx != -1:
                marker_pos = idx + len(marker)
                break
        if marker_pos is None:
            print("[signals] heatmap: ytInitialData not found on page (layout may have changed)", flush=True)
            return []

        blob = _extract_balanced_json(html, marker_pos)
        if not blob:
            print("[signals] heatmap: could not extract a balanced JSON block", flush=True)
            return []

        data = json.loads(blob)
        markers = _find_heatmap_markers(data)
        if not markers:
            print("[signals] heatmap: no heatmap on this video (needs ~50k+ views, or none generated)", flush=True)
            return []

        out = []
        for m in markers:
            start_ms = m.get("startMillis") or m.get("visibleTimeRangeStartMillis")
            duration_ms = m.get("durationMillis") or (
                (m.get("visibleTimeRangeEndMillis", 0) - m.get("visibleTimeRangeStartMillis", 0))
                if "visibleTimeRangeEndMillis" in m else None
            )
            intensity = m.get("intensityScoreNormalized")
            if start_ms is None or duration_ms is None or intensity is None:
                continue
            start_sec = float(start_ms) / 1000.0
            out.append((start_sec, start_sec + float(duration_ms) / 1000.0, float(intensity)))

        out.sort(key=lambda t: t[0])
        print(f"[signals] heatmap: found {len(out)} segments", flush=True)
        return out
    except Exception as e:
        print(f"[signals] heatmap unavailable, skipping this signal: {e}", flush=True)
        return []


# ---------------------------------------------------------------------------
# Signal 2 — Audio energy peaks (fully local, no network at all)
# ---------------------------------------------------------------------------

def get_audio_energy_peaks(
    audio_or_video_path: str,
    window_seconds: float = 5.0,
    top_n: int = 12,
    min_gap_seconds: float = 30.0,
) -> List[Tuple[float, float, float]]:
    """Returns the top_n loudest windows as [(start_sec, end_sec, rms_norm_0_1), ...],
    spaced at least min_gap_seconds apart (non-max suppression — otherwise
    the top N could all cluster inside one loud burst). [] on any failure."""
    wav_path = audio_or_video_path + ".__energy_tmp.wav"
    try:
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", audio_or_video_path,
            "-vn", "-ac", "1", "-ar", "16000",
            wav_path,
        ]
        subprocess.run(cmd, check=True)

        with wave.open(wav_path, "rb") as wf:
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            samples = np.frombuffer(wf.readframes(n_frames), dtype=np.int16).astype(np.float32)

        window_size = int(window_seconds * sample_rate)
        if window_size <= 0 or len(samples) < window_size:
            return []

        n_windows = len(samples) // window_size
        rms = np.array([
            np.sqrt(np.mean(np.square(samples[i * window_size:(i + 1) * window_size])) + 1e-9)
            for i in range(n_windows)
        ])
        max_rms = float(rms.max()) if len(rms) else 0.0
        if max_rms <= 0:
            return []
        rms_norm = rms / max_rms

        order = np.argsort(rms_norm)[::-1]  # loudest windows first
        picked: List[Tuple[float, float, float]] = []
        for idx in order:
            start = idx * window_seconds
            if any(abs(start - p[0]) < min_gap_seconds for p in picked):
                continue
            picked.append((start, start + window_seconds, float(rms_norm[idx])))
            if len(picked) >= top_n:
                break

        picked.sort(key=lambda t: t[0])
        print(f"[signals] audio energy: {len(picked)} peaks found", flush=True)
        return picked
    except Exception as e:
        print(f"[signals] audio energy analysis failed, skipping this signal: {e}", flush=True)
        return []
    finally:
        if os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Signal 3 — Comments mentioning a timestamp (YouTube Data API v3, API key only)
# ---------------------------------------------------------------------------

_TIMESTAMP_RE = re.compile(r"\b(\d{1,2}:\d{2}(?::\d{2})?)\b")


def _timestamp_to_seconds(ts: str) -> Optional[float]:
    parts = ts.split(":")
    try:
        parts = [int(p) for p in parts]
    except ValueError:
        return None
    if len(parts) == 2:
        return float(parts[0] * 60 + parts[1])
    if len(parts) == 3:
        return float(parts[0] * 3600 + parts[1] * 60 + parts[2])
    return None


def get_comment_timestamps(
    video_id: str,
    max_comments: int = 100,
    top_n: int = 10,
    window_seconds: float = 20.0,
) -> List[Tuple[float, float, float]]:
    """Returns [(start_sec, end_sec, weight_0_1), ...] built from timestamp
    mentions in top comments, weighted by like count. [] if comments are
    disabled, no API key configured, or no mentions found — never raises."""
    if not YOUTUBE_API_KEY:
        print("[signals] comments: YOUTUBE_API_KEY not set, skipping this signal", flush=True)
        return []
    try:
        import urllib.parse
        import urllib.request

        url = "https://www.googleapis.com/youtube/v3/commentThreads?" + urllib.parse.urlencode({
            "part": "snippet",
            "videoId": video_id,
            "order": "relevance",
            "maxResults": min(max_comments, 100),
            "textFormat": "plainText",
            "key": YOUTUBE_API_KEY,
        })
        with urllib.request.urlopen(url, timeout=15.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        weighted_times: List[Tuple[float, float]] = []  # (seconds, like_count_weight)
        for item in data.get("items", []):
            snippet = item.get("snippet", {}).get("topLevelComment", {}).get("snippet", {})
            text = snippet.get("textDisplay", "") or ""
            likes = int(snippet.get("likeCount", 0) or 0)
            for match in _TIMESTAMP_RE.findall(text):
                seconds = _timestamp_to_seconds(match)
                if seconds is not None:
                    weighted_times.append((seconds, 1.0 + likes))

        if not weighted_times:
            print("[signals] comments: no timestamp mentions found", flush=True)
            return []

        # Cluster mentions within window_seconds of each other, summing weight —
        # several people mentioning "12:34" and "12:40" should count as one strong signal.
        weighted_times.sort(key=lambda t: t[0])
        clusters: List[List[float]] = []  # each: [center_time, total_weight]
        for t, w in weighted_times:
            if clusters and t - clusters[-1][0] <= window_seconds:
                clusters[-1][0] = (clusters[-1][0] + t) / 2  # drift center slightly
                clusters[-1][1] += w
            else:
                clusters.append([t, w])

        clusters.sort(key=lambda c: c[1], reverse=True)
        top = clusters[:top_n]
        max_w = max(c[1] for c in top) if top else 1.0

        out = [
            (max(0.0, c[0] - window_seconds / 2), c[0] + window_seconds / 2, c[1] / max_w)
            for c in top
        ]
        out.sort(key=lambda t: t[0])
        print(f"[signals] comments: {len(out)} timestamp clusters found", flush=True)
        return out
    except Exception as e:
        print(f"[signals] comment timestamps unavailable, skipping this signal: {e}", flush=True)
        return []


# ---------------------------------------------------------------------------
# Signal 4 — Scene change detection (ffmpeg, fast, free)
# ---------------------------------------------------------------------------

def get_scene_changes(
    video_path: str,
    threshold: float = 0.4,
    top_n: int = 15,
    min_gap_seconds: float = 20.0,
) -> List[Tuple[float, float, float]]:
    """Detect scene changes using ffmpeg's scene detection filter.
    Returns [(start, end, score), ...] where score is the scene change
    confidence. Scene changes often mark important transitions (new topic,
    reaction, etc.) that make good highlight boundaries."""
    try:
        cmd = [
            "ffmpeg", "-i", video_path,
            "-filter:v", f"select='gt(scene,{threshold})',showinfo",
            "-f", "null", "-",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        timestamps = []
        for line in result.stderr.splitlines():
            if "pts_time:" in line:
                match = re.search(r"pts_time:([\d.]+)", line)
                if match:
                    t = float(match.group(1))
                    timestamps.append(t)

        if not timestamps:
            return []

        picked = []
        for t in timestamps:
            if any(abs(t - p[0]) < min_gap_seconds for p in picked):
                continue
            picked.append((t, t + 5.0, 0.8))
            if len(picked) >= top_n:
                break

        print(f"[signals] scene changes: {len(picked)} transitions found", flush=True)
        return picked
    except Exception as e:
        print(f"[signals] scene change detection failed, skipping: {e}", flush=True)
        return []


# ---------------------------------------------------------------------------
# Signal 5 — Motion analysis (OpenCV, fast, free)
# ---------------------------------------------------------------------------

def get_motion_peaks(
    video_path: str,
    top_n: int = 10,
    window_seconds: float = 10.0,
    min_gap_seconds: float = 20.0,
) -> List[Tuple[float, float, float]]:
    """Detect high-motion segments using frame differencing.
    High motion = action, gameplay intensity, physical comedy.
    Returns [(start, end, score), ...]."""
    try:
        import cv2
    except ImportError:
        print("[signals] opencv not available, skipping motion signal", flush=True)
        return []

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if total_frames > 0 else 0

        window_frames = int(window_seconds * fps)
        step_frames = max(1, window_frames // 2)

        ret, prev = cap.read()
        if not ret:
            cap.release()
            return []

        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        motion_scores = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            if frame_idx % step_frames != 0:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(prev_gray, gray)
            score = float(np.mean(diff))
            motion_scores.append((frame_idx / fps, score))
            prev_gray = gray

        cap.release()

        if not motion_scores:
            return []

        scores_arr = np.array([s[1] for s in motion_scores])
        if scores_arr.max() <= 0:
            return []
        scores_norm = scores_arr / scores_arr.max()

        order = np.argsort(scores_arr)[::-1]
        picked = []
        for idx in order:
            t = motion_scores[idx][0]
            if any(abs(t - p[0]) < min_gap_seconds for p in picked):
                continue
            picked.append((t, min(t + window_seconds, duration), float(scores_norm[idx])))
            if len(picked) >= top_n:
                break

        picked.sort(key=lambda t: t[0])
        print(f"[signals] motion: {len(picked)} high-motion segments found", flush=True)
        return picked
    except Exception as e:
        print(f"[signals] motion analysis failed, skipping: {e}", flush=True)
        return []


# ---------------------------------------------------------------------------
# Combine all signals into final candidate windows
# ---------------------------------------------------------------------------

def get_candidate_windows(
    video_id: str,
    audio_or_video_path: str,
    duration: float,
    top_n: int = 50,
    pad_before: float = 5.0,
    pad_after: float = 10.0,
    merge_gap: float = 10.0,
) -> List[Dict]:
    """Combines all signals into up to top_n padded candidate windows,
    each with a combined score and which signals confirmed it (a window
    flagged by 2-3+ signals is much more likely to be a real highlight than
    one flagged by just one). Returns [] if every signal failed — the
    caller should fall back to the old "send everything" behavior in that
    case, not treat [] as "no highlights exist".

    Signals included (all free, no LLM):
      1. YouTube "Most Replayed" heatmap — audience rewatch data
      2. Audio energy peaks — laughs, reactions, exclamations
      3. Comment timestamps — community interaction
      4. Scene changes — topic transitions
      5. Motion analysis — action/intensity peaks
    """
    raw: List[Tuple[float, float, float, str]] = []

    for start, end, score in get_most_replayed_heatmap(video_id):
        raw.append((start, end, score * 1.0, "heatmap"))
    for start, end, score in get_audio_energy_peaks(audio_or_video_path, top_n=max(12, top_n // 2)):
        raw.append((start, end, score * 0.7, "audio"))
    for start, end, score in get_comment_timestamps(video_id, top_n=max(10, top_n // 3)):
        raw.append((start, end, score * 0.8, "comments"))
    for start, end, score in get_scene_changes(audio_or_video_path, top_n=max(10, top_n // 3)):
        raw.append((start, end, score * 0.6, "scene"))
    for start, end, score in get_motion_peaks(audio_or_video_path, top_n=max(8, top_n // 4)):
        raw.append((start, end, score * 0.5, "motion"))

    if not raw:
        return []

    raw.sort(key=lambda t: t[0])
    merged: List[Dict] = []
    for start, end, score, source in raw:
        if merged and start - merged[-1]["end_time"] <= merge_gap:
            m = merged[-1]
            m["end_time"] = max(m["end_time"], end)
            m["score"] += score
            if source not in m["sources"]:
                m["sources"].append(source)
        else:
            merged.append({"start_time": start, "end_time": end, "score": score, "sources": [source]})

    for m in merged:
        m["score"] *= (1.0 + 0.5 * (len(m["sources"]) - 1))

    merged.sort(key=lambda m: m["score"], reverse=True)
    top = merged[:top_n]

    for m in top:
        m["start_time"] = max(0.0, m["start_time"] - pad_before)
        m["end_time"] = min(duration, m["end_time"] + pad_after) if duration > 0 else m["end_time"] + pad_after

    top.sort(key=lambda m: m["start_time"])
    print(
        f"[signals] {len(top)} candidate windows after merging "
        f"(from {len(raw)} raw signal hits across heatmap/audio/comments/scene/motion)",
        flush=True,
    )
    return top
