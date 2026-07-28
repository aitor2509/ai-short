"""Module 1 — Trend Finder.

Scans a YouTube playlist and selects the video(s) with the highest recent
traction, ready to feed into generate_shorts() unchanged.

Uses yt-dlp only (already a dependency of --mode local — no API key, no
quota to manage). Two-pass strategy:
  1. Flat playlist listing (fast, one request for the whole playlist).
  2. Full metadata per video, fetched in parallel (views/likes/comments/date).

Traction score:
  velocity   = views / days_since_upload       -> how fast it's growing right now
  engagement = (likes + comments) / views       -> how much it resonates, not just reach
  score      = log10(velocity + 1) * (1 + engagement * ENGAGEMENT_WEIGHT)

log10 keeps a video with 10x the views from dominating the ranking 10x over —
without it, an old video with a huge total view count always wins even with
zero recent momentum, which defeats the point of a *trend* finder.
"""
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional
from urllib.parse import parse_qs, urlparse

ENGAGEMENT_WEIGHT = 5.0  # how much engagement rate matters relative to view velocity


def _import_ytdlp():
    try:
        import yt_dlp  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "yt-dlp is required for the Trend Finder. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e
    return yt_dlp


def is_playlist_url(url: str) -> bool:
    """True only when the URL points at an entire playlist.

    A single-video URL copied while watching inside a playlist usually
    contains *both* ?v=... *and* &list=... (e.g. watch?v=abc123&list=PLxyz).
    In that case the user wants THAT video, not the whole playlist — so a
    present 'v' (or /shorts/, /embed/) always wins over 'list'.
    """
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    has_list = "list" in qs
    has_single_video = "v" in qs or "/shorts/" in parsed.path or "/embed/" in parsed.path
    is_playlist_path = parsed.path.rstrip("/").endswith("/playlist")
    return is_playlist_path or (has_list and not has_single_video)


@dataclass
class VideoMetrics:
    id: str
    url: str
    title: str
    view_count: int
    like_count: int
    comment_count: int
    duration: int
    upload_date: Optional[datetime]
    score: float = 0.0

    @property
    def age_days(self) -> float:
        if not self.upload_date:
            return 3650.0  # no date -> treat as "old", pulls score down
        delta = datetime.now(timezone.utc) - self.upload_date
        return max(delta.total_seconds() / 86400.0, 0.25)  # floor at 6h, avoids div-by-~0


def _extract_playlist_entries(playlist_url: str, yt_dlp) -> list:
    """Fast listing of the playlist without pulling full metadata per video."""
    ydl_opts = {
        "extract_flat": "in_playlist",
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(playlist_url, download=False)

    if info is None:
        raise RuntimeError(f"yt-dlp could not read playlist: {playlist_url}")
    entries = info.get("entries") or []
    if not entries:
        raise RuntimeError(f"Playlist has no videos or is not a valid playlist: {playlist_url}")
    return entries


def _fetch_full_metadata(video_url: str, yt_dlp) -> Optional[VideoMetrics]:
    """Second pass: one request per video to get views/likes/comments/date."""
    ydl_opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
    except Exception as e:
        print(f"[trend_finder]   skip (metadata error) {video_url}: {e}", flush=True)
        return None

    upload_date_str = info.get("upload_date")  # YYYYMMDD
    upload_date = None
    if upload_date_str:
        try:
            upload_date = datetime.strptime(upload_date_str, "%Y%m%d").replace(tzinfo=timezone.utc)
        except ValueError:
            pass

    return VideoMetrics(
        id=info.get("id", ""),
        url=info.get("webpage_url") or video_url,
        title=info.get("title", "(untitled)"),
        view_count=int(info.get("view_count") or 0),
        like_count=int(info.get("like_count") or 0),
        comment_count=int(info.get("comment_count") or 0),
        duration=int(info.get("duration") or 0),
        upload_date=upload_date,
    )


def _score(video: VideoMetrics) -> float:
    views = max(video.view_count, 1)
    velocity = views / video.age_days
    engagement = (video.like_count + video.comment_count) / views
    return math.log10(velocity + 1) * (1 + engagement * ENGAGEMENT_WEIGHT)


def rank_playlist(
    playlist_url: str,
    min_duration_seconds: int = 90,
    max_videos_to_scan: int = 50,
    max_workers: int = 6,
) -> List[VideoMetrics]:
    """Scan the playlist and return videos sorted by traction score, desc.

    max_videos_to_scan caps how many playlist entries get full metadata
    (one HTTP request each) — matters on long playlists, where scanning
    everything can get slow and risks YouTube rate-limiting.
    min_duration_seconds filters out Shorts/clips already sitting in the
    playlist: they're not useful source material for cutting 30-60s
    highlights out of.
    """
    yt_dlp = _import_ytdlp()
    entries = _extract_playlist_entries(playlist_url, yt_dlp)[:max_videos_to_scan]

    urls = []
    for e in entries:
        vid = e.get("id")
        url = e.get("url") or (f"https://www.youtube.com/watch?v={vid}" if vid else None)
        if url:
            urls.append(url)

    print(f"[trend_finder] scanning {len(urls)} videos from the playlist...", flush=True)

    results: List[VideoMetrics] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_full_metadata, u, yt_dlp): u for u in urls}
        for future in as_completed(futures):
            metrics = future.result()
            if metrics is None:
                continue
            if metrics.duration < min_duration_seconds:
                continue
            metrics.score = _score(metrics)
            results.append(metrics)

    if not results:
        raise RuntimeError(
            "No playlist video passed the filters (minimum duration / available metadata)."
        )

    results.sort(key=lambda v: v.score, reverse=True)
    print(f"[trend_finder] ranked {len(results)} candidates, top 10:", flush=True)
    for i, v in enumerate(results[:10], 1):
        print(
            f"[trend_finder]   #{i} score={v.score:.3f} views={v.view_count:,} "
            f"({v.age_days:.1f}d old) - {v.title[:70]}",
            flush=True,
        )
    return results


def get_best_videos(playlist_url: str, top_n: int = 1, **kwargs) -> List[VideoMetrics]:
    """Return the top_n highest-traction videos in the playlist."""
    ranked = rank_playlist(playlist_url, **kwargs)
    return ranked[:top_n]


def find_best_video_url(playlist_url: str, **kwargs) -> str:
    """Shortcut: return just the #1 video's URL, ready for generate_shorts()."""
    best = get_best_videos(playlist_url, top_n=1, **kwargs)[0]
    print(f"[trend_finder] winner: {best.title} ({best.url})", flush=True)
    return best.url


if __name__ == "__main__":
    import sys

    playlist = sys.argv[1] if len(sys.argv) > 1 else input("Playlist URL: ").strip()
    top = get_best_videos(playlist, top_n=5)
    print("\nTop videos by traction:")
    for i, v in enumerate(top, 1):
        print(f"{i}. [{v.score:.2f}] {v.title} - {v.view_count:,} views - {v.url}")
