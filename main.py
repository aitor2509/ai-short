"""CLI entry point.

Usage:
    python main.py "https://www.youtube.com/watch?v=..." \
        --num-clips 3 --aspect-ratio 9:16
"""
import argparse
import json
import sys

# Windows uses 'charmap' by default, which can't encode Unicode characters
# like →. Reconfigure stdout/stderr to UTF-8 so output works on all platforms.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from shorts_generator import generate_shorts


def main() -> int:
    parser = argparse.ArgumentParser(description="AI YouTube Shorts Generator")
    parser.add_argument("url", help="YouTube URL, file:// URL, or local file path")
    parser.add_argument(
        "--mode",
        choices=["api", "local", "editor"],
        default="api",
        help="api (default, MuAPI), local (self-hosted), or editor (optimized bulk generation).",
    )
    parser.add_argument("--num-clips", type=int, default=3, help="How many shorts to render (default: 3)")
    parser.add_argument("--aspect-ratio", default="9:16", help="Output aspect ratio (default: 9:16)")
    parser.add_argument("--format", default="720", help="Source download resolution: 360 / 480 / 720 / 1080 (default: 720)")
    parser.add_argument("--language", default=None, help="Force Whisper language code, e.g. 'en' (default: auto-detect)")
    parser.add_argument("--output-json", default=None, help="Write the full result JSON to this path")
    parser.add_argument(
        "--playlist-scan-limit",
        type=int,
        default=50,
        help="Max videos to inspect when 'url' is a playlist, via Trend Finder (default: 50)",
    )
    parser.add_argument(
        "--upload-drive",
        action="store_true",
        help="Upload rendered shorts to Google Drive and delete the local video files afterwards",
    )
    parser.add_argument(
        "--reframe-mode",
        choices=["blur", "crop"],
        default="blur",
        help="local mode only: 'blur' pads with a blurred background (no cropping, default) "
        "or 'crop' slides a crop window on the tracked face",
    )
    parser.add_argument("--min-duration", type=float, default=15.0, help="Minimum clip length in seconds (default: 15)")
    parser.add_argument("--max-duration", type=float, default=90.0, help="Maximum clip length in seconds (default: 90)")
    args = parser.parse_args()

    from shorts_generator.trend_finder import find_best_video_url, is_playlist_url

    source_url = args.url
    if is_playlist_url(source_url):
        print(f"[main] '{source_url}' looks like a playlist — running Trend Finder...", flush=True)
        try:
            source_url = find_best_video_url(source_url, max_videos_to_scan=args.playlist_scan_limit)
        except Exception as e:
            print(f"\nFAILED (Trend Finder): {e}", file=sys.stderr)
            return 1
        print(f"[main] highest-traction video selected: {source_url}", flush=True)

    try:
        result = generate_shorts(
            youtube_url=source_url,
            num_clips=args.num_clips,
            aspect_ratio=args.aspect_ratio,
            download_format=args.format,
            language=args.language,
            mode=args.mode,
            reframe_mode=args.reframe_mode,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
        )
    except Exception as e:
        print(f"\nFAILED: {e}", file=sys.stderr)
        return 1

    if args.upload_drive:
        import asyncio

        from shorts_generator.config import GDRIVE_FOLDER_ID, GDRIVE_SERVICE_ACCOUNT_FILE
        from shorts_generator.drive_uploader import upload_shorts_and_cleanup

        print("\n[main] uploading shorts to Google Drive...", flush=True)
        try:
            result["shorts"] = asyncio.run(upload_shorts_and_cleanup(
                result["shorts"],
                service_account_file=GDRIVE_SERVICE_ACCOUNT_FILE,
                folder_id=GDRIVE_FOLDER_ID,
                source_video_path=result.get("source_video_url") if result.get("mode") == "local" else None,
            ))
        except Exception as e:
            print(f"[main] Drive upload failed, local files were kept on disk: {e}", file=sys.stderr)

    print("\n" + "=" * 72)
    print(f"Mode:          {result.get('mode', args.mode)}")
    print(f"Source video:  {result['source_video_url']}")
    print(f"Highlights:    {len(result['highlights'])} candidates → kept top {len(result['shorts'])}")
    print("=" * 72)
    for i, s in enumerate(result["shorts"], 1):
        print(f"\n#{i}  score={s.get('score')}  {s.get('start_time'):.1f}s → {s.get('end_time'):.1f}s")
        print(f"     title:  {s.get('title')}")
        print(f"     hook:   {s.get('hook_sentence')}")
        if s.get("drive_url"):
            print(f"     drive:  {s['drive_url']}")
        elif s.get("clip_url"):
            print(f"     clip:   {s['clip_url']}")
        else:
            print(f"     clip:   FAILED ({s.get('error')})")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nFull JSON written to {args.output_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
