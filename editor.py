"""Editor CLI — bulk short-form video generator.

Usage:
    python editor.py "https://youtu.be/..." --count 50

    Or batch from a list:
    python editor.py --batch urls.txt --count 50

    In GitHub Actions (auto-detected):
    python editor.py "https://youtu.be/..." --count 50 --output /tmp/clips
"""
import argparse
import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from shorts_generator import generate_shorts


def main() -> int:
    parser = argparse.ArgumentParser(description="AI Shorts Editor — bulk clip generator")
    parser.add_argument("url", nargs="?", help="YouTube URL to process")
    parser.add_argument("--count", type=int, default=50, help="Number of shorts to generate (default: 50)")
    parser.add_argument("--output", default=None, help="Output directory for clips")
    parser.add_argument("--batch", default=None, help="File with one URL per line (overrides positional url)")
    parser.add_argument("--aspect-ratio", default="9:16", help="Output aspect ratio (default: 9:16)")
    parser.add_argument("--format", default="720", help="Source resolution: 360/480/720/1080")
    parser.add_argument("--reframe", choices=["ffmpeg", "blur", "crop"], default="ffmpeg", help="Clipping mode: ffmpeg (fast, default), blur (face-aware, slow), crop")
    parser.add_argument("--min-duration", type=float, default=15.0)
    parser.add_argument("--max-duration", type=float, default=90.0)
    parser.add_argument("--json", default=None, help="Write full result JSON to this path")
    args = parser.parse_args()

    urls = []
    if args.batch:
        with open(args.batch) as f:
            urls = [line.strip() for line in f if line.strip()]
    elif args.url:
        urls = [args.url]
    else:
        print("Provide a URL or --batch file")
        return 1

    for i, url in enumerate(urls):
        print(f"\n{'='*60}")
        print(f"Video {i + 1}/{len(urls)}: {url}")
        print(f"{'='*60}")

        try:
            result = generate_shorts(
                youtube_url=url,
                num_clips=args.count,
                aspect_ratio=args.aspect_ratio,
                download_format=args.format,
                mode="editor",
                reframe_mode=args.reframe,
                min_duration=args.min_duration,
                max_duration=args.max_duration,
                out_dir=args.output,
            )
        except Exception as e:
            print(f"\nFAILED: {e}", file=sys.stderr)
            continue

        print(f"\nGenerated {len(result['shorts'])} shorts from {url}")
        for j, s in enumerate(result["shorts"], 1):
            status = s.get("clip_url", "FAILED")
            print(f"  #{j}: {s.get('title','?')} ({s.get('start_time',0):.0f}s-{s.get('end_time',0):.0f}s) → {status}")

        if args.json:
            mode = "a" if i > 0 else "w"
            with open(args.json, mode) as f:
                json.dump(result, f, indent=2)
                f.write("\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
