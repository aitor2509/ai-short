"""Find the most viral-worthy highlights in a transcript.

Logic ported from ViralVadoo's transcript_analysis/highlight_generator.py:
  - content-type / density detection
  - chunking for long videos with overlap
  - virality-criteria prompt
  - score-based dedupe with overlap suppression

The LLM call is pluggable via the `llm_fn` argument so the same prompts can
drive either MuAPI (default, --mode api) or a direct local LLM client
(--mode local).
"""
import json
import re
from typing import Callable, Dict, List, Optional

from . import muapi


LLMFn = Callable[[str], str]


CONTENT_TYPE_PROMPT = """Analyze this video transcript sample and classify the content type.
Choose one: podcast, interview, tutorial, lecture, commentary, debate, vlog, other.
Also estimate content density: low (mostly filler/chit-chat), medium, or high (dense info/stories).
Respond with JSON only: {"content_type": "...", "density": "..."}"""


VIRALITY_CRITERIA = """
Virality signals to prioritize (ranked by impact):
1. HOOK MOMENTS — statements that create immediate curiosity ("The secret is...", "Nobody talks about...", "I was completely wrong about...")
2. EMOTIONAL PEAKS — genuine surprise, laughter, anger, vulnerability, excitement; raw unscripted reactions
3. OPINION BOMBS — strong, polarizing or counter-intuitive statements that trigger agree/disagree
4. REVELATION MOMENTS — surprising facts, stats, or confessions that reframe how the viewer thinks
5. CONFLICT/TENSION — disagreement, pushback, or a problem being confronted head-on
6. QUOTABLE ONE-LINERS — a sentence that works as a standalone quote card
7. STORY PEAKS — the climax or twist of an anecdote; the payoff moment
8. PRACTICAL VALUE — a concrete tip, hack, or insight the viewer can immediately apply
"""


HIGHLIGHT_SYSTEM_PROMPT = """You are an elite short-form video editor who has studied thousands of viral clips on TikTok, Instagram Reels, and YouTube Shorts. You know exactly what makes viewers stop scrolling, watch to the end, and share.

{virality_criteria}

Content type: {content_type} | Density: {density}

Your task: identify the most viral-worthy highlights from the transcript.

Rules:
- Every highlight must open with a strong HOOK — a line that grabs attention within the first 3 seconds
- {duration_instruction}
- Never cut mid-sentence or mid-thought — each clip must feel complete and self-contained
- Clips must not overlap significantly with each other
- Score 0-100 on viral potential (not general quality)
- {num_clips_instruction}
- For each highlight, identify the single best "hook_sentence" — the opening line that would make someone stop scrolling
- Explain in one sentence why this clip is viral ("virality_reason")

Respond ONLY with valid JSON (no markdown, no explanation):
{{"highlights":[{{"title":"string","start_time":float,"end_time":float,"score":int,"hook_sentence":"string","virality_reason":"string"}}]}}"""


DEFAULT_MIN_CLIP_SECONDS = 15.0
DEFAULT_MAX_CLIP_SECONDS = 90.0


CHUNK_SIZE_SECONDS = 1200       # 20-min chunks for long videos
LONG_VIDEO_THRESHOLD = 1800     # chunk videos longer than 30 min
CHUNK_OVERLAP_SECONDS = 60
GPT_CALL_TIMEOUT_SECONDS = 300  # cap LLM polls at 5 min — a wedged call should fail fast
MAX_HIGHLIGHT_API_ATTEMPTS = 3


CANDIDATE_MODE_INSTRUCTIONS = """
IMPORTANT — this transcript has already been pre-filtered to {n} candidate zones using real
signals (YouTube audience rewatch data, local audio energy peaks, and/or viewer comments that
mention a timestamp), each marked with a "=== CANDIDATE ZONE" header showing its signal score
and which signals flagged it. Judge each zone on its own merit — turn it into a highlight if
it's genuinely good, or skip it if the signal was a false positive (e.g. a loud noise that
wasn't actually funny or interesting). Stay within the timestamps given inside each zone; do
not invent timestamps outside the provided zones.
"""


def build_prefiltered_transcript_text(transcript: Dict, candidate_windows: List[Dict]) -> str:
    """Extract only the transcript segments inside each candidate window,
    as labeled blocks — used instead of the full transcript when
    signals.get_candidate_windows() found something worth checking (see
    get_highlights). Cuts what gets sent to the LLM from "the whole video"
    down to just these few windows.
    """
    segments = transcript.get("segments", [])
    blocks = []
    for i, w in enumerate(candidate_windows, 1):
        w_start, w_end = w["start_time"], w["end_time"]
        window_segs = [s for s in segments if s["end"] > w_start and s["start"] < w_end]
        if not window_segs:
            continue
        text = "\n".join(f"[{s['start']:.1f}s] {s['text'].strip()}" for s in window_segs)
        sources = ", ".join(w.get("sources", []))
        blocks.append(
            f"=== CANDIDATE ZONE {i} (score={w.get('score', 0):.2f}, signals: {sources}, "
            f"window {w_start:.1f}s-{w_end:.1f}s) ===\n{text}"
        )
    return "\n\n".join(blocks)


def call_muapi_llm(prompt: str) -> str:
    """Default LLM backend: MuAPI gpt-5-mini."""
    result = muapi.run(
        "gpt-5-mini",
        {"prompt": prompt},
        label="gpt-5-mini",
        timeout=GPT_CALL_TIMEOUT_SECONDS,
    )

    outputs = result.get("outputs")
    if isinstance(outputs, list) and outputs and isinstance(outputs[0], str) and outputs[0].strip():
        return outputs[0]

    for key in ("output", "text", "response", "result", "content"):
        v = result.get(key)
        if isinstance(v, str) and v.strip():
            return v
        if isinstance(v, dict):
            inner = v.get("text") or v.get("content")
            if isinstance(inner, str) and inner.strip():
                return inner
        if isinstance(v, list) and v and isinstance(v[0], str):
            return v[0]

    raise RuntimeError(f"Could not extract gpt-5-mini text from response: {result}")


def _strip_trailing_commas(text: str) -> str:
    """Remove a comma immediately before a closing ] or } — a common JSON
    mistake from open-weight models (seen from Groq/Llama in practice,
    rarer from GPT/Gemini). Not needed when the model gets it right, so
    this is purely a repair pass, never a required step."""
    return re.sub(r",\s*([\]}])", r"\1", text)


def _parse_json_loose(raw: str) -> Dict:
    """Best-effort JSON parsing for LLM output: strips markdown fences and
    repairs the most common near-miss mistakes (trailing commas) before
    giving up."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    try:
        return json.loads(_strip_trailing_commas(text))
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        block = text[start:end + 1]
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            return json.loads(_strip_trailing_commas(block))
    raise json.JSONDecodeError("no JSON object found in model output", text, 0)


def _coerce_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _sanitize_highlights(
    raw_highlights: object,
    duration: float,
    min_duration: float = DEFAULT_MIN_CLIP_SECONDS,
    max_duration: float = DEFAULT_MAX_CLIP_SECONDS,
) -> List[Dict]:
    """Normalize model output into the expected shape; skip invalid entries.

    Also hard-enforces [min_duration, max_duration]: the prompt already asks
    for this range, but LLMs don't always comply, so this is the real
    guarantee. Too long -> trim the tail (keep the hook, which is always at
    the start). Too short -> try stretching the tail up to min_duration
    first (bounded by the video's real end); if there's nowhere left to
    stretch into, drop the clip rather than ship something under spec.
    """
    if not isinstance(raw_highlights, list):
        return []

    max_end = duration if duration > 0 else float("inf")
    cleaned: List[Dict] = []
    for item in raw_highlights:
        if not isinstance(item, dict):
            continue

        start = _coerce_float(item.get("start_time"), default=-1.0)
        end = _coerce_float(item.get("end_time"), default=-1.0)
        if start < 0 or end <= start:
            continue

        if max_end != float("inf"):
            start = min(start, max_end)
            end = min(end, max_end)
            if end <= start:
                continue

        if end - start > max_duration:
            end = start + max_duration

        if end - start < min_duration:
            end = min(start + min_duration, max_end)
            if end - start < min_duration:
                continue  # not enough video left to reach the minimum — drop it

        cleaned.append(
            {
                "title": str(item.get("title") or "Untitled Highlight").strip(),
                "start_time": start,
                "end_time": end,
                "score": max(0, min(100, _coerce_int(item.get("score"), default=0))),
                "hook_sentence": str(item.get("hook_sentence") or "").strip(),
                "virality_reason": str(item.get("virality_reason") or "").strip(),
            }
        )

    return cleaned


def detect_content_type(transcript: Dict, llm_fn: LLMFn = call_muapi_llm) -> Dict[str, str]:
    segments = transcript.get("segments", [])
    sample = " ".join(s["text"] for s in segments[:25])[:3000]
    prompt = f"{CONTENT_TYPE_PROMPT}\n\nTranscript sample:\n{sample}"
    try:
        raw = llm_fn(prompt)
        return _parse_json_loose(raw)
    except Exception:
        return {"content_type": "other", "density": "medium"}


def build_transcript_text(transcript: Dict) -> str:
    segments = transcript.get("segments", [])
    return "\n".join(f"[{s['start']:.1f}s] {s['text'].strip()}" for s in segments)


def chunk_transcript(transcript: Dict) -> List[Dict]:
    segments = transcript.get("segments", [])
    duration = transcript.get("duration", segments[-1]["end"] if segments else 0)
    chunks = []
    start = 0
    while start < duration:
        end = min(start + CHUNK_SIZE_SECONDS, duration)
        chunk_segs = [
            s for s in segments
            if s["start"] >= start and s["end"] <= end + CHUNK_OVERLAP_SECONDS
        ]
        if chunk_segs:
            chunk = dict(transcript)
            chunk["segments"] = chunk_segs
            chunk["duration"] = end - start
            chunk["_offset"] = start
            chunks.append(chunk)
        start += CHUNK_SIZE_SECONDS - CHUNK_OVERLAP_SECONDS
    return chunks


def call_highlight_api(
    transcript_text: str,
    content_info: Dict,
    duration: float,
    num_clips: int,
    is_chunk: bool = False,
    llm_fn: LLMFn = call_muapi_llm,
    min_duration: float = DEFAULT_MIN_CLIP_SECONDS,
    max_duration: float = DEFAULT_MAX_CLIP_SECONDS,
    candidate_mode: bool = False,
    num_candidates: int = 0,
) -> Dict:
    # Ask for ~2× the user's target so dedupe has headroom, but cap so the model
    # doesn't have to generate a huge JSON payload (which times out gpt-5-mini).
    target = max(num_clips * 2, 5)
    natural_max = max(2 if is_chunk else 3, int(duration / max(max_duration, 1)))
    min_clips = min(target, natural_max, 8)
    duration_instruction = (
        f"Every clip's duration MUST be between {min_duration:.0f} and {max_duration:.0f} seconds. "
        f"Prefer the middle of that range; only push to the edges when the content genuinely needs it."
    )
    system = HIGHLIGHT_SYSTEM_PROMPT.format(
        virality_criteria=VIRALITY_CRITERIA,
        content_type=content_info.get("content_type", "other"),
        density=content_info.get("density", "medium"),
        duration_instruction=duration_instruction,
        num_clips_instruction=f"Generate at least {min_clips} highlights",
    )
    if candidate_mode:
        system += "\n" + CANDIDATE_MODE_INSTRUCTIONS.format(n=num_candidates)
    base_prompt = f"{system}\n\nTranscript:\n{transcript_text}"
    prompt = base_prompt
    last_error = "unknown"

    for attempt in range(1, MAX_HIGHLIGHT_API_ATTEMPTS + 1):
        raw = llm_fn(prompt)
        try:
            parsed = _parse_json_loose(raw)
            highlights = _sanitize_highlights(
                parsed.get("highlights"), duration=duration, min_duration=min_duration, max_duration=max_duration
            )
            if highlights:
                return {"highlights": highlights}
            last_error = "no valid highlights in response"
        except Exception as e:
            last_error = str(e)

        if attempt < MAX_HIGHLIGHT_API_ATTEMPTS:
            print(
                f"[highlights] invalid model output on attempt {attempt}/{MAX_HIGHLIGHT_API_ATTEMPTS}; retrying",
                flush=True,
            )
            prompt = (
                base_prompt
                + "\n\nIMPORTANT: Return ONLY valid JSON with a top-level 'highlights' array."
                + " Each item must include: title, start_time, end_time, score, hook_sentence, virality_reason."
                + " No markdown fences, no commentary."
            )

    raise RuntimeError(
        f"Highlight generator produced invalid output after {MAX_HIGHLIGHT_API_ATTEMPTS} attempts: {last_error}"
    )


def dedupe_highlights(highlights: List[Dict]) -> List[Dict]:
    """Drop a highlight if it overlaps >50% with a higher-scoring one already kept."""
    highlights = sorted(highlights, key=lambda x: int(x.get("score", 0)), reverse=True)
    kept: List[Dict] = []
    for h in highlights:
        h_start = float(h["start_time"])
        h_end = float(h["end_time"])
        h_dur = h_end - h_start
        overlapping = False
        for k in kept:
            latest_start = max(h_start, float(k["start_time"]))
            earliest_end = min(h_end, float(k["end_time"]))
            overlap = earliest_end - latest_start
            if overlap > 0 and overlap > 0.5 * h_dur:
                overlapping = True
                break
        if not overlapping:
            kept.append(h)
    return kept


def get_highlights(
    transcript: Dict,
    num_clips: int = 3,
    llm_fn: Optional[LLMFn] = None,
    min_duration: float = DEFAULT_MIN_CLIP_SECONDS,
    max_duration: float = DEFAULT_MAX_CLIP_SECONDS,
    candidate_windows: Optional[List[Dict]] = None,
) -> Dict:
    """Main entry point — returns {highlights: [...]} sorted by score.

    `llm_fn` swaps the underlying LLM. Defaults to MuAPI gpt-5-mini; local
    mode passes in a local LLM-backed callable.
    `min_duration`/`max_duration` (seconds) are a hard requirement, not just
    a prompt hint — see `_sanitize_highlights`.
    `candidate_windows`: optional pre-filter result from
    shorts_generator.local.signals.get_candidate_windows(). When provided
    and non-empty, skips scanning the whole transcript in ~19min chunks —
    instead sends ONE call covering just those candidate zones, which is
    the whole point (far fewer tokens/calls for the same or better result,
    since the zones are already backed by real signals). Falls back to the
    normal full-scan behavior if not provided, or if none of the windows
    line up with actual transcript segments.
    """
    llm_fn = llm_fn or call_muapi_llm
    duration = transcript.get("duration", 0)
    content_info = detect_content_type(transcript, llm_fn=llm_fn)
    print(f"[highlights] content={content_info.get('content_type')} density={content_info.get('density')} duration={duration:.0f}s", flush=True)

    if candidate_windows:
        text = build_prefiltered_transcript_text(transcript, candidate_windows)
        if text.strip():
            print(
                f"[highlights] using {len(candidate_windows)} pre-filtered candidate zones "
                f"(1 call instead of a full chunked scan)",
                flush=True,
            )
            result = call_highlight_api(
                text, content_info, duration, num_clips=num_clips, llm_fn=llm_fn,
                min_duration=min_duration, max_duration=max_duration,
                candidate_mode=True, num_candidates=len(candidate_windows),
            )
            return {"highlights": dedupe_highlights(result.get("highlights", []))}
        print("[highlights] candidate zones didn't match any transcript segments, falling back to full scan", flush=True)

    if duration >= LONG_VIDEO_THRESHOLD:
        chunks = chunk_transcript(transcript)
        print(f"[highlights] long video — splitting into {len(chunks)} chunks", flush=True)
        all_highlights: List[Dict] = []
        failed_chunks = 0
        for i, chunk in enumerate(chunks):
            offset = chunk.get("_offset", 0)
            text = build_transcript_text(chunk)
            print(f"[highlights] chunk {i + 1}/{len(chunks)} (offset {offset:.0f}s)", flush=True)
            try:
                result = call_highlight_api(
                    text, content_info, chunk["duration"], num_clips=num_clips, is_chunk=True, llm_fn=llm_fn,
                    min_duration=min_duration, max_duration=max_duration,
                )
            except Exception as e:
                # One bad chunk (rate limit, model refusal, persistent bad
                # JSON) shouldn't throw away everything already found in the
                # other chunks — especially painful on a 2h video where each
                # chunk is its own round of LLM retries. Skip it and keep going.
                failed_chunks += 1
                print(f"[highlights] chunk {i + 1}/{len(chunks)} skipped after repeated failures: {e}", flush=True)
                continue
            for h in result.get("highlights", []):
                h["start_time"] = float(h["start_time"]) + offset
                h["end_time"] = float(h["end_time"]) + offset
                all_highlights.append(h)

        if failed_chunks == len(chunks):
            raise RuntimeError(
                f"All {len(chunks)} chunks failed to produce highlights — see the "
                "per-chunk errors above (likely a provider outage or invalid API key)."
            )
        if failed_chunks:
            print(f"[highlights] {failed_chunks}/{len(chunks)} chunks skipped, continuing with the rest", flush=True)
        highlights = dedupe_highlights(all_highlights)
    else:
        text = build_transcript_text(transcript)
        result = call_highlight_api(
            text, content_info, duration, num_clips=num_clips, llm_fn=llm_fn,
            min_duration=min_duration, max_duration=max_duration,
        )
        highlights = dedupe_highlights(result.get("highlights", []))

    return {"highlights": highlights}
