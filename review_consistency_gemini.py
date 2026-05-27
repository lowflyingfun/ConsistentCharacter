#!/usr/bin/env python3
"""
Character Consistency Reviewer — Gemini Edition
Scores generated images against a reference for anatomy, identity, and outfit consistency.
Uses the google-genai SDK with Gemini 2.5 Flash.

Usage:
    python review_consistency_gemini.py --reference ref.png --folder ./output_images

Requirements:
    - GEMINI_API_KEY set in your environment
    - pip install google-genai Pillow
"""

import argparse
import json
import os
import sys
import time
import io
from pathlib import Path
from datetime import datetime

try:
    from google import genai
    from google.genai import types
    from PIL import Image
except ImportError:
    print("Missing dependencies. Run: pip install google-genai Pillow")
    sys.exit(1)

# ── Config ───────────────────────────────────────────────────────────────────

MODEL = "gemini-2.5-flash"
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
DEFAULT_RESULTS_FILE = "consistency_results.json"

# Rate-limit safety delays (Free Tier)
INTER_IMAGE_DELAY  = 5.0   # seconds between successful requests
RETRY_WAIT_SECONDS = 65    # seconds to wait after a 429 before retrying
MAX_RETRIES        = 3     # per-image retry attempts on rate-limit errors

REVIEW_PROMPT = """You are a character anatomy and identity auditor.
Your goal is to ensure the GENERATED IMAGE is the EXACT SAME PERSON as the REFERENCE IMAGE,
regardless of the pose, background, or camera angle.

DIRECTIONS:
1. POSE INDEPENDENCE: Ignore the arm/leg positions. Poses SHOULD be different.
2. ANATOMICAL CONSISTENCY: Compare the ratio of head-to-body, limb length (arm-to-torso ratio),
   and hand/foot scale.
3. IDENTITY LOCK: Verify hair silhouette, eye shape, and facial landmarks (nose-to-mouth distance).
4. OUTFIT DNA: Check for consistent patterns, colors, and specific accessories.

Return ONLY a JSON object with these exact keys:
- anatomy_proportions: {score: 1-5, notes: "Focus on limb ratios/head size"}
- identity_features:   {score: 1-5, notes: "Focus on face/hair consistency"}
- outfit_dna:          {score: 1-5, notes: "Focus on costume details"}
- overall_score: float
- pass: boolean (true if overall_score >= 3.5)
- issues: [list of strings describing any anatomical drift or identity loss]
"""

# ── Helpers ──────────────────────────────────────────────────────────────────

def pil_to_bytes(img: Image.Image) -> bytes:
    """Convert a PIL image to PNG bytes for the API."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def load_results(results_file: Path) -> dict:
    """Load existing results from disk, or return a fresh results structure."""
    if results_file.exists():
        with open(results_file) as f:
            return json.load(f)
    return {
        "meta": {
            "created_at": datetime.now().isoformat(),
            "model": MODEL,
        },
        "results": {}
    }


def save_results(data: dict, results_file: Path):
    """Atomically write results to disk via a temp file."""
    tmp = results_file.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(results_file)


def review_image(client: genai.Client, ref_img: Image.Image, gen_img: Image.Image) -> dict:
    """
    Send reference + generated image to Gemini and return the parsed JSON result.
    Raises on API errors so the caller can handle retries.
    """
    response = client.models.generate_content(
        model=MODEL,
        contents=[
            types.Part.from_text("REFERENCE IMAGE:"),
            types.Part.from_bytes(data=pil_to_bytes(ref_img), mime_type="image/png"),
            types.Part.from_text("GENERATED IMAGE:"),
            types.Part.from_bytes(data=pil_to_bytes(gen_img), mime_type="image/png"),
            types.Part.from_text(REVIEW_PROMPT),
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        ),
    )
    return json.loads(response.text)


def print_result(filename: str, result: dict):
    """Pretty-print a single image result to the terminal."""
    status = "PASS" if result.get("pass") else "FAIL"
    score  = result.get("overall_score", "?")
    print(f"\n  [{status}]  [{score}/5]  {filename}")

    for key, label in [
        ("anatomy_proportions", "Anatomy    "),
        ("identity_features",   "Identity   "),
        ("outfit_dna",          "Outfit DNA "),
    ]:
        sub   = result.get(key, {})
        val   = sub.get("score", "?") if isinstance(sub, dict) else sub
        notes = sub.get("notes", "")  if isinstance(sub, dict) else ""
        print(f"    {label}: {val}/5 -- {notes}")

    issues = result.get("issues", [])
    if issues:
        print(f"    Issues: {'; '.join(issues)}")


def print_summary(data: dict):
    """Print pass/fail totals and average score across all reviewed images."""
    reviewed = [r for r in data["results"].values() if "error" not in r]
    if not reviewed:
        return

    passed = sum(1 for r in reviewed if r.get("pass"))
    avg    = sum(r.get("overall_score", 0) for r in reviewed) / len(reviewed)

    print("\n" + "=" * 60)
    print(f"  SUMMARY  :  {len(reviewed)} images reviewed")
    print(f"  PASS     :  {passed}   FAIL: {len(reviewed) - passed}   Avg: {avg:.2f}/5")
    print("=" * 60 + "\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Score generated images for character consistency against a reference."
    )
    parser.add_argument("--reference", required=True,  help="Path to the reference image")
    parser.add_argument("--folder",    required=True,  help="Folder of generated images to review")
    parser.add_argument("--results",   default=DEFAULT_RESULTS_FILE,
                        help=f"Output JSON file (default: {DEFAULT_RESULTS_FILE})")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable is not set.")
        print("  Set it with:  export GEMINI_API_KEY=your_key_here")
        sys.exit(1)

    ref_path = Path(args.reference).resolve()
    fld_path = Path(args.folder).resolve()
    res_path = Path(args.results).resolve()

    if not ref_path.exists():
        print(f"ERROR: Reference image not found: {ref_path}")
        sys.exit(1)
    if not fld_path.is_dir():
        print(f"ERROR: Folder not found: {fld_path}")
        sys.exit(1)

    images = sorted(p for p in fld_path.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS)
    data   = load_results(res_path)

    # Skip images that already have a clean result; retry ones that errored
    todo = [
        img for img in images
        if img.name not in data["results"] or "error" in data["results"][img.name]
    ]

    if not todo:
        print("All images already reviewed. Nothing to do.")
        print_summary(data)
        return

    client  = genai.Client(api_key=api_key)
    ref_img = Image.open(ref_path).convert("RGB")

    print(f"Reference : {ref_path.name}")
    print(f"Folder    : {fld_path}")
    print(f"To review : {len(todo)} images\n")

    for i, img_path in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {img_path.name}...", end="", flush=True)

        try:
            gen_img = Image.open(img_path).convert("RGB")

            result = None
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    result = review_image(client, ref_img, gen_img)
                    break
                except Exception as e:
                    if "429" in str(e) and attempt < MAX_RETRIES:
                        print(f"\n  [rate limit] waiting {RETRY_WAIT_SECONDS}s before retry {attempt + 1}...",
                              end="", flush=True)
                        time.sleep(RETRY_WAIT_SECONDS)
                    else:
                        raise

            data["results"][img_path.name] = result
            print_result(img_path.name, result)

        except Exception as e:
            print(f"\n  ERROR: {e}")
            data["results"][img_path.name] = {"error": str(e)}

        save_results(data, res_path)

        if i < len(todo):
            time.sleep(INTER_IMAGE_DELAY)

    print_summary(data)


if __name__ == "__main__":
    main()
