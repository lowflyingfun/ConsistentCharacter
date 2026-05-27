#!/usr/bin/env python3
"""
Character Consistency Reviewer — Gemini Edition (Free Tier)
Optimized for Gemini 2.0 Flash Lite & Google-GenAI SDK
"""

import argparse
import json
import os
import sys
import time
import io
import base64
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

# Safety delays for Free Tier
INTER_IMAGE_DELAY = 5.0
RETRY_WAIT_SECONDS = 65
MAX_RETRIES = 3

REVIEW_PROMPT = """You are a character anatomy and identity auditor. 
Your goal is to ensure the GENERATED IMAGE is the EXACT SAME PERSON as the REFERENCE IMAGE, 
regardless of the pose, background, or camera angle.

DIRECTIONS:
1. POSE INDEPENDENCE: Ignore the arm/leg positions. Poses SHOULD be different.
2. ANATOMICAL CONSISTENCY: Compare the ratio of head-to-body, limb length (arm-to-torso ratio), 
   and hand/foot scale.
3. IDENTITY LOCK: Verify hair silhouette, eye shape, and facial landmarks (nose-to-mouth distance).
4. OUTFIT DNA: Check for consistent patterns, colors, and specific accessories.

Return ONLY a JSON object with these keys:
- anatomy_proportions: {score: 1-5, notes: "Focus on limb ratios/head size"}
- identity_features: {score: 1-5, notes: "Focus on face/hair consistency"}
- outfit_dna: {score: 1-5, notes: "Focus on costume details"}
- overall_score: float
- pass: boolean (true if overall_score >= 3.5)
- issues: [list of strings regarding anatomical drift or identity loss]
"""

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_results(results_file: Path) -> dict:
    if results_file.exists():
        with open(results_file) as f:
            return json.load(f)
    return {
        "meta": {
            "created_at": datetime.now().isoformat(),
            "reference": None,
            "folder": None,
            "model": MODEL,
        },
        "results": {}
    }

def save_results(data: dict, results_file: Path):
    tmp = results_file.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(results_file)

def pil_to_bytes(img: Image.Image) -> bytes:
    """Helper to convert PIL image to bytes for the API."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

import google.generativeai as old_genai # Change at top of script

def review_image(client, ref_img, gen_img):
    # Initialize the old way
    old_genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    model = old_genai.GenerativeModel(MODEL)
    
    response = model.generate_content(
        [
            "REFERENCE IMAGE:", ref_img, 
            "GENERATED IMAGE:", gen_img, 
            REVIEW_PROMPT
        ],
        generation_config={"response_mime_type": "application/json"}
    )
    return json.loads(response.text)

def print_result(filename: str, result: dict):
    status = "PASS" if result.get("pass") else "FAIL"
    score = result.get("overall_score", "?")
    print(f"\n  [{status}]  [{score}/5]  {filename}")
    
    # Update these keys to match your NEW prompt exactly:
    for key, label in [
        ("anatomy_proportions", "Anatomy      "),
        ("identity_features",  "Identity     "),
        ("outfit_dna",          "Outfit DNA   "),
    ]:
        sub = result.get(key, {})
        # If the AI returns a flat score instead of a nested dict, 
        # this handles both cases:
        val = sub.get('score', '?') if isinstance(sub, dict) else sub
        notes = sub.get('notes', '') if isinstance(sub, dict) else ""
        print(f"    {label}: {val}/5 -- {notes}")
    
    issues = result.get("issues", [])
    if issues:
        print(f"    Issues: {'; '.join(issues)}")

def print_summary(data: dict):
    results = data["results"]
    reviewed = [r for r in results.values() if "error" not in r]
    if not reviewed: return

    passed = sum(1 for r in reviewed if r.get("pass"))
    avg = sum(r.get("overall_score", 0) for r in reviewed) / len(reviewed)

    print("\n" + "=" * 60)
    print(f"  SUMMARY: {len(reviewed)} images reviewed")
    print(f"  PASS: {passed}   FAIL: {len(reviewed)-passed}   Avg: {avg:.2f}/5")
    print("=" * 60 + "\n")

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--folder", required=True)
    parser.add_argument("--results", default=DEFAULT_RESULTS_FILE)
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set.")
        sys.exit(1)

    ref_path = Path(args.reference).resolve()
    fld_path = Path(args.folder).resolve()
    res_path = Path(args.results).resolve()

    if not ref_path.exists() or not fld_path.is_dir():
        print("ERROR: Invalid paths.")
        sys.exit(1)

    images = sorted([p for p in fld_path.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS])
    data = load_results(res_path)
    
    todo = [img for img in images if img.name not in data["results"] or "error" in data["results"][img.name]]
    if not todo:
        print("Nothing left to review.")
        print_summary(data)
        return

    client = genai.Client(api_key=api_key)
    ref_img = Image.open(ref_path).convert("RGB")

    print(f"Starting review of {len(todo)} images...\n")

    for i, img_path in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {img_path.name}...", end="", flush=True)
        
        try:
            gen_img = Image.open(img_path).convert("RGB")
            
            # Retry logic for rate limits
            result = None
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    result = review_image(client, ref_img, gen_img)
                    break
                except Exception as e:
                    if "429" in str(e) and attempt < MAX_RETRIES:
                        time.sleep(RETRY_WAIT_SECONDS)
                    else:
                        raise e

            data["results"][img_path.name] = result
            print_result(img_path.name, result)

        except Exception as e:
            print(f" ERROR: {e}")
            data["results"][img_path.name] = {"error": str(e)}

        save_results(data, res_path)
        if i < len(todo): time.sleep(INTER_IMAGE_DELAY)

    print_summary(data)

if __name__ == "__main__":
    main()