#!/usr/bin/env python3
"""
ComfyUI LoRA Dataset Queue Runner — Turnaround Sheet Generator
Generates 4 consistent turnaround views (front / right / back / left)
for use with Hunyuan3D mesh + texture pipeline.

Usage:
    python run_lora_queue.py [--seed SEED] [--retry-failed]

Requirements:
    - ComfyUI running at http://127.0.0.1:8188
    - pip install requests
    - image_512x512.png must be in ComfyUI's input folder

Key design decisions:
    - All 4 turnaround views share ONE seed for character consistency.
    - Pass --seed NNNN to lock a specific seed across runs.
    - Completion polling waits for status.completed before moving on.
    - State file survives crashes; already-completed shots are skipped.
"""

import os, sys, json, time, uuid, random, argparse
from datetime import datetime, timezone

import requests

# ── Config ─────────────────────────────────────────────────────────────────
COMFYUI_URL   = "http://127.0.0.1:8188"
POLL_INTERVAL = 5        # seconds between history polls
MAX_RETRIES   = 3        # per-shot retry attempts on error
STATE_FILE    = "queue_state.json"
LOG_FILE      = "run_log.jsonl"
STOP_FILE     = "STOP"   # touch this file to pause between shots

# ── Character anchor ────────────────────────────────────────────────────────
# Describes ONLY stable physical attributes. Pose is specified per-shot.
CHAR = (
    "She has long wavy blonde hair, brown eyes. "
    "She wears a beige floral short-sleeve dress with scalloped hem edges, "
    "a brown leather belt with a small brown leather belt pouch clipped on her RIGHT hip, "
    "and brown lace-up ankle boots that end just above the ankle bone. "
    "Long slender legs with knees at the halfway point of her full-body height. "
    "Anime art style. "
    "Plain white background. "
    "Keep the same face, hairstyle, clothing, colors, and body proportions in every view."
)

# Negative clause appended to every prompt — tells the model what to avoid.
# Flux Kontext ignores a true negative node, so embed it in the positive text.
NEG = (
    "Do NOT change the arm position. "
    "Do NOT cross her arms. "
    "Do NOT fold her arms. "
    "Do NOT place her hands on her hips or touching the dress. "
    "Do NOT change the belt pouch size or position. "
    "Do NOT change the hair volume or wave pattern. "
    "Do NOT add any background objects or shadows. "
)

# ── A-pose arm clause (identical wording used in all 4 shots) ───────────────
# "A-pose" = arms angled ~30° out from the body, elbows straight, hands open.
# Explicit and identical across all views so the model has no ambiguity.
ARM = (
    "Both arms hang straight down in a relaxed A-pose: "
    "each arm angled approximately 30 degrees away from her body, "
    "elbows straight, hands open, fingers together and relaxed, "
    "a clear visible gap between each hand and her thigh/hip. "
    "Hands do NOT touch or press against the dress at any point. "
)


def P(pose):
    """Compose a full prompt: pose description + arm clause + character + negative."""
    return f"{pose} {ARM} {CHAR} {NEG}"


# ── Shot definitions ────────────────────────────────────────────────────────
# Order matters for Hunyuan3D: front → right → back → left.
SHOTS = [
    (
        "turnaround_front",
        P(
            "Full body turnaround sheet — FRONT VIEW. "
            "The character stands facing directly toward the viewer. "
            "Her face, chest, belt pouch, and the entire front of her dress are fully visible. "
            "Her nose points straight at the viewer. "
            "Both ears are visible. "
            "Show her complete body from the crown of her head to the soles of her boots."
        )
    ),
    (
        "turnaround_right_side",
        P(
            "Full body turnaround sheet — RIGHT SIDE PROFILE VIEW (her right side, viewer's left). "
            "The character is rotated exactly 90 degrees clockwise from the front view. "
            "Her RIGHT shoulder, RIGHT arm, and RIGHT leg are closest to the viewer. "
            "Her nose points toward the LEFT edge of the image. "
            "Only her RIGHT ear is visible; her left ear is hidden behind her head. "
            "Her left arm is hidden behind her body — only a thin silhouette of her left side is visible. "
            "The belt pouch on her right hip is fully visible from this angle. "
            "Her hair falls straight down behind her. "
            "Show her complete body from the crown of her head to the soles of her boots."
        )
    ),
    (
        "turnaround_back",
        P(
            "Full body turnaround sheet — BACK VIEW. "
            "The character is facing completely away from the viewer. "
            "Only the BACK of her body is visible: the back of her head, the back of her hair, "
            "the back of her dress, the back of her belt, and the backs of her boots. "
            "Her face is completely hidden — no facial features, cheek, or eye visible at all. "
            "The belt pouch is visible from behind at her right hip. "
            "Show her complete body from the crown of her head to the soles of her boots."
        )
    ),
    (
        "turnaround_left_side",
        P(
            "Full body turnaround sheet — LEFT SIDE PROFILE VIEW (her left side, viewer's right). "
            "The character is rotated exactly 90 degrees counter-clockwise from the front view. "
            "Her LEFT shoulder, LEFT arm, and LEFT leg are closest to the viewer. "
            "Her nose points toward the RIGHT edge of the image. "
            "Only her LEFT ear is visible; her right ear is hidden behind her head. "
            "Her right arm is hidden behind her body — only a thin silhouette of her right side is visible. "
            "The belt pouch on her right hip is hidden or partially visible from this angle. "
            "Her hair falls straight down behind her. "
            "Show her complete body from the crown of her head to the soles of her boots."
        )
    ),
]


# ── Workflow builder ────────────────────────────────────────────────────────
def build_prompt(filename: str, prompt_text: str, seed: int) -> dict:
    """
    Build a ComfyUI API-format prompt dict.
    Uses Flux Kontext with FaceDetailer + HandDetailer chain.
    seed is passed explicitly — caller is responsible for consistency.
    """
    return {
        # Reference image loader
        "190": {
            "class_type": "LoadImage",
            "inputs": {"image": "image_512x512.png"}
        },

        # Model loaders
        "192:39": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "ae.safetensors"}
        },
        "192:38": {
            "class_type": "DualCLIPLoader",
            "inputs": {
                "clip_name1": "clip_l.safetensors",
                "clip_name2": "t5xxl_fp8_e4m3fn_scaled.safetensors",
                "type": "flux",
                "device": "default"
            }
        },
        "192:37": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "flux1-dev-kontext_fp8_scaled.safetensors",
                "weight_dtype": "default"
            }
        },

        # Image prep — stitch reference image for Kontext context panel
        "192:146": {
            "class_type": "ImageStitch",
            "inputs": {
                "direction": "right",
                "match_image_size": True,
                "spacing_width": 0,
                "spacing_color": "white",
                "image1": ["190", 0]
            }
        },
        "192:42": {
            "class_type": "FluxKontextImageScale",
            "inputs": {"image": ["192:146", 0]}
        },
        "192:124": {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": ["192:42", 0],
                "vae": ["192:39", 0]
            }
        },

        # Conditioning
        "192:6": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": prompt_text,
                "clip": ["192:38", 0]
            }
        },
        "192:135": {
            "class_type": "ConditioningZeroOut",
            "inputs": {"conditioning": ["192:6", 0]}
        },
        "192:177": {
            "class_type": "ReferenceLatent",
            "inputs": {
                "conditioning": ["192:6", 0],
                "latent": ["192:124", 0]
            }
        },
        "192:35": {
            "class_type": "FluxGuidance",
            "inputs": {
                "guidance": 3.5,
                "conditioning": ["192:177", 0]
            }
        },

        # Sampler
        "192:31": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": 28,
                "cfg": 1,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1,
                "model": ["192:37", 0],
                "positive": ["192:35", 0],
                "negative": ["192:135", 0],
                "latent_image": ["192:124", 0]
            }
        },
        "192:8": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["192:31", 0],
                "vae": ["192:39", 0]
            }
        },

        # ── Face detailer ────────────────────────────────────────────────
        "200": {
            "class_type": "UltralyticsDetectorProvider",
            "inputs": {"model_name": "bbox/face_yolov8m.pt"}
        },
        "201": {
            "class_type": "FaceDetailer",
            "inputs": {
                "image": ["192:8", 0],
                "model": ["192:37", 0],
                "clip": ["192:38", 0],
                "vae": ["192:39", 0],
                "positive": ["192:6", 0],
                "negative": ["192:135", 0],
                "bbox_detector": ["200", 0],
                "guide_size": 512,
                "guide_size_for": True,
                "max_size": 1024,
                "seed": seed,
                "steps": 20,
                "cfg": 1,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 0.4,
                "feather": 20,
                "noise_mask": True,
                "force_inpaint": True,
                "bbox_threshold": 0.5,
                "bbox_dilation": 10,
                "bbox_crop_factor": 3.0,
                "sam_detection_hint": "center-1",
                "sam_dilation": 0,
                "sam_threshold": 0.93,
                "sam_bbox_expansion": 0,
                "sam_mask_hint_threshold": 0.7,
                "sam_mask_hint_use_negative": "False",
                "drop_size": 10,
                "wildcard": "",
                "cycle": 1,
                "inpaint_model": False,
                "noise_mask_feather": 20,
                "sam_model_opt": None,
                "segm_detector_opt": None,
            }
        },

        # ── Hand detailer ────────────────────────────────────────────────
        "202": {
            "class_type": "UltralyticsDetectorProvider",
            "inputs": {"model_name": "bbox/hand_yolov8s.pt"}
        },
        "203": {
            "class_type": "FaceDetailer",   # FaceDetailer is reused for hands
            "inputs": {
                "image": ["201", 0],        # chains from face detailer output
                "model": ["192:37", 0],
                "clip": ["192:38", 0],
                "vae": ["192:39", 0],
                "positive": ["192:6", 0],
                "negative": ["192:135", 0],
                "bbox_detector": ["202", 0],
                "guide_size": 512,
                "guide_size_for": True,
                "max_size": 1024,
                "seed": seed,
                "steps": 20,
                "cfg": 1,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 0.4,
                "feather": 20,
                "noise_mask": True,
                "force_inpaint": True,
                "bbox_threshold": 0.5,
                "bbox_dilation": 10,
                "bbox_crop_factor": 2.5,
                "sam_detection_hint": "center-1",
                "sam_dilation": 0,
                "sam_threshold": 0.93,
                "sam_bbox_expansion": 0,
                "sam_mask_hint_threshold": 0.7,
                "sam_mask_hint_use_negative": "False",
                "drop_size": 10,
                "wildcard": "",
                "cycle": 1,
                "inpaint_model": False,
                "noise_mask_feather": 20,
                "sam_model_opt": None,
                "segm_detector_opt": None,
            }
        },

        # ── Save ─────────────────────────────────────────────────────────
        "136": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": filename,
                "images": ["203", 0]
            }
        },
    }


# ── ComfyUI API helpers ─────────────────────────────────────────────────────
def queue_prompt(api_prompt: dict) -> str:
    client_id = str(uuid.uuid4())
    resp = requests.post(
        f"{COMFYUI_URL}/prompt",
        json={"prompt": api_prompt, "client_id": client_id},
        timeout=30
    )
    if resp.status_code != 200:
        try:
            err = resp.json()
        except Exception:
            err = resp.text
        raise RuntimeError(f"HTTP {resp.status_code}: {json.dumps(err)[:600]}")
    data = resp.json()
    if "prompt_id" not in data:
        raise RuntimeError(f"Unexpected response: {data}")
    return data["prompt_id"]


def wait_for_completion(prompt_id: str) -> bool:
    """
    Poll /history/{prompt_id} until status.completed is True.
    Raises RuntimeError on execution_error.
    Returns True on success.
    """
    while True:
        try:
            resp = requests.get(f"{COMFYUI_URL}/history/{prompt_id}", timeout=10)
            if resp.status_code == 200:
                history = resp.json()
                if prompt_id in history:
                    status = history[prompt_id].get("status", {})
                    # Check for execution error first
                    for msg in status.get("messages", []):
                        if isinstance(msg, (list, tuple)) and msg[0] == "execution_error":
                            raise RuntimeError(f"Execution error: {msg[1]}")
                    if status.get("completed", False):
                        return True
        except RuntimeError:
            raise
        except requests.exceptions.ConnectionError:
            print("  [warning] Lost connection, retrying...")
        time.sleep(POLL_INTERVAL)


def get_completed_filenames() -> set:
    """Return set of base filenames already saved in ComfyUI history."""
    try:
        resp = requests.get(f"{COMFYUI_URL}/history", timeout=10)
        if resp.status_code != 200:
            return set()
        done = set()
        for entry in resp.json().values():
            for node_output in entry.get("outputs", {}).values():
                for img in node_output.get("images", []):
                    fname = img.get("filename", "")
                    # Strip trailing _00001_ counter and extension
                    base = fname.rsplit("_", 1)[0] if "_" in fname else fname
                    base = base.rsplit(".", 1)[0]
                    done.add(base)
        return done
    except Exception:
        return set()


# ── State helpers ───────────────────────────────────────────────────────────
def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "completed": {},
        "failed": {},
        "seed": None,
        "started_at": datetime.now(timezone.utc).isoformat()
    }


def save_state(state: dict):
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_FILE)


def log_event(event: dict):
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(event) + "\n")


def should_stop() -> bool:
    return os.path.exists(STOP_FILE)


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="ComfyUI turnaround sheet generator")
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Fixed seed for all 4 views. Omit to auto-generate (saved to state file)."
    )
    parser.add_argument(
        "--retry-failed", action="store_true",
        help="Re-run shots that previously failed instead of skipping them."
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  ComfyUI Turnaround Sheet Generator")
    print(f"  Target : {COMFYUI_URL}")
    print(f"  Shots  : {len(SHOTS)}")
    print("=" * 60)

    # ── Check ComfyUI is reachable ─────────────────────────────────────
    try:
        requests.get(f"{COMFYUI_URL}/system_stats", timeout=5)
        print("\nComfyUI connection: OK")
    except Exception:
        print(f"\n[ERROR] Cannot reach ComfyUI at {COMFYUI_URL}")
        sys.exit(1)

    # ── Resolve seed ───────────────────────────────────────────────────
    state = load_state()

    if args.seed is not None:
        # Caller specified a seed explicitly
        shared_seed = args.seed
        print(f"\nSeed (from --seed arg): {shared_seed}")
    elif state.get("seed") is not None:
        # Resume a previous run — reuse the same seed for consistency
        shared_seed = state["seed"]
        print(f"\nSeed (resumed from state): {shared_seed}")
    else:
        # First run — generate and persist
        shared_seed = random.randint(0, 2**32 - 1)
        print(f"\nSeed (newly generated): {shared_seed}")

    state["seed"] = shared_seed
    save_state(state)

    # ── Test shot ──────────────────────────────────────────────────────
    print("\nSubmitting test job (front view, seed locked)...")
    try:
        test_prompt = build_prompt("_test_shot", SHOTS[0][1], shared_seed)
        pid = queue_prompt(test_prompt)
        print(f"  Queued (id={pid[:8]}...) — waiting for completion...")
        wait_for_completion(pid)
        print("  Test passed! Starting full run.\n")
    except Exception as e:
        print(f"\n[ERROR] Test job failed: {e}")
        sys.exit(1)

    # ── Check ComfyUI history for already-saved images ─────────────────
    print("Checking history for already-completed images...")
    already_done = get_completed_filenames()
    print(f"  {len(already_done)} already saved in ComfyUI history.\n")

    # ── Clear failed shots if retry requested ──────────────────────────
    if args.retry_failed and state["failed"]:
        print(f"--retry-failed: clearing {len(state['failed'])} failed shots for retry.")
        state["failed"] = {}
        save_state(state)

    # ── Main loop ──────────────────────────────────────────────────────
    total     = len(SHOTS)
    completed = 0
    skipped   = 0
    errors    = 0

    for idx, (name, prompt_text) in enumerate(SHOTS):
        print(f"\n[{idx+1}/{total}] {name}")

        if should_stop():
            print("  STOP file detected. Exiting safely.")
            break

        # Skip if already completed in this run's state
        if name in state["completed"]:
            print("  [SKIP] Already completed in state file.")
            skipped += 1
            continue

        # Skip if already found in ComfyUI history
        if name in already_done:
            print("  [SKIP] Found in ComfyUI history.")
            state["completed"][name] = {"skipped": True}
            save_state(state)
            skipped += 1
            continue

        # Skip failed shots unless --retry-failed was passed
        if name in state["failed"] and not args.retry_failed:
            print(f"  [SKIP] Previously failed: {state['failed'][name].get('error', '?')}")
            skipped += 1
            continue

        # ── Run with retries ───────────────────────────────────────────
        success = False
        last_error = None
        start_iso = datetime.now(timezone.utc).isoformat()
        start_time = time.time()

        for attempt in range(1, MAX_RETRIES + 1):
            if attempt > 1:
                wait = 10 * attempt
                print(f"  Retry {attempt}/{MAX_RETRIES} in {wait}s...")
                time.sleep(wait)

            try:
                api_prompt = build_prompt(name, prompt_text, shared_seed)
                prompt_id  = queue_prompt(api_prompt)
                print(f"  Queued (id={prompt_id[:8]}...) — waiting...")
                wait_for_completion(prompt_id)
                success = True
                break
            except Exception as e:
                last_error = e
                print(f"  [attempt {attempt} failed] {e}")

        duration = time.time() - start_time

        if success:
            completed += 1
            state["completed"][name] = {
                "duration_sec": round(duration, 2),
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "seed": shared_seed,
            }
            # Remove from failed if it was there before
            state["failed"].pop(name, None)
            save_state(state)
            log_event({
                "type": "success",
                "name": name,
                "seed": shared_seed,
                "duration_sec": round(duration, 2),
                "started_at": start_iso,
                "ended_at": datetime.now(timezone.utc).isoformat(),
            })
            print(f"  [DONE] {duration:.1f}s")
        else:
            errors += 1
            state["failed"][name] = {
                "error": str(last_error),
                "duration_sec": round(duration, 2),
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "seed": shared_seed,
            }
            save_state(state)
            log_event({
                "type": "error",
                "name": name,
                "seed": shared_seed,
                "error": str(last_error),
                "duration_sec": round(duration, 2),
                "started_at": start_iso,
                "ended_at": datetime.now(timezone.utc).isoformat(),
            })
            print(f"  [ERROR] {last_error}")

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  Done!  completed={completed}  skipped={skipped}  errors={errors}")
    print(f"  Seed used: {shared_seed}")
    print(f"  Re-run with: python run_lora_queue.py --seed {shared_seed}")
    if errors:
        print(f"  To retry failed shots: python run_lora_queue.py --seed {shared_seed} --retry-failed")
    print("=" * 60)


if __name__ == "__main__":
    main()
