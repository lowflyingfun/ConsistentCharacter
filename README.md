# ComfyUI Turnaround Sheet Generator

A Python script that automates the generation of 4-view character turnaround sheets (front, right, back, left) using **Flux Kontext** inside ComfyUI. Designed for feeding clean turnaround poses into Hunyuan3D or LoRA training pipelines.

---

## What it does

- Submits 4 sequential generation jobs to ComfyUI via its API
- All 4 views share a **single seed** for character consistency across angles
- Each generation runs through a **FaceDetailer → HandDetailer** chain automatically
- Crash-safe: state is persisted to disk so interrupted runs can be resumed
- Supports a `STOP` file to pause cleanly between shots

---

## Requirements

### System
- Python 3.9+
- ComfyUI installed and running with the **API enabled**
- A reference image named `image_512x512.png` placed in ComfyUI's `input/` folder

### Python dependencies

```bash
pip install requests
```

### ComfyUI models

Place these in your ComfyUI `models/` subfolders:

| File | Folder |
|---|---|
| `flux1-dev-kontext_fp8_scaled.safetensors` | `models/unet/` |
| `ae.safetensors` | `models/vae/` |
| `clip_l.safetensors` | `models/clip/` |
| `t5xxl_fp8_e4m3fn_scaled.safetensors` | `models/clip/` |
| `face_yolov8m.pt` | `models/ultralytics/bbox/` |
| `hand_yolov8s.pt` | `models/ultralytics/bbox/` |

> **Flux Kontext** is available from Black Forest Labs. The fp8-scaled variant is recommended for VRAM efficiency. The YOLO detector models can be downloaded via the ComfyUI Manager model browser.

### ComfyUI custom nodes

Install these via [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager):

| Node pack | Purpose |
|---|---|
| `ComfyUI-Impact-Pack` | FaceDetailer and UltralyticsDetectorProvider nodes |
| `ComfyUI-KJNodes` | ImageStitch node |
| `ComfyUI-Flux-Kontext` | FluxKontextImageScale and ReferenceLatent nodes |

---

## Enabling the ComfyUI API

ComfyUI's API must be active for this script to work. Start ComfyUI with:

```bash
python main.py --listen 127.0.0.1 --port 8188
```

The API is enabled by default when you run ComfyUI normally. You can verify it's working by opening `http://127.0.0.1:8188/system_stats` in your browser — it should return a JSON response.

---

## Customising the character

Open `run_lora_queue.py` and edit the three prompt constants near the top of the file:

- **`CHAR`** — stable physical attributes: hair, eyes, outfit, body proportions, art style. This is injected into every shot.
- **`ARM`** — arm/hand pose clause. The default is a relaxed A-pose. Modify if your character needs a different neutral stance.
- **`NEG`** — things to avoid (embedded in the positive prompt since Flux ignores a true negative). Add or remove constraints here.

The `SHOTS` list defines the 4 camera angles. Each entry is a `(name, prompt)` tuple. Names are used as the output filename prefix and for state tracking, so keep them unique.

---

## Usage

### Basic run (auto-generates a new seed)

```bash
CharacterPoseGenerator.py
```

### Lock a specific seed

```bash
python CharacterPoseGenerator.py --seed 3141592653
```

### Resume an interrupted run

Just re-run the same command. The script reads `queue_state.json` and skips any shots already marked complete.

### Retry failed shots

```bash
python CharacterPoseGenerator.py --seed YOUR_SEED --retry-failed
```

### Pause between shots

While the script is running, create a file named `STOP` in the same directory:

```bash
touch STOP   # Linux/macOS
echo. > STOP # Windows
```

The script will exit cleanly after the current shot finishes. Delete the file before resuming.

---

## Output files

| File | Description |
|---|---|
| `queue_state.json` | Persisted run state: completed/failed shots and the active seed |
| `run_log.jsonl` | Per-shot event log with timestamps and durations |
| `turnaround_front_*.png` etc. | Generated images, saved to ComfyUI's `output/` folder |

The seed used for a run is printed at the end of execution and stored in `queue_state.json`. To reproduce a previous run exactly, pass that seed via `--seed`.

---

## Troubleshooting

**`Cannot reach ComfyUI at http://127.0.0.1:8188`**
ComfyUI isn't running or is on a different port. Start it first, or update `COMFYUI_URL` at the top of the script.

**`HTTP 400` / node errors in the ComfyUI terminal**
A custom node is missing or a model file isn't found. Check that all entries in the models and nodes tables above are installed and in the correct subfolders.

**`face_yolov8m.pt` / `hand_yolov8s.pt` not found**
Open ComfyUI Manager → Model Manager → search for `face_yolov8m` and `hand_yolov8s` and install them from there.

**Images generate but faces look degraded**
The FaceDetailer denoise is set to `0.4` by default. Lower it (e.g. `0.3`) in the `build_prompt()` function if the detailer is changing the face too aggressively.

**All 4 shots look identical / poses aren't changing**
Check that the `SHOTS` list has unique pose descriptions per entry and that you're not accidentally pinning the same conditioning across all views.





# Character Consistency Reviewer — Gemini Edition

A Python script that uses **Gemini 2.5 Flash** to automatically score a batch of generated images against a reference for character consistency. Useful for auditing LoRA training datasets or AI-generated turnaround sheets before feeding them into a 3D pipeline.

Each image is scored across three dimensions:

| Dimension | What it checks |
|---|---|
| Anatomy | Head-to-body ratio, limb length, hand/foot scale |
| Identity | Hair silhouette, eye shape, facial landmarks |
| Outfit DNA | Costume colours, patterns, accessories |

Results are saved incrementally to a JSON file so interrupted runs can be resumed.

---

## Requirements

- Python 3.9+
- A free Google Gemini API key (see below)

### Python dependencies

```bash
pip install google-genai Pillow
```

---

## Getting a Gemini API key

1. Go to [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey)
2. Sign in with a Google account
3. Click **Create API key**
4. Copy the key — you won't be shown it again, but you can always create a new one

The free tier is sufficient for this script. The only constraint is rate limits, which the script handles automatically with delays and retries.

---

## Setting your API key

The script reads the key from an environment variable called `GEMINI_API_KEY`. You need to set this in your terminal session before running the script.

### Windows (Command Prompt)

```cmd
set GEMINI_API_KEY=your_key_here
```

This only lasts for the current CMD window. To make it permanent, add it via **System Properties → Environment Variables**.

### Windows (PowerShell)

```powershell
$env:GEMINI_API_KEY = "your_key_here"
```

Again, this is session-only. To persist it:

```powershell
[System.Environment]::SetEnvironmentVariable("GEMINI_API_KEY", "your_key_here", "User")
```

### macOS / Linux

```bash
export GEMINI_API_KEY=your_key_here
```

To make it permanent, add that line to your `~/.bashrc`, `~/.zshrc`, or equivalent shell config file.

---

## Usage

```bash
python review_consistency_gemini.py --reference ref.png --folder ./generated_images
```

### Arguments

| Argument | Required | Description |
|---|---|---|
| `--reference` | Yes | Path to your reference character image |
| `--folder` | Yes | Path to the folder of generated images to review |
| `--results` | No | Output JSON filename (default: `consistency_results.json`) |

### Example

```bash
python review_consistency_gemini.py \
  --reference ./my_character_ref.png \
  --folder ./batch_01 \
  --results batch_01_results.json
```

---

## Output

### Terminal

Each image prints inline as it's processed:

```
[1/24] turnaround_front_00001_.png...
  [PASS]  [4.1/5]  turnaround_front_00001_.png
    Anatomy    : 4/5 -- Proportions consistent with reference
    Identity   : 4/5 -- Hair silhouette and eye shape match
    Outfit DNA : 4/5 -- Belt pouch and dress colours consistent

SUMMARY  :  24 images reviewed
PASS     :  21   FAIL: 3   Avg: 3.87/5
```

### JSON results file

A `consistency_results.json` file is written (and updated after every image) with the full scores and notes for each file. You can use this to filter, sort, or script further automation on the results.

---

## Resuming interrupted runs

If the script is stopped partway through, just re-run the same command. It reads the results file on startup and skips any images that already have a clean result. Images that errored will be retried automatically.

---

## Rate limits

The free tier of the Gemini API has per-minute request limits. The script handles this automatically:

- A **5-second delay** is inserted between every image
- If a `429 Too Many Requests` error is received, the script waits **65 seconds** and retries up to **3 times** before marking that image as an error

If you're running very large batches on the free tier, you can increase `INTER_IMAGE_DELAY` at the top of the script to reduce the chance of hitting the limit at all.

---

## Supported image formats

`.png`, `.jpg`, `.jpeg`, `.webp`
