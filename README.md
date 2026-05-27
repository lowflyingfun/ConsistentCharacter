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
