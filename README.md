# nanobanana-smart-refiner

<p align="center">
  <img src="BANANA.png" alt="Smart Image Refiner Demo" width="300"/>
  <br/>
  Minimal image generator with strict self‑evaluation.
  <br/>
  Prompt → generate → score → repeat N times.
  <br/>
  Saves all images to `current/`.
</p>

## Demo

![Demo Video](demo_video.mp4)

https://github.com/pabed/nanobanana-smart-refiner/blob/main/demo_video.mp4?raw=true

*Watch the nanobanana-smart-refiner in action: transforming vintage poses into dancing poses while preserving all other image characteristics with crystal-clear text readability (1920x1000, broadcast quality)*

## Quick Start

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export GOOGLE_API_KEY="your_key"
```

## Usage

```bash
python enhanced_agent_v2.py "<prompt>" [optional_reference_image] --iterations <N>
```

Examples (smart and exact N):
```bash
# Text‑only, smart mode (up to 6 with early stop)
python enhanced_agent_v2.py "two people handshake in office"

# Text‑only, exactly 3 images
python enhanced_agent_v2.py "two people handshake in office" --iterations 3

# With one reference image, smart mode
python enhanced_agent_v2.py "portrait with natural lighting" ./1.png

# With one reference image, exactly 2 images
python enhanced_agent_v2.py "portrait with natural lighting" ./1.png --iterations 2
```

Output files: `current/iteration_<i>_<session>.png` and a concise score line per iteration. In smart mode, it stops early once all minimums are met and overall ≥ 8.5.

## Models
- Generation: `gemini-2.5-flash-image-preview`
- Evaluation: `models/gemini-2.5-flash`

## Scoring (concise)
- Metrics: Face Fidelity, Anatomy, Consistency, Accuracy, Quality, Satisfaction (weighted; Face+Anatomy emphasized).
- Minimums: Accuracy ≥ 8.5, Quality ≥ 8.0, Satisfaction ≥ 8.0. If a minimum fails, overall is capped at 8.4.

Notes
- The same prompt runs for exactly N iterations (no prompt refinement).
- One optional reference image is supported (positional argument).