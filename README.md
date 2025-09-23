# nanobanana-smart-refiner (simplified)

<p align="center">
  <img src="BANANA.png" alt="Smart Image Refiner Demo" width="300"/>
  <br/>
  Minimal image generator with strict self‑evaluation.
  <br/>
  Prompt → generate → score → repeat N times.
  <br/>
  Saves all images to `current/`.
  </p>

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

Examples:
```bash
# Text‑only, 3 images
python enhanced_agent_v2.py "two people handshake in office" --iterations 3

# With one reference image, 2 images
python enhanced_agent_v2.py "portrait with natural lighting" ./1.png --iterations 2
```

Output files: `current/iteration_<i>_<session>.png` and a concise score line per iteration.

## Models
- Generation: `gemini-2.5-flash-image-preview`
- Evaluation: `models/gemini-2.5-flash`

## Scoring (concise)
- Metrics: Face Fidelity, Anatomy, Consistency, Accuracy, Quality, Satisfaction (weighted; Face+Anatomy emphasized).
- Minimums: Accuracy ≥ 8.5, Quality ≥ 8.0, Satisfaction ≥ 8.0. If a minimum fails, overall is capped at 8.4.

Notes
- The same prompt runs for exactly N iterations (no prompt refinement).
- One optional reference image is supported (positional argument).