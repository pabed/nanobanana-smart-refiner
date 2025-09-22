# nanobanana-smart-refiner

<p align="center">
  <img src="BANANA.png" alt="Smart Image Refiner Demo" width="300"/>
</p>

Make great images without prompt gymnastics. Give it a goal and (ideally) a reference image — it iterates, self‑evaluates, tweaks the prompt, and saves the best shots. Easy.

## How The Evaluation Works

The system uses **6 weighted criteria** with strict minimum requirements:

### Scoring Weights
- **Face Fidelity (35%)**: Identity consistency and natural facial proportions
- **Anatomy (35%)**: Correct human anatomy - counts limbs, digits, checks proportions  
- **Consistency (5%)**: Internal image consistency (no duplicated/morphing parts)
- **Accuracy (15%)**: How well it fulfills your prompt
- **Quality (7%)**: Technical quality, clarity, composition  
- **Satisfaction (3%)**: Overall user satisfaction

### Minimum Thresholds ⚠️
For an image to be approved (score ≥8.5/10), these minimums must be met:
- **Accuracy**: ≥8.5/10 *(critical - must nail the prompt)*
- **Quality**: ≥8.0/10 *(must be technically sound)*
- **Satisfaction**: ≥8.0/10 *(must be pleasing)*

If any minimum fails, the overall score is capped at 8.4/10, preventing approval and forcing another iteration.

## Modes (Two‑Axis Model)

There are two independent choices — combine them as you like:

- **Input source**
  - Image + Prompt: refine an existing image with instructions *(recommended)*
  - Text‑only: generate from scratch with just a prompt
- **Strategy**
  - Single‑pass: one shot, fast baseline (`--iterations 1`)
  - Smart Iteration: self‑evaluate and refine (`--iterations > 1`)

### Examples
```bash
# Image + Prompt + Smart Iteration (6 rounds)
python enhanced_agent_v2.py "shiny chrome banana" /path/to/BANANA.png --iterations 6

# Image + Prompt + Single‑pass
python enhanced_agent_v2.py "shiny chrome banana" /path/to/BANANA.png --iterations 1

# Text‑only + Smart Iteration
python enhanced_agent_v2.py "shiny chrome banana on black velvet" --iterations 6

# Text‑only + Single‑pass
python enhanced_agent_v2.py "shiny chrome banana on black velvet" --iterations 1
```

- Model: `gemini-2.5-flash-image-preview`
- Max 6 smart iterations
- Strict evaluation: face fidelity, anatomy (fingers/toes/limbs counts), consistency
- Saves results to `current/`

## Quick run (copy‑paste)

```bash
cd <your/project/path>
# Replace with your prompt and your image path
python enhanced_agent_v2.py "your prompt" /path/to/your/image.png
```

## Quick start

```bash
cd <your/project/path>
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export GOOGLE_API_KEY="your_key"
```

## Run it

```bash
# Image-to-image (recommended)
python enhanced_agent_v2.py "your prompt" /path/to/your/image.png

# Text-only (works, but this model prefers a reference image)
python enhanced_agent_v2.py "a serene lake at sunrise"
```

That’s it. Check the `current/` folder for outputs.