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
  - Smart default: up to 6 rounds with early stop on success (no flag)
  - Exact N: run exactly N rounds (1–10), even if the first shot meets all minimums (`--iterations N`)

### Examples
```bash
# Image + Prompt + Smart default (up to 6, early stop)
python enhanced_agent_v2.py "shiny chrome banana" /path/to/BANANA.png

# Image + Prompt + Single‑pass
python enhanced_agent_v2.py "shiny chrome banana" /path/to/BANANA.png --iterations 1

# Text‑only + Smart default (up to 6, early stop)
python enhanced_agent_v2.py "shiny chrome banana on black velvet"

# Text‑only + Exact 4 rounds (outputs 4 images, prints scores for each)
python enhanced_agent_v2.py "shiny chrome banana on black velvet" --iterations 4

# Text‑only + Exact 7 rounds (outputs 7 images, prints scores for each)
python enhanced_agent_v2.py "shiny chrome banana on black velvet" --iterations 7


## Iteration Behavior

- **Default (smart)**: If you do not pass `--iterations`, the app runs up to 6 rounds and stops early when all minimums are met and overall ≥ 8.5.
- **Exact N (1–10)**: If you pass `--iterations N`, it always runs exactly N rounds and produces N images, even if the first meets all minimums. Each iteration’s scores are printed and the best is highlighted at the end.

### Exact-N Examples (outputs N images + scores)
```bash
# Text-only, exactly 4 images (prints scores for 1..4)
python enhanced_agent_v2.py "chrome banana on black velvet" --iterations 4

# Text-only, exactly 7 images (prints scores for 1..7)
python enhanced_agent_v2.py "chrome banana on black velvet" --iterations 7

# Image + Prompt, exactly 4 images
python enhanced_agent_v2.py "studio portrait, soft key light" /path/to/BANANA.png --iterations 4

# Image + Prompt, exactly 7 images with multiple refs
python enhanced_agent_v2.py "match outfit and lighting" --refs ref1.jpg ref2.png ref3.jpg --iterations 7
```

Result files are saved under `current/` as `iteration_<i>_<session>.png`. The console prints the per-iteration scores, notes, and the final best.

## Multi‑Image References

Use `--refs` (or `-r`) to condition on several images:

```bash
python enhanced_agent_v2.py "golden retriever wearing red bandana" --refs ./dog1.jpg ./dog2.png
```

When multiple references are provided, the generator conditions on all of them and the evaluator compares the output against the set.

## Prompt Helper: Multi‑Image Roles

When using multiple references, map roles to the order of your images and be explicit in the prompt.

### Pattern
- Person A = first reference image (preserve identity: face, hair, overall look)
- Person B = second reference image (preserve identity)
- Describe the action, composition, camera hints, and constraints (anatomy, no duplicates).

### Template
Two dancers performing together in a bright dance studio with a wooden floor. Person A = first reference image (preserve identity: face, hair, general look). Person B = second reference image (preserve identity). Choreography: Person A dances with Person B; Person B is spinning on her feet (pirouette), mid‑turn with arms poised. Show both full body with clear view of feet, natural lighting, sharp focus, balanced exposure. Keep key outfit cues from each reference. Maintain facial fidelity and natural proportions. No extra or merged limbs; five fingers per visible hand; no duplicated body parts; realistic anatomy throughout.

### Run Examples
```bash
# Smart default (up to 6, early stop)
python enhanced_agent_v2.py "Two dancers performing together in a bright dance studio with a wooden floor. Person A = first reference image (preserve identity: face, hair, general look). Person B = second reference image (preserve identity). Choreography: Person A dances with Person B; Person B is spinning on her feet (pirouette), mid‑turn with arms poised. Show both full body with clear view of feet, natural lighting, sharp focus, balanced exposure. Keep key outfit cues from each reference. Maintain facial fidelity and natural proportions. No extra or merged limbs; five fingers per visible hand; no duplicated body parts; realistic anatomy throughout." \
  --refs image1.jpg image2.jpg

# Exact 4 images (prints scores for 1..4)
python enhanced_agent_v2.py "Two dancers performing together in a bright dance studio with a wooden floor. Person A = first reference image (preserve identity: face, hair, general look). Person B = second reference image (preserve identity). Choreography: Person A dances with Person B; Person B is spinning on her feet (pirouette), mid‑turn with arms poised. Show both full body with clear view of feet, natural lighting, sharp focus, balanced exposure. Keep key outfit cues from each reference. Maintain facial fidelity and natural proportions. No extra or merged limbs; five fingers per visible hand; no duplicated body parts; realistic anatomy throughout." \
  --refs image1.jpg image2.jpg --iterations 4
```
# Exact iteration control: passing --iterations N runs exactly N rounds (1–10),
# saving N images and printing per‑iteration metric scores, even if the first meets all minimums.
python enhanced_agent_v2.py "shiny chrome banana" /path/to/BANANA.png --iterations 4
python enhanced_agent_v2.py "shiny chrome banana on black velvet" --iterations 7
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