# nanobanana-smart-refiner

Make great images without prompt gymnastics. Give it a goal and (ideally) a reference image — it iterates, self-evaluates, tweaks the prompt, and saves the best shots. Easy.

- Model: `gemini-2.5-flash-image-preview`
- Max 6 smart iterations
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