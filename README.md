# nanobanana-smart-refiner

Make great images without prompt gymnastics. Give it a goal and (ideally) a reference image — it iterates, self-evaluates, tweaks the prompt, and saves the best shots. Easy.

- Model: `gemini-2.5-flash-image-preview`
- Max 6 smart iterations
- Saves results to `current/`

## Quick start

```bash
cd /root/image-agent/enhanced_agent_v2_clean
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export GOOGLE_API_KEY="your_key"
```

## Run it

```bash
# Image-to-image (recommended)
python enhanced_agent_v2.py "make them dance, keep everything else the same" /root/image-agent/multi-agent-image-generator/vintage.png

# Text-only (works, but this model prefers a reference image)
python enhanced_agent_v2.py "a serene lake at sunrise"
```

That’s it. Check the `current/` folder for outputs.