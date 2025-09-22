#!/usr/bin/env python3
"""
Enhanced Image Agent v2.0 - Clean GitHub Version (Standalone)
Uses model: gemini-2.5-flash-image-preview
"""

import os
import sys
import time
import random
import argparse
import re
from PIL import Image
from io import BytesIO
import base64
import google.generativeai as genai

def setup_environment():
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("❌ Error: GOOGLE_API_KEY not found in environment variables")
        print("💡 export GOOGLE_API_KEY='your_key_here'")
        sys.exit(1)
    return api_key

def load_image_as_pil(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    with open(image_path, 'rb') as f:
        img = Image.open(BytesIO(f.read()))
        return img

def save_pil_image(image, filepath):
    out_dir = os.path.dirname(filepath) or 'current'
    os.makedirs(out_dir, exist_ok=True)
    image.save(filepath)
    print(f"💾 Saved: {filepath} ({image.width}x{image.height})")

class EnhancedImageAgentV2:
    def __init__(self, api_key: str = None, max_iterations: int = 6):
        self.api_key = api_key or setup_environment()
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash-image-preview")
        self.max_iterations = max_iterations
        self.target_score = 8.0
        self.session_id = f"{int(time.time())}_{random.randint(10000, 99999)}"
        os.makedirs("current", exist_ok=True)

    def generate_image(self, prompt: str, reference_image: Image.Image = None, max_retries: int = 3):
        for attempt in range(max_retries):
            try:
                contents = [f"Based on this reference image, {prompt}", reference_image] if reference_image else prompt
                response = self.model.generate_content(contents=contents)
                if hasattr(response, 'candidates') and response.candidates:
                    cand = response.candidates[0]
                    if hasattr(cand, 'content') and hasattr(cand.content, 'parts'):
                        for part in cand.content.parts:
                            if hasattr(part, 'inline_data') and part.inline_data is not None:
                                mime = getattr(part.inline_data, 'mime_type', None)
                                data = getattr(part.inline_data, 'data', b"")
                                if not data:
                                    continue
                                if mime and not any(mime.endswith(x) for x in ["png", "jpeg", "jpg", "webp"]):
                                    continue
                                try:
                                    if isinstance(data, (bytes, bytearray)):
                                        image_bytes = bytes(data)
                                    elif isinstance(data, str):
                                        image_bytes = base64.b64decode(data)
                                    else:
                                        image_bytes = bytes(data)
                                    if len(image_bytes) < 100:
                                        continue
                                    img = Image.open(BytesIO(image_bytes))
                                    img.load()
                                    return img
                                except Exception:
                                    try:
                                        rb = data if isinstance(data, (bytes, bytearray)) else bytes(data)
                                        if len(rb) < 100:
                                            continue
                                        img = Image.open(BytesIO(rb))
                                        img.load()
                                        return img
                                    except Exception:
                                        continue
                if hasattr(response, 'candidates') and response.candidates:
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, 'text'):
                            txt = part.text or ""
                            if 'image' in txt.lower():
                                print(f"⚠️ Model returned text instead of pixels: {txt[:100]}...")
                raise Exception("No image data found in response")
            except Exception as e:
                print(f"⚠️ Generation error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(3)
        print("❌ Failed to generate image after all retries")
        if reference_image is None:
            print("ℹ️ Tip: Provide a reference image; this model is optimized for image-to-image.")
        return None

    def evaluate_image(self, image: Image.Image, original_prompt: str, iteration: int = 1) -> dict:
        try:
            eval_prompt = f"""
            Evaluate this generated image against the request: "{original_prompt}"
            Rate 1-10: ACCURACY, QUALITY, SATISFACTION. Return lines:
            ACCURACY: X/10\nQUALITY: X/10\nSATISFACTION: X/10\nOVERALL: X.X/10
            """
            response = self.model.generate_content(contents=[eval_prompt, image])
            eval_text = response.candidates[0].content.parts[0].text
            scores = {}
            for key in ['ACCURACY','QUALITY','SATISFACTION','OVERALL']:
                m = re.search(rf'{key}:\s*(\d+(?:\.\d+)?)', eval_text)
                scores[key.lower()] = float(m.group(1)) if m else 5.0
            scores['feedback'] = eval_text
            scores['iteration'] = iteration
            return scores
        except Exception as e:
            print(f"⚠️ Evaluation error: {e}")
            return {'accuracy':5,'quality':5,'satisfaction':5,'overall':5,'feedback':str(e),'iteration':iteration}

    def improve_prompt(self, current_prompt: str, evaluation: dict, original_goal: str) -> str:
        try:
            improvement_prompt = f"""
            Original goal: {original_goal}
            Current prompt: {current_prompt}
            Scores: A={evaluation.get('accuracy',5)}, Q={evaluation.get('quality',5)}, S={evaluation.get('satisfaction',5)}
            Improve the prompt to better satisfy the goal. Output ONLY the improved prompt.
            """
            resp = self.model.generate_content(improvement_prompt)
            text = resp.candidates[0].content.parts[0].text.strip()
            return text.strip('"\'`').strip() or current_prompt
        except Exception:
            return current_prompt

    def run(self, goal: str, reference_image_path: str = None):
        ref_img = None
        if reference_image_path:
            ref_img = load_image_as_pil(reference_image_path)
            print(f"✅ Loaded image: {reference_image_path} ({ref_img.width}x{ref_img.height})")
        print(f"🎯 Goal: {goal}")
        if reference_image_path:
            print(f"🖼️ Input image: {reference_image_path}")
        print()
        current_prompt = goal
        best_score = 0.0
        best_iter = 0
        for i in range(1, self.max_iterations+1):
            print(f"🔄 ITERATION {i}/{self.max_iterations}")
            print("="*50)
            print(f"📝 Current prompt: {current_prompt[:100]+('...' if len(current_prompt)>100 else '')}")
            print("🎨 Generating image...")
            gen_img = self.generate_image(current_prompt, ref_img)
            if gen_img is None:
                print("❌ Skipping iteration due to generation failure")
                continue
            out_path = f"current/iteration_{i}_{self.session_id}.png"
            save_pil_image(gen_img, out_path)
            print("🔍 Evaluating image...")
            scores = self.evaluate_image(gen_img, goal, i)
            overall = scores.get('overall', (scores['accuracy']+scores['quality']+scores['satisfaction'])/3)
            print(f"📊 Scores - Accuracy: {scores['accuracy']}/10, Quality: {scores['quality']}/10, Satisfaction: {scores['satisfaction']}/10")
            print(f"🎯 Overall Score: {overall:.1f}/10")
            if overall > best_score:
                best_score = overall
                best_iter = i
                print(f"⭐ New best score! (Iteration {i})")
            if overall >= self.target_score:
                print(f"🎉 TARGET ACHIEVED! Score: {overall:.1f}/10")
                break
            if i < self.max_iterations:
                print("🔧 Improving prompt for next iteration...")
                current_prompt = self.improve_prompt(current_prompt, scores, goal)
                print("✨ Enhanced prompt ready\n")
        print("📈 FINAL RESULTS")
        print("="*50)
        print(f"🏆 Best score: {best_score:.1f}/10 (iteration {best_iter})")
        print(f"🎯 Target: {self.target_score}/10")
        print("✅ SUCCESS - Target achieved!" if best_score>=self.target_score else "📊 COMPLETED - Best effort achieved")
        print("📁 All images saved in: current/")
        print(f"🆔 Session ID: {self.session_id}")

def main():
    p = argparse.ArgumentParser(description='Enhanced Image Agent v2.0 (Standalone)')
    p.add_argument('goal', help='Desired transformation or generation goal')
    p.add_argument('image', nargs='?', help='Path to reference image (optional)')
    args = p.parse_args()
    agent = EnhancedImageAgentV2()
    agent.run(args.goal, args.image)

if __name__ == '__main__':
    main()
