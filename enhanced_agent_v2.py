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
        self.target_score = 8.5  # Raised approval threshold
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

    def evaluate_image(self, image: Image.Image, original_prompt: str, iteration: int = 1, reference_image: Image.Image = None) -> dict:
        try:
            if reference_image is not None:
                eval_prompt = f"""
                You are an extremely strict visual QA evaluator. Assess the FIRST image (Generated) and compare it to the SECOND image (Reference) where applicable.

                Goal: "{original_prompt}"

                Evaluate with 1-10 scores (10 is perfect):
                - FACE_FIDELITY: If a human face is visible, how closely does identity/likeness match the reference (same person, features, proportions)? If no face or not applicable, use 10.
                - ANATOMY: Human anatomy correctness. Explicitly check: number of hands (2), arms (2), legs (2), feet (2), eyes (2), facial symmetry; count fingers per visible hand (5 each, unless occluded), toes per visible foot (5). Penalize extra/missing/merged digits, warped limbs, or distortions.
                - CONSISTENCY: Consistency of key attributes (clothing, hair color/length, accessories) vs reference when they should remain.
                - ACCURACY: How well it fulfills the goal prompt changes.
                - QUALITY: Technical quality (clarity, composition, lighting, artifacts).
                - SATISFACTION: Overall end-user satisfaction.

                Return ONLY these lines in this exact order:
                FACE_FIDELITY: X/10
                ANATOMY: X/10
                CONSISTENCY: X/10
                ACCURACY: X/10
                QUALITY: X/10
                SATISFACTION: X/10
                NOTES: <one short sentence about any issues found>
                """
                contents = [eval_prompt, image, reference_image]
            else:
                eval_prompt = f"""
                You are an extremely strict visual QA evaluator. Assess the image.

                Goal: "{original_prompt}"

                Evaluate with 1-10 scores (10 is perfect):
                - FACE_FIDELITY: If a human face is visible, ensure identity consistency across the image and natural facial proportions. If not applicable, use 10.
                - ANATOMY: Human anatomy correctness. Explicitly check: number of hands (2), arms (2), legs (2), feet (2), eyes (2); count fingers per visible hand (5), toes per visible foot (5). Penalize extra/missing/merged digits, warped limbs, or distortions.
                - CONSISTENCY: Internal consistency (no duplicated limbs, no morphing items).
                - ACCURACY: How well it fulfills the goal prompt.
                - QUALITY: Technical quality (clarity, composition, lighting, artifacts).
                - SATISFACTION: Overall end-user satisfaction.

                Return ONLY these lines in this exact order:
                FACE_FIDELITY: X/10
                ANATOMY: X/10
                CONSISTENCY: X/10
                ACCURACY: X/10
                QUALITY: X/10
                SATISFACTION: X/10
                NOTES: <one short sentence about any issues found>
                """
                contents = [eval_prompt, image]

            response = self.model.generate_content(contents=contents)
            eval_text = response.candidates[0].content.parts[0].text

            # Parse scores
            keys = ['FACE_FIDELITY','ANATOMY','CONSISTENCY','ACCURACY','QUALITY','SATISFACTION']
            scores = {}
            for key in keys:
                m = re.search(rf'{key}:\s*(\d+(?:\.\d+)?)', eval_text, re.IGNORECASE)
                scores[key.lower()] = float(m.group(1)) if m else (10.0 if key == 'FACE_FIDELITY' and reference_image is None else 5.0)

            # Apply minimum thresholds
            min_thresholds = {
                'accuracy': 8.5,
                'quality': 8.0,
                'satisfaction': 8.0
            }
            
            # Check if key criteria meet minimums
            meets_minimums = True
            for metric, min_val in min_thresholds.items():
                if scores.get(metric, 0) < min_val:
                    meets_minimums = False
                    break

            # Weighted overall (prioritize fidelity + anatomy)
            w = {
                'face_fidelity': 0.35,
                'anatomy': 0.35,
                'consistency': 0.05,
                'accuracy': 0.15,
                'quality': 0.07,
                'satisfaction': 0.03,
            }
            weighted_overall = (
                scores['face_fidelity'] * w['face_fidelity'] +
                scores['anatomy'] * w['anatomy'] +
                scores['consistency'] * w['consistency'] +
                scores['accuracy'] * w['accuracy'] +
                scores['quality'] * w['quality'] +
                scores['satisfaction'] * w['satisfaction']
            )
            
            # If minimum thresholds not met, cap overall at 8.4 (below approval)
            if not meets_minimums:
                weighted_overall = min(weighted_overall, 8.4)

            # Extract brief notes
            m_notes = re.search(r'NOTES:\s*(.+)', eval_text, re.IGNORECASE)
            notes = m_notes.group(1).strip() if m_notes else ''

            scores['overall'] = round(weighted_overall, 2)
            scores['feedback'] = notes or eval_text
            scores['iteration'] = iteration
            return scores
        except Exception as e:
            print(f"⚠️ Evaluation error: {e}")
            return {
                'face_fidelity': 10.0 if reference_image is None else 5.0,
                'anatomy': 5.0,
                'consistency': 5.0,
                'accuracy': 5.0,
                'quality': 5.0,
                'satisfaction': 5.0,
                'overall': 5.0,
                'feedback': str(e),
                'iteration': iteration
            }

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
            scores = self.evaluate_image(gen_img, goal, i, ref_img)
            overall = scores.get('overall', 5.0)
            print(
                "📊 Scores - Face: {face}/10, Anatomy: {anat}/10, Consistency: {cons}/10, Accuracy: {acc}/10, Quality: {qual}/10, Satisfaction: {sat}/10".format(
                    face=scores.get('face_fidelity', 0), anat=scores.get('anatomy', 0), cons=scores.get('consistency', 0),
                    acc=scores.get('accuracy', 0), qual=scores.get('quality', 0), sat=scores.get('satisfaction', 0)
                )
            )
            print(f"🧠 Notes: {scores.get('feedback','').strip()[:180]}")
            print(f"🎯 Overall (weighted): {overall:.1f}/10")
            
            # Check minimum requirements
            acc_ok = scores.get('accuracy', 0) >= 8.5
            qual_ok = scores.get('quality', 0) >= 8.0 
            sat_ok = scores.get('satisfaction', 0) >= 8.0
            
            if not (acc_ok and qual_ok and sat_ok):
                missing = []
                if not acc_ok: missing.append(f"Accuracy<8.5 ({scores.get('accuracy', 0):.1f})")
                if not qual_ok: missing.append(f"Quality<8.0 ({scores.get('quality', 0):.1f})")
                if not sat_ok: missing.append(f"Satisfaction<8.0 ({scores.get('satisfaction', 0):.1f})")
                print(f"❌ Minimum thresholds not met: {', '.join(missing)}")
            
            if overall > best_score:
                best_score = overall
                best_iter = i
                print(f"⭐ New best score! (Iteration {i})")
            if overall >= self.target_score and acc_ok and qual_ok and sat_ok:
                print(f"🎉 TARGET ACHIEVED! Score: {overall:.1f}/10 (All minimums met)")
                break
            elif overall >= self.target_score:
                print(f"⚠️ Score {overall:.1f}/10 reached, but minimum thresholds not met")
            if i < self.max_iterations:
                print("🔧 Improving prompt for next iteration...")
                current_prompt = self.improve_prompt(current_prompt, scores, goal)
                print("✨ Enhanced prompt ready\n")
        print("📈 FINAL RESULTS")
        print("="*50)
        print(f"🏆 Best score: {best_score:.1f}/10 (iteration {best_iter})")
        print(f"🎯 Target: {self.target_score}/10 + minimums (Accuracy≥8.5, Quality≥8.0, Satisfaction≥8.0)")
        print("✅ SUCCESS - Target achieved!" if best_score>=self.target_score else "📊 COMPLETED - Best effort achieved")
        print("📁 All images saved in: current/")
        print(f"🆔 Session ID: {self.session_id}")

def main():
    p = argparse.ArgumentParser(description='Enhanced Image Agent v2.0 (Standalone)')
    p.add_argument('goal', help='Desired transformation or generation goal')
    p.add_argument('image', nargs='?', help='Path to reference image (optional)')
    p.add_argument('--iterations', '-n', type=int, default=6,
                   help='Number of iterations: 1 for single-pass, >1 for smart iteration (default: 6)')
    args = p.parse_args()
    max_iters = max(1, int(args.iterations))
    agent = EnhancedImageAgentV2(max_iterations=max_iters)
    agent.run(args.goal, args.image)

if __name__ == '__main__':
    main()
