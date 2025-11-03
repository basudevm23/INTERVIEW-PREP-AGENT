import os
import re
import torch
from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

# ====== Env config ======
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
LLM_LOCAL_MODEL = os.getenv("LLM_LOCAL_MODEL", "./tiny_llama")
LORA_MODEL_DIR = os.getenv("LORA_MODEL_DIR", "./adapters/checkpoint-1")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ====== Gemini client ======
class GeminiClient:
    def __init__(self):
        self.available = bool(GEMINI_API_KEY)
        if self.available:
            self.model = genai.GenerativeModel(GEMINI_MODEL)

    def ask(self, prompt):
        """Safely call Gemini and handle missing parts or invalid finish reasons."""
        if not self.available:
            return "⚠️ Gemini not configured — using fallback."

        try:
            response = self.model.generate_content(prompt)
            # Ensure response has usable text
            if hasattr(response, "text") and response.text:
                return response.text.strip()
            elif hasattr(response, "candidates") and response.candidates:
                parts = response.candidates[0].content.parts
                if parts and hasattr(parts[0], "text"):
                    return parts[0].text.strip()
            return "⚠️ Gemini returned no text output. Try rephrasing the prompt."
        except Exception as e:
            return f"⚠️ Gemini error: {e}"

# ====== Local LoRA model ======


class LocalLoRAModel:
    def __init__(self, base_dir, adapter_dir):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(base_dir)
            base_model = AutoModelForCausalLM.from_pretrained(
                base_dir,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            self.model = PeftModel.from_pretrained(base_model, adapter_dir)
            self.model = self.model.merge_and_unload()  # merge LoRA weights
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load LoRA: {e}")

    def score_answer(self, question, answer):
        try:
            prompt = f"Question: {question}\nAnswer: {answer}\nScore this answer from 1 to 10."
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(**inputs, max_new_tokens=50)
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return self._extract_score(result)
        except Exception as e:
            return f"Scoring error: {e}"

    def _extract_score(self, text):
        import re
        match = re.search(r"(\d+(\.\d+)?)", text)
        return float(match.group(1)) if match else "N/A"
