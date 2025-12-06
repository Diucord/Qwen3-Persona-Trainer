from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import torch
import re
import os

# =============================================================================
# 1. Model Path and Device Configuration
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
# Model path points to the merged model from finetune.py
MODEL_PATH = BASE_DIR / "finetune" / "universal-persona-tuned" / "merged-model" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Check if model path exists before proceeding
if not os.path.exists(MODEL_PATH):
    print(f"[ERROR] Merged model not found at {MODEL_PATH}. Please run finetune.py first.")
    exit()

# =============================================================================
# 2. Tokenizer and Model Loading
# =============================================================================
try:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        use_fast=True,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    print(f"[INFO] Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"[ERROR] Failed to load model or tokenizer: {e}")
    exit()

# Special token addition for ChatML style if not already in tokenizer config
# These tokens were added during the finetuning process.
special_tokens = {"additional_special_tokens": ["<|user|>", "<|assistant|>"]}
tokenizer.add_special_tokens(special_tokens)
model.resize_token_embeddings(len(tokenizer))

# =============================================================================
# 3. Utility Functions
# =============================================================================
def clean_input(text: str) -> str:
    """Removes unnecessary chat format expressions from input."""
    text = re.sub(r"(?i)^human:\s*", "", text.strip())
    text = re.sub(r"<\|/?(user|assistant)\|>", "", text)
    return text.strip()

# =============================================================================
# 4. Qwen Inference Wrapper Class
# =============================================================================
class QwenLLM:
    def __init__(self):
        self.tokenizer = tokenizer
        self.model = model
        self.device = DEVICE

    def create_chat_completion(self, instruction: str, input_text: str, max_tokens: int = 128) -> str:
        """Generates a response based on the instruction and input."""
        
        # Clean the input text
        cleaned_input = clean_input(input_text)
        user_message = f"{instruction}\n{cleaned_input}" if cleaned_input else instruction
        
        # Construct the Qwen-like ChatML prompt for inference
        prompt = f"<|user|>\n{user_message}\n<|assistant|>\n"

        # Tokenize the input and move to device
        inputs = tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )

        # Decode only the newly generated part of the output
        output_text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        # Clean up any potential remaining special tokens
        return output_text.split("<|")[0].strip()


# =============================================================================
# 5. External Interface and Execution Example
# =============================================================================
llm = QwenLLM()

@dataclass
class ChatConfig:
    """Data class for configuring the chat request."""
    prompt: str
    session_id: str = "default"
    max_tokens: int = 128
    persona: str = "Universal Assistant" # <--- Generic persona name
    language: str = "ko"


def generate_response_llm(config: ChatConfig, user_input: str) -> str:
    """Primary function called by external systems."""
    
    # SYSTEM_PROMPT (instruction) used during finetuning is crucial here
    system_prompt = f"You are a helpful {config.persona} who communicates in {config.language}. Please adhere to the defined persona."
    
    response = llm.create_chat_completion(
        instruction=system_prompt,
        input_text=user_input,
        max_tokens=config.max_tokens
    )
    return response

if __name__ == "__main__":
    # Example usage
    
    # 1. Define the desired persona and configuration
    current_config = ChatConfig(
        prompt="Act as a kind and professional AI researcher.",
        persona="AI Researcher Bot",
        max_tokens=200
    )
    
    user_query = "What is the key advantage of QLoRA over standard LoRA?"
    
    print("=====================================================================")
    print(f"Persona: {current_config.persona} | Query: {user_query}")
    print("=====================================================================")

    # 2. Generate the response
    final_response = generate_response_llm(current_config, user_query)
    
    # 3. Output the result
    print(f"\n[Assistant Response]:\n{final_response}")
    print("=====================================================================")