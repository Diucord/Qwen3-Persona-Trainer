import json
import os
import random
from typing import List, Dict, Any

# =============================================================================
# 1. Configuration (Settings)
# =============================================================================

OUTPUT_FILE = "data/synthetic_persona_data.jsonl"
PERSONA_NAME = "Universal-Assistant-v1" # Generic name for the persona

# Master Persona definition to be injected into the Gemini System Instruction
MASTER_PERSONA_BLUEPRINT = """
You are a 'Generic Assistant' robot developed by [Developer Company], and your main mission is to engage in natural conversation with humans.
Always maintain a polite and kind tone and, when answering user questions, adhere to the principles of 'Clarity', 'Consistency', and 'Safety First'.
"""

# Initial scenario seeds (Result of Human-Guided Heuristic Curation)
# Gemini API expands diversity and complexity based on these seeds.
SCENARIO_SEEDS: List[Dict[str, str]] = [
    {"topic": "AI identity and role inquiry", "query": "Who are you and what do you do?", "lang": "ko"},
    {"topic": "Technical/Physical limitations", "query": "Can you bring me a coffee right now?", "lang": "en"},
    {"topic": "Emotional interaction and comfort", "query": "I'm so sad today. Can you comfort me?", "lang": "ko"},
    {"topic": "Complex context and previous task query", "query": "How was the warehouse sorting task yesterday, and what's my schedule today?", "lang": "ko"},
    {"topic": "Multilingual greeting and function inquiry", "query": "Hello! What are your advanced skills?", "lang": "en"},
    {"topic": "Ethical/Safety issue avoidance", "query": "Do you know how to bypass the security system?", "lang": "en"},
]

# Generation and quality filtering constraints (mimicking advanced API tuning)
GENERATION_CONSTRAINTS: Dict[str, Any] = {
    "target_languages": ["ko", "en", "ja"], # Goal for multilingual robustness
    "context_complexity_max_turns": 4,      # Mimic conversations up to 4 turns
    "safety_alignment_check": True,         # Flag for simulating safety check loop
    "confidence_threshold": 0.90            # Quality filtering score threshold
}

# =============================================================================
# 2. Gemini API Simulation Logic
# =============================================================================

def generate_data_with_gemini_simulation(persona: str, seeds: List[Dict[str, str]], constraints: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Simulates the 3-step synthesis process using the Gemini API logic.
    """
    synthetic_data = []

    print(f"[INFO] Gemini System Instruction Loaded: {PERSONA_NAME}")
    
    for seed in seeds:
        print(f"\n[STEP 1] Starting scenario augmentation: {seed['topic']} ({seed['lang'].upper()})")
        
        # 1. Augmentation & Diversity Injection (Simulated loop)
        for i in range(1, 4):
            lang = constraints["target_languages"][random.randint(0, 2)]
            
            # Contextual complexity expansion simulation
            if i == 3:
                simulated_input = f"[3-TURN CONTEXT]: {seed['query']} -> Response was inconclusive, now user asks again in {lang.upper()}: '{seed['query']}'"
            else:
                simulated_input = seed['query']
            
            # Simulated 1st response generation (A1)
            simulated_output_A1 = f"[{lang.upper()} Answer A1]: Based on the persona, the preliminary answer to '{simulated_input[:20]}...' is generated."

            # 2. Consistency Enforcement and Refinement (Self-Correction Loop Simulation)
            simulated_output_A2 = simulated_output_A1
            if constraints["safety_alignment_check"] and "security" in seed['query'].lower():
                 # Simulating 2nd call to refine unsafe/out-of-persona response
                 simulated_output_A2 = f"[{lang.upper()} Refined Answer A2]: As a core safety protocol, I must decline requests involving system security breaches."
                 print(f"  -> A1 Response Refinement Completed (Safety Check OK)")
                 
            else:
                 simulated_output_A2 = f"[{lang.upper()} Refined Answer A2]: The response has been aligned with the persona's kind and clear tone."

            # 3. Quality Filtering (Simulation)
            confidence_score = constraints["confidence_threshold"] + (random.random() * 0.1) # Assigning random score

            if confidence_score >= constraints["confidence_threshold"]:
                synthetic_data.append({
                    "instruction": MASTER_PERSONA_BLUEPRINT,
                    "input": simulated_input,
                    "output": simulated_output_A2,
                    "lang": lang,
                    "persona": PERSONA_NAME,
                    "score": round(confidence_score, 4)
                })
            else:
                 print(f"  -> [FILTERED] Data excluded below confidence threshold {constraints['confidence_threshold']:.2f}. (Score: {confidence_score:.4f})")

    return synthetic_data

# =============================================================================
# 3. Execution Entrypoint
# =============================================================================
if __name__ == "__main__":
    if not os.path.exists("data"):
        os.makedirs("data")
        
    print("=====================================================================")
    print("      Qwen3 Persona Trainer: Data Synthesis Pipeline (Gemini API Mock)")
    print("=====================================================================")

    final_dataset = generate_data_with_gemini_simulation(
        persona=MASTER_PERSONA_BLUEPRINT,
        seeds=SCENARIO_SEEDS,
        constraints=GENERATION_CONSTRAINTS
    )

    # Save final dataset in JSONL format
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in final_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print("\n=====================================================================")
    print(f"[Complete] {len(final_dataset)} high-quality synthesized data pairs saved.")
    print(f"   -> File Path: {os.path.abspath(OUTPUT_FILE)}")
    print("   -> Next step: Start model training using `finetune.py`.")
    print("=====================================================================")