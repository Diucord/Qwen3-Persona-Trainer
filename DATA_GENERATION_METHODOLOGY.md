# Dataset Construction Methodology: Advanced Persona-Instruct Data Synthesis

---
## 1. Overview and Objectives
The **Persona-Instruct Dataset (PID)** is designed to effectively inject a **Custom Persona** into an LLM. We utilized the conceptual framework of **Gemini Advanced Reasoning and Multi-turn Context Awareness** to synthesize and refine high-quality training data, ensuring structural integrity and high conversational fidelity.

---
## 2. Methodology: Human-Guided Heuristic Curation & LLM Expansion

### 2.1. Scenario-Driven Initial Design (Human-Guided Curation)
The initial dataset seeds were defined through a **Human-Guided Heuristic** approach, focusing on covering critical conversational **Edge Cases** relevant to an AI assistant's deployment (e.g., HRI scenarios).

| Taxonomy | Objective |
| :--- | :--- |
| **Identity & Integrity** | Maintain consistency regarding the **Target Persona's** identity, affiliation, and technical limitations. |
| **Multi-Lingual Robustness** | Ensure consistent performance across various language environments (Korean, English, etc.). |
| **Operational & Safety** | Define polite refusal and safety guidelines for handling dangerous or unfulfillable requests. |
| **Emotional/Social Context** | Model empathy and social awareness when responding to emotional user input. |

### 2.2. LLM-Based Data Synthesis Pipeline (Mocking Gemini API)
The data generation process is structured into a rigorous 3-step synthesis and refinement pipeline using Python, conceptually mirroring advanced LLM API capabilities.

#### Step 1: Scenario Augmentation & Diversity Injection (Expansion)
1.  **System Instruction Enforcement:** The **Persona Blueprint** is conceptually enforced via a `SYSTEM_PROMPT` to guide the model's output tone and behavior during synthesis.
2.  **Context and Language Expansion:** Initial seeds are expanded by directing the LLM to generate two key variations: **Multilingual Augmentation** (simulating diverse language queries) and **Contextual Complexity** (simulating multi-turn conversation summarization into single `input` fields).

#### Step 2: Persona Consistency Enforcement and Refinement (Self-Correction)
1.  **Self-Correction Loop Implementation:** The model's initial answer ($A_1$) is conceptually put through a **Self-Correction Loop**. The model is instructed to evaluate $A_1$ against the **Persona Blueprint** and technical constraints, generating a refined answer ($A_2$) if misalignment is detected (e.g., safety protocol violation).
2.  **Schema and Format Validation:** The final output is constrained to the strict `{"instruction": ..., "input": ..., "output": ...}` JSON schema, ensuring structural integrity for the downstream fine-tuning process.

#### Step 3: Automated Quality Filtering
Each synthesized data pair is assigned a conceptual **Confidence Score**. Data below the **Confidence Threshold $T$** (e.g., 0.90) is automatically filtered out, ensuring the final training dataset (SFT Fidelity) remains exceptionally high quality and aligned with the target persona.

---
## 3. Conclusion
The PID dataset is a result of combining **human expertise** with **advanced LLM synthesis capabilities**. This ensures that the fine-tuned Qwen3 model maintains the defined persona consistently, safely, and robustly across multilingual and complex conversational contexts.