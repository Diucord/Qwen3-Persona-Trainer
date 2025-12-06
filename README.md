# Qwen3-Persona-Trainer: Universal Persona Injection Pipeline

> This framework provides an integrated pipeline for **efficiently injecting any custom persona** into the **Qwen3 (Qwen3-1.7B-Instruct)** model using the **QLoRA technique**.

> It emphasizes **Human-Guided High-Quality Data Synthesis** to achieve strong persona alignment with minimal data. **(This project is a universal model training pipeline, independent of any specific product or service.)**

---
## End-to-End Workflow
The project follows a rigorous three-stage pipeline:

### 1.  Data Generation
Create a high-fidelity synthetic dataset using `generate_train_dataset.py` (mocking Gemini API logic).

### 2.  Model Training
Inject the persona using the memory-efficient QLoRA method via `finetune.py`. 

### 3.  Model Inference
Load the merged model for real-time conversational inference using `llm_chat.py`.

---
## Directory Structure

```bash
root/
├── models/
│   └── qwen3-1.7b-instruct/           # Base model files
├── data/
│   ├── synthetic_persona_data.jsonl   # Generated dataset
│   ├── train.json                     # Sample training data
│   └── labeling.json                  # Sample labeling guide
├── finetune/
│   ├── universal-persona-tuned/       # Training outputs and merged model
│   └── finetune.py                    # QLoRA Training script
├── generate_train_dataset.py          # Data synthesis script
└── llm_chat.py                        # Inference/Chat interface
```

---
## Requirements & Setup
Recommend a Python 3.10+ environment with CUDA support.
```bash
# Install core dependencies
pip install torch torchvision torchaudio
pip install transformers peft datasets accelerate bitsandbytes matplotlib
```

### 1. Data Preparation
Run the synthesis script to generate the high-quality instruction dataset. 
Refer to DATA_GENERATION_METHODOLOGY.md for the technical details of the data construction.
```bash
python generate_train_dataset.py
```

### 2. Model Training
Execute the fine-tuning script. This will save the LoRA adapter and then merge it into the base model, placing the final artifacts in the universal-persona-tuned/merged-model directory.
```bash
python finetune.py
```

### 3. Model Inference
Use the chat script to test the conversational abilities of the fine-tuned model.
```bash
python llm_chat.py
```

---
## Author
- Seyoon Oh
- Korea University | Industrial and Management Engineering
- Contact : osy7336@korea.ac.kr