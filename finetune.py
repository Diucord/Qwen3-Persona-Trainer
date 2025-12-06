import os
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback
)

# =============================================================================
# 1. Path and Hyperparameter Configuration
# =============================================================================
# Assumes project root is parent.parent of this file (e.g., root/finetune/finetune.py)
BASE_DIR = Path(__file__).resolve().parent.parent 
MODEL_PATH = BASE_DIR / "models" / "qwen3-1.7b-instruct" # Base model path
DATA_PATH = BASE_DIR / "data" / "synthetic_persona_data.jsonl"  # Use generated data
OUTPUT_DIR = BASE_DIR / "finetune" / "universal-persona-tuned" # <--- Generic output folder

# Persona System Prompt (Must match the one used for data generation)
SYSTEM_PROMPT = "You are an artificial intelligence robot that communicates naturally with people. You perform the defined persona according to the user's questions." 

# Training Hyperparameters
MAX_SEQ_LENGTH = 512  # Max sequence length (512-1024 recommended)
LORA_R = 8            # LoRA Rank
LORA_ALPHA = 16       # LoRA Scaling factor
BATCH_SIZE = 4
GRAD_ACCUMULATION = 2
LEARNING_RATE = 5e-5
EPOCHS = 5

print(f"[INFO] Model Path: {MODEL_PATH}")
print(f"[INFO] Output Dir: {OUTPUT_DIR}")

# =============================================================================
# 2. Tokenizer and Model Loading (QLoRA)
# =============================================================================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

# Set padding token (Causal LMs often use EOS token for padding)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id 

# 4-bit quantization configuration for QLoRA (Memory efficiency)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Load Base Model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map="auto"
)
model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False # Disable cache during training

# =============================================================================
# 3. LoRA (PEFT) Configuration
# =============================================================================
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] 
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters() # Print number of trainable parameters

# =============================================================================
# 4. Data Loading and Preprocessing
# =============================================================================
# Load dataset from JSONL file
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Data file not found: {DATA_PATH}. Please run generate_train_dataset.py first.")

data = load_dataset("json", data_files=str(DATA_PATH))["train"]

def preprocess(example): 
    """
    Converts data to ChatML format and tokenizes.
    Masks the input/system part to calculate loss only on the assistant's response.
    """
    instruction = example.get("instruction", "").strip()
    input_text = example.get("input", "").strip()
    output = example.get("output", "").strip()

    # Combine instruction and input
    user_message = f"{instruction}\n{input_text}" if input_text else instruction

    # Construct the full prompt (ChatML style)
    # Note: Using SYSTEM_PROMPT defined globally
    prompt_text = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_message}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    full_text = prompt_text + output + "<|im_end|>"

    # 1. Tokenize the prompt part (for length calculation)
    prompt_ids = tokenizer(
        prompt_text, 
        truncation=True, 
        max_length=MAX_SEQ_LENGTH
    )["input_ids"]

    # 2. Tokenize the full text
    full_tokenized = tokenizer(
        full_text,
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    )

    # 3. Create labels and masking (set prompt part to -100)
    labels = full_tokenized["input_ids"].copy()
    prompt_len = len(prompt_ids)
    
    # Mask the prompt part to exclude it from loss calculation
    labels[:prompt_len] = [-100] * prompt_len
    full_tokenized["labels"] = labels
    
    return full_tokenized

# Apply preprocessing and split data
print("[INFO] Processing dataset...")
tokenized_data = data.map(preprocess, remove_columns=data.column_names)
split = tokenized_data.train_test_split(test_size=0.1, seed=42)
train_data = split["train"]
val_data = split["test"]

# =============================================================================
# 5. Training Setup and Callback
# =============================================================================
class LossPlotCallback(TrainerCallback):
    """Callback to save the Loss graph after training"""
    def __init__(self):
        self.train_losses = []
        self.eval_losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None: return
        if "loss" in logs:
            self.train_losses.append((state.global_step, logs["loss"]))
        if "eval_loss" in logs:
            self.eval_losses.append((state.global_step, logs["eval_loss"]))

    def on_train_end(self, args, state, control, **kwargs):
        # Plotting the graph
        steps_train, losses_train = zip(*self.train_losses) if self.train_losses else ([], [])
        steps_eval, losses_eval = zip(*self.eval_losses) if self.eval_losses else ([], [])

        plt.figure(figsize=(10, 5))
        if steps_train:
            plt.plot(steps_train, losses_train, label="Train Loss")
        if steps_eval:
            plt.plot(steps_eval, losses_eval, label="Eval Loss")
        plt.xlabel("Global Step")
        plt.ylabel("Loss")
        plt.title("Training & Evaluation Loss")
        plt.legend()
        plt.grid()
        output_plot_path = OUTPUT_DIR / "loss_plot.png"
        plt.savefig(str(output_plot_path))
        print(f"[INFO] Loss plot saved to {output_plot_path}")
        plt.close()

args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUMULATION,     
    learning_rate=LEARNING_RATE,              
    num_train_epochs=EPOCHS,               
    logging_steps=10,                
    save_strategy="steps",
    save_steps=50,                  
    eval_strategy="steps",
    eval_steps=50,                  
    weight_decay=0.01,
    warmup_ratio=0.05,
    load_best_model_at_end=True,
    save_total_limit=2,
    fp16=True,
    report_to="none"
)

# =============================================================================
# 6. Training Execution
# =============================================================================
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_data,
    eval_dataset=val_data,
    callbacks=[LossPlotCallback()]
)

print("[INFO] Starting training...")
trainer.train()

# Print evaluation results
metrics = trainer.evaluate()
print("Evaluation metrics:", metrics)

# =============================================================================
# 7. Model Save and Merge
# =============================================================================
# Save LoRA adapter
final_adapter_dir = OUTPUT_DIR / "final-adapter"
model.save_pretrained(str(final_adapter_dir))
tokenizer.save_pretrained(str(final_adapter_dir))
print(f"[INFO] LoRA adapter saved to {final_adapter_dir}")

# Cleanup VRAM (optional, recommended for low VRAM)
del model
del trainer
torch.cuda.empty_cache()

# Merge: Reload base model (FP16 recommended) and merge LoRA weights
print("[INFO] Merging LoRA weights into base model...")
peft_config = PeftConfig.from_pretrained(str(final_adapter_dir))

# Reload base model for merging
base_model = AutoModelForCausalLM.from_pretrained(
    peft_config.base_model_name_or_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Combine adapter
model_to_merge = PeftModel.from_pretrained(base_model, str(final_adapter_dir))
merged_model = model_to_merge.merge_and_unload()

# Save merged model
merged_model_dir = OUTPUT_DIR / "merged-model"
merged_model.save_pretrained(str(merged_model_dir))
tokenizer.save_pretrained(str(merged_model_dir))
print(f"[✅ Complete] Merged model saved to: {merged_model_dir}")