import os 
import comet_ml

from dotenv import load_dotenv

load_dotenv()

import torch

from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported

from huggingface_hub import login
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForLanguageModeling

os.environ["COMET_LOG_ASSETS"] = "True"
login(token=os.getenv("HF_TOKEN"))

comet_ml.login(api_key=os.getenv("COMET_API_KEY"), project_name="medqwen3-finetune")

# MODEL & HUGGINGFACE ID
MODEL_NAME = "hf_id/name_model"
MY_HUGGINGFACE_ID = "hf_id/name_repo"

# MODEL CONFIGS
MAX_SEQ_LENGTH = 4096
DTYPE = None
LOAD_IN_4BIT = False
FULL_FINETUNING = True

# MODEL & TOKENIZER
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype = DTYPE,
    load_in_4bit = LOAD_IN_4BIT,
    full_finetuning = FULL_FINETUNING,
)

# DATASET
dataset = load_dataset("codin-research/pretrain-dataset", split="train")

# PROMPTING
medical_prompt = """Bệnh học Việt Nam
### Chủ đề: {}

### Nội dung:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    titles = examples["title"]
    texts  = examples["content"]
    outputs = []
    for title, text in zip(titles, texts):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = medical_prompt.format(title, text) + EOS_TOKEN
        outputs.append(text)
    return {"text" : outputs}


dataset = dataset.map(formatting_prompts_func, batched = True)

# DATA COLLATOR
data_collator = DataCollatorForLanguageModeling(
    tokenizer = tokenizer,
    mlm = False,
)

# TRAINING ARGUMENTS
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    data_collator = data_collator,
    dataset_num_proc = 4,
    packing = False, # Can make training 5x faster for short sequences.
    
    args = TrainingArguments(
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 2,
        
        num_train_epochs = 1,
        warmup_ratio = 0.1,
        
        learning_rate = 2e-5,
        embedding_learning_rate = 1e-5,
        weight_decay = 1e-4,
        logging_steps = 100,

        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        optim = "adamw_torch_fused",
        lr_scheduler_type = "cosine",
        seed = 3407,
        output_dir = "./outputs",
        report_to = ["comet_ml"], # Use this for WandB etc
    ),
)

trainer_stats = trainer.train()

model.push_to_hub(MY_HUGGINGFACE_ID)
tokenizer.push_to_hub(MY_HUGGINGFACE_ID)