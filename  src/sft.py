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
from transformers import TrainingArguments


comet_ml.login(project_name="unsloth-finetune")


os.environ["COMET_PROJECT_NAME"] = "medqwen3-finetune"  
os.environ["COMET_LOG_ASSETS"] = "True"
login(token=os.getenv("HUGGINGFACE_API_KEY"))

max_seq_length = 2048
dtype = None
load_in_4bit = False
load_in_8bit = False
full_finetuning = True
model_name = "ntkhoi/MedQwen3-4B"
my_huggingface_id = "ntkhoi/MedQwen3-4B-finetuned"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    load_in_8bit = load_in_8bit,
    full_finetuning = full_finetuning,
)


def apply_chat_template(example):
    messages = [
        {'role': 'user', 'content': example['input']},
        {'role': 'assistant', 'content': example['response']}
    ]

    chat_format = tokenizer.apply_chat_template(messages, tokenize=False)
    return {
        'text': chat_format
    }

train_ds = load_dataset("tmnam20/ViMedAQA", split='train')
train_ds = train_ds.map(apply_chat_template, remove_columns=dataset.features)
test_ds = load_dataset("tmnam20/ViMedAQA", split='validation')
test_ds = train_ds.map(apply_chat_template, remove_columns=dataset.features)


trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_ds,
    eval_dataset = test_ds,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = True,
    args = TrainingArguments(
        per_device_train_batch_size = 16,
        gradient_accumulation_steps = 1,
        warmup_ratio = 0.1,
        num_train_epochs = 2,
        save_total_limit=3,
        eval_steps = 100,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 10,
        optim = "adamw_torch_fused",
        weight_decay = 1e-4,
        lr_scheduler_type = "cosine",
        seed = 3407,
        output_dir = "outputs",
        report_to = ["comet_ml"],
    ),
)

trainer_stats = trainer.train()

model.save_pretrained("./trained_model/")

model.push_to_hub(my_huggingface_id)
tokenizer.push_to_hub(my_huggingface_id)