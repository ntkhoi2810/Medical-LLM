import argparse
import os
import warnings
from typing import Dict, List

import comet_ml
from comet_ml import Experiment

import torch
import deepspeed
from datasets import load_dataset
from loguru import logger

from unsloth import is_bfloat16_supported

from utils import load_yaml_config
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer, PreTrainedTokenizer

import dotenv

dotenv.load_dotenv()

from huggingface_hub import login

login(os.getenv("HF_TOKEN"))

os.environ["COMET_LOG_ASSETS"] = "True"
experiment = Experiment(
    api_key=os.getenv("COMET_API_KEY"),
    project_name="medqwen3-continual-pretraining",
)


def preprocess_func(examples: Dict[str, List[str]], tokenizer: PreTrainedTokenizer, max_length: int) -> Dict[str, List[int]]:
    """Preprocess the data for the model."""
    texts = examples["text"]
    inputs = tokenizer(
        texts,
        padding = True,
        truncation = True,
        max_length = max_length,
        return_tensors = "pt",
    )
    inputs["labels"] = inputs.input_ids.clone()
    return inputs

def training_pipeline(config: Dict):
    """Training pipeline for continual pretraining."""

    ### MODEL DEFINITION
    model = AutoModelForCausalLM.from_pretrained(
        config["model_args"]["name"],
        torch_dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16,
        # device_map = None,  # Keep model on CPU initially for DeepSpeed
    )
    tokenizer = AutoTokenizer.from_pretrained(config["model_args"]["name"])

    # model = model.to("cuda")

    EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
    tokenizer.pad_token = EOS_TOKEN

    def format_data(examples: Dict) -> Dict:
        """Format the data for the model."""
        # PROMPTING
        medical_prompt = """### Chủ đề: {}

        ### Nội dung:
        {}"""

        titles = examples["title"]
        texts  = examples["content"]
        outputs = []
        for title, text in zip(titles, texts):
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            text = medical_prompt.format(title, text) + EOS_TOKEN
            outputs.append(text)
        return {"text" : outputs}

    ### DATASET
    # Load the dataset
    dataset = load_dataset(config["datasets"]["sources"])
    
    # Format the data
    dataset = dataset.map(format_data, batched=True, remove_columns=["title", "content", "markdown"])
    # Preprocess the data

    dataset = dataset.map(
        lambda examples: preprocess_func(examples, tokenizer, config["model_args"]["max_seq_length"]),
        batched = True,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer = tokenizer,
        mlm = False,
    )

    training_args = TrainingArguments(
        **config["training_args"],
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        deepspeed = "./configs/ds_config_zero3.json",
    )

    trainer = Trainer(
        model = model,
        tokenizer = tokenizer,
        args = training_args,
        train_dataset = dataset['train'],
        data_collator = data_collator,
    )

    with torch.autocast("cuda"):
        trainer.train()

    model.push_to_hub(config["artifacts"]["model_hub_id"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/qwen_cpt_config.yaml")
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    local_rank = args.local_rank

    deepspeed.init_distributed()
    training_pipeline(config)

if __name__ == "__main__":
    main()