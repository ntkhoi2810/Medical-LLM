import yaml
import re
import json
from loguru import logger
import datasets
from datasets import load_dataset, concatenate_datasets

def load_yaml_config(path: str) -> dict:
    """Load a yaml file and return a dictionary."""

    try: 
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading yaml file {path}: {e}")
        raise e

def formatting_prompts_func(examples, tokenizer):
    return {"text" : [example + tokenizer.eos_token for example in examples["text"]]}


def load_and_merge_datasets(config: dict) -> datasets.Dataset:
    """Load and merge datasets."""

    datasets = []
    for dataset_name in config["datasets"]["names"]:
        datasets.append(load_dataset(dataset_name, split="train"))

    return concatenate_datasets(datasets).shuffle(seed=3047)

def apply_chat_template(example, tokenizer):

    messages = [
        {
            "role": "system", 
            "content": """Bạn là trợ lý AI chuyên về giải đáp các câu hỏi y tế bằng tiếng Việt. Mục tiêu của bạn là giúp người dùng hiểu các chủ đề y tế một cách chính xác và an toàn.
            Dưới đây là một hướng dẫn mô tả nhiệm vụ, kèm theo phần thông tin đầu vào để làm rõ ngữ cảnh. Viết một phản hồi phù hợp nhằm hoàn thành yêu cầu."""
        },
        {
            "role": "user",
            "content": str(example['instruction']) + '\n' + example["input"]},
        {
            "role": "assistant",
            "content": example["output"]}
    ]

    chat_format = tokenizer.apply_chat_template(messages, tokenize=False)
    return {
        'text': chat_format
    }

def format_dpo_dataset(example, tokenizer):
    rejected_messages = [
        {
            "role": "user",
            "content": example['question']},
        {
            "role": "assistant",
            "content": example["rejected"]}
    ]

    chosen_messages = [
        {
            "role": "user",
            "content": example['question']},
        {
            "role": "assistant",
            "content": example['chosen']}
    ]
    
    return {
        'rejected': tokenizer.apply_chat_template(rejected_messages, tokenize=False),
        'chosen': tokenizer.apply_chat_template(chosen_messages, tokenize=False)
    }
