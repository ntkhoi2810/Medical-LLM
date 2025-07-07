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
