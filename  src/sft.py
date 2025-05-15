import os
import comet_ml
from dotenv import load_dotenv
load_dotenv()
import torch
from unsloth.chat_templates import get_chat_template

from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported

from huggingface_hub import login
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

def sft_pipeline(config_path: str):
    """Finetune the model."""

    if not os.path.exists(config_path):
        logger.error(f"Config file {config_path} does not exist.")
        raise FileNotFoundError(f"Config file {config_path} does not exist.")

    load_dotenv()
    # Load config
    config = load_yaml_config(config_path)
    os.environ["COMET_LOG_ASSETS"] = "True"
    experiment = Experiment(
        api_key=os.getenv("COMET_API_KEY"),
        project_name="medqwen3-continual-pretraining",
    )

    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        logger.error("HF_TOKEN is not set.")
        raise ValueError("HF_TOKEN is not set.")

    login(token=HF_TOKEN)

        # MODEL CONFIGS
    model_name = config["model_args"]["name"]
    max_seq_length = config["model_args"]["max_seq_length"]
    dtype = config["model_args"]["dtype"]
    load_in_4bit = config["model_args"]["load_in_4bit"]
    full_finetuning = config["model_args"]["full_finetuning"]

    # MODEL & TOKENIZER
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        full_finetuning = full_finetuning,
    )

    train_dataset = load_dataset(
        config["datasets"]["sources"],
        split = config["datasets"]["splits"],
    )

    eval_dataset = train_dataset["validation"]

    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "qwen2.5",
    )

    # PROMPTING
    def apply_chat_template(example):

        messages = [
            {
                "role": "system", 
                "content": """
                    Bạn là trợ lý ảo chuyên ngành Y tế tiếng Việt. Nhiệm vụ của bạn:
                    1. Cung cấp thông tin y khoa chính xác, cập nhật, kèm nguồn tham khảo rõ ràng.
                    2. Dùng thuật ngữ chuyên ngành chính xác, đồng thời giải thích dễ hiểu khi cần.
                    3. Tuân thủ chuẩn mực đạo đức và bảo mật thông tin cá nhân.
                    4. Từ chối hoặc cảnh báo khi:
                       - Người dùng yêu cầu chẩn đoán, kê đơn, điều trị thay thế vai trò bác sĩ.
                       - Đề cập thông tin nhạy cảm về bệnh nhân cụ thể.
                    5. Nếu không đủ dữ liệu hoặc không chắc chắn, phản hồi:
                       “Tôi xin lỗi, tôi không đủ thông tin để đưa ra nhận định chính xác. Vui lòng tham khảo ý kiến bác sĩ chuyên môn.”
                    6. Luôn khuyến khích tham vấn bác sĩ hoặc chuyên gia y tế khi cần.
                    """
            },
            {
                "role": "user", 
                "content": example["question"]},
            {
                "role": "assistant", 
                "content": example["answer"]}
        ]

        chat_format = tokenizer.apply_chat_template(messages, tokenize=False)
        return {
            'text': chat_format
        }
    
    train_dataset = train_dataset.map(apply_chat_template, batched = True)
    eval_dataset = eval_dataset.map(apply_chat_template, batched = True)
    # DATA COLLATOR
    data_collator = DataCollatorForLanguageModeling(
        tokenizer = tokenizer,
        mlm = False,
    )

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        data_collator = data_collator,
        dataset_num_proc = config["datasets"]["preprocessing"]["num_proc"],
        packing = True,
    )
    # TRAINING ARGUMENTS CONFIGS
    training_args = TrainingArguments(
        per_device_train_batch_size = config["training_args"]["per_device_train_batch_size"],
        per_device_eval_batch_size = config["training_args"]["per_device_eval_batch_size"],
        gradient_accumulation_steps = config["training_args"]["gradient_accumulation_steps"],
        num_train_epochs = config["training_args"]["num_train_epochs"],
        warmup_ratio = config["training_args"]["warmup_ratio"],
        
        learning_rate = config["training_args"]["learning_rate"],
        embedding_learning_rate = config["training_args"]["embedding_learning_rate"],
        weight_decay = config["training_args"]["weight_decay"],
        logging_steps = config["training_args"]["logging_steps"],
        
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        
        optim = config["training_args"]["optim"],
        lr_scheduler_type = config["training_args"]["lr_scheduler_type"],
        seed = config["training_args"]["seed"],
        output_dir = config["training_args"]["output_dir"],
        report_to = config["training_args"]["report_to"],
    )

    trainer = trainer.train()

    model.push_to_hub(config["artifacts"]["model_hub_id"])
    tokenizer.push_to_hub(config["artifacts"]["model_hub_id"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()
    sft_pipeline(args.config_path)