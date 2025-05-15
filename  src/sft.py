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

os.environ["COMET_API_KEY"] = os.getenv("COMET_API_KEY")
os.environ["COMET_PROJECT_NAME"] = "medqwen3-finetune"  
os.environ["COMET_LOG_ASSETS"] = "True"
login(token=os.getenv("HUGGINGFACE_API_KEY"))

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

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "qwen2.5",
)

# CHAT TEMPLATE
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

# DATASET
train_ds = load_dataset("tmnam20/ViMedAQA", split='train')
train_ds = train_ds.map(apply_chat_template, remove_columns=train_ds.features)
test_ds = load_dataset("tmnam20/ViMedAQA", split='validation')
test_ds = test_ds.map(apply_chat_template, remove_columns=test_ds.features)

# CONFIGURE TRAINER
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
        per_device_train_batch_size = 32,
        gradient_accumulation_steps = 2,
        warmup_ratio = 0.1,
        num_train_epochs = 1,
        # save_total_limit=3,
        eval_steps = 100,
        learning_rate = 2e-5,
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

# TRAINING
trainer_stats = trainer.train()

model.save_pretrained("./trained_model/")

model.push_to_hub(MY_HUGGINGFACE_ID)
tokenizer.push_to_hub(MY_HUGGINGFACE_ID)