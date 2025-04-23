import json
import os

import torch
from datasets import load_dataset
from huggingface_hub import login
from transformers import AutoTokenizer, Gemma2ForCausalLM
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq

model_id = "google/gemma-2-2b-it"
repo_id = "sg2023/Gemma2-2B-IT-Sms-Verification_Code_Extraction"

hf_token = os.environ["HF_TOKEN"]

login(token=hf_token)

ds = load_dataset(
    "csv",
    data_files={"train": "train.csv", "test": "test.csv"},
    column_names=["sms_body", "code"],
)["train"]

tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=True)

model = Gemma2ForCausalLM.from_pretrained(
    model_id,
    device_map="auto",  # 자동으로 GPU↔CPU 파라미터 분산
    torch_dtype=torch.bfloat16,  # BF16 로드
    offload_folder="offload",  # CPU 오프로딩 폴더
    offload_state_dict=True,
    attn_implementation="eager"
)
with open("deepspeed_config.json") as f:
    deep_speed_config = json.load(f)


def preprocess(row):
    request = row["sms_body"]
    response = str(int(row["code"]))

    request_template = [{"role": "user", "content": request}]
    response_template = [{"role": "model", "content": response}]

    prompt = tokenizer.apply_chat_template(request_template + response_template, tokenize=False, add_generation_prompt=False) + tokenizer.eos_token
    encode_prompt = tokenizer(prompt, truncation=True, max_length=512)

    input_ids = encode_prompt["input_ids"]
    resp_ids = tokenizer(response + prompt.split(response)[-1], add_special_tokens=False)["input_ids"]

    labels = [-100] * (len(input_ids) - len(resp_ids)) + resp_ids
    labels = [l if l != tokenizer.pad_token_id else -100 for l in labels]

    encode_prompt["labels"] = labels

    return encode_prompt


tokenized_ds = ds.map(preprocess, batched=False)

training_args = TrainingArguments(
    output_dir=model_id,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=3,
    learning_rate=5e-5,
    fp16=False,
    bf16=True,
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    deepspeed=deep_speed_config,
    gradient_checkpointing=True
)

data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds,
    data_collator=data_collator
)

trainer.train()
trainer.save_model(f"{model_id}/final")

print("Training finished")

model.push_to_hub(repo_id, use_auth_token=True)
tokenizer.push_to_hub(repo_id, use_auth_token=True)

print("Model pushed to Hugging Face Hub")
