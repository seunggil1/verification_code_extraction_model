import os

from datasets import load_dataset
from huggingface_hub import login
from transformers import AutoTokenizer, Gemma3ForCausalLM
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq

model_id = "google/gemma-3-1b-it"
repo_id = "sg2023/Gemma3-1B-IT-Sms-Verification_Code_Extraction"
ds = load_dataset(
    "csv",
    data_files={"train": "train.csv", "test": "test.csv"},
    column_names=["sms_body", "code"],
)["train"]
hf_token = os.environ["hf_token"]

model = Gemma3ForCausalLM.from_pretrained(model_id, attn_implementation="eager")

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token


def preprocess(row):
    prompt = row["sms_body"]
    response = str(int(row["code"]))
    enc = tokenizer(
        prompt + tokenizer.eos_token + response + tokenizer.eos_token,
        truncation=True,
        max_length=512
    )
    enc["labels"] = enc["input_ids"].copy()
    return enc


tokenized_ds = ds.map(preprocess, batched=False)

training_args = TrainingArguments(
    output_dir=model_id,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=5e-5,
    fp16=True,
    logging_steps=50,
    save_steps=500,
    save_total_limit=2
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

login(token=hf_token)
model.push_to_hub(repo_id, use_auth_token=True)
tokenizer.push_to_hub(repo_id, use_auth_token=True)

print("Model pushed to Hugging Face Hub")
