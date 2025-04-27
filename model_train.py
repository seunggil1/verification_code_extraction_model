import argparse
import json
import os

from datasets import load_dataset, Features, Value
from huggingface_hub import login
from transformers import AutoTokenizer, Gemma2ForCausalLM
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq


def main(args):
    model_id = "google/gemma-2-2b-it"

    hf_token = os.environ["HF_TOKEN"]
    login(token=hf_token, add_to_git_credential=True)

    features = Features({"sms_body": Value("string"), "code": Value("string")})

    ds = load_dataset(
        "csv",
        data_files={"train": args.train_dataset, "test": args.test_dataset},
        column_names=["sms_body", "code"],
        features=features,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = Gemma2ForCausalLM.from_pretrained(
        model_id,
        device_map="auto",  # 자동으로 GPU↔CPU 파라미터 분산
        offload_folder="offload",  # CPU 오프로딩 폴더
        offload_state_dict=True,
        attn_implementation="eager",
    )
    model.gradient_checkpointing_enable()

    def preprocess(row):
        request = row["sms_body"]
        response = str(row["code"])

        request_template = [{"role": "user", "content": request}]
        response_template = [{"role": "model", "content": response}]

        prompt = (
            tokenizer.apply_chat_template(
                request_template + response_template,
                tokenize=False,
                add_generation_prompt=False,
            )
            + tokenizer.eos_token
        )
        encode_prompt = tokenizer(prompt, truncation=True, max_length=512)

        input_ids = encode_prompt["input_ids"]
        resp_ids = tokenizer(
            response + prompt.split(response)[-1], add_special_tokens=False
        )["input_ids"]

        labels = [-100] * (len(input_ids) - len(resp_ids)) + resp_ids
        labels = [l if l != tokenizer.pad_token_id else -100 for l in labels]

        encode_prompt["labels"] = labels

        return encode_prompt

    ds["train"] = ds["train"].map(preprocess, batched=False)
    ds["test"] = ds["test"].map(preprocess, batched=False)
    # ds['train'] = ds['train'].select(range(10))
    # ds['test'] = ds['test'].select(range(10))

    with open("deepspeed_config.json") as f:
        deep_speed_config = json.load(f)

    training_args = TrainingArguments(
        output_dir=model_id,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        num_train_epochs=2,
        learning_rate=5e-5,
        fp16=True,
        bf16=False,
        eval_strategy="steps",
        save_strategy="steps",
        logging_steps=100,
        save_steps=100,
        save_total_limit=1,
        save_safetensors=True,
        deepspeed=deep_speed_config,
        gradient_checkpointing=True,
        do_train=True,
        do_eval=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(f"{model_id}/final")

    print("Training finished")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--train_dataset", type=str, default="./train.csv")
    argparser.add_argument("--test_dataset", type=str, default="./test.csv")
    parsed_args = argparser.parse_args()
    main(parsed_args)
