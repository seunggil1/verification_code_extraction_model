import os

import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

if __name__ == "__main__":
    hf_token = os.environ["hf_token"]

    repo_id = "sg2023/gemma3-270m-it-sms-verification_code_extraction"
    model_path = "D:/model/gemma-3-270m-it/checkpoint-40"
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)
    # tuning_model = Gemma2ForCausalLM.from_pretrained(repo_id, token=hf_token)
    tuning_model = AutoModelForCausalLM.from_pretrained(model_path)
    tuning_model.eval()


    df = pd.read_csv("test.csv", encoding="utf-8", dtype=str)

    correct = 0
    incorrect = 0

    for index, row in df.iterrows():
        prompt = row.iloc[0]
        response = row.iloc[1]
        request_template = [{"role": "user", "content": prompt}]
        response_template = tokenizer.apply_chat_template(
            request_template, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(response_template, return_tensors="pt").to(
            tuning_model.device
        )
        input_ids = inputs["input_ids"][0]  # Tensor shape: (seq_len,)
        input_len = input_ids.shape[0]

        outputs = tuning_model.generate(
            **inputs,
            max_new_tokens=64,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )[0]
        outputs = outputs[input_len:]
        outputs = tokenizer.decode(outputs, skip_special_tokens=True)
        outputs = outputs.strip()
        if response != outputs:
            print(
                f"Wrong prediction :: \n {prompt} \n\n Expected: {response} / Got: {outputs}"
            )
            incorrect += 1
        else:
            print(f"Correct : {response}")
            correct += 1

    print(f"Correct: {correct} / Incorrect: {incorrect}")
