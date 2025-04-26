import os

import pandas as pd
from transformers import Gemma2ForCausalLM, AutoTokenizer

if __name__ == '__main__':
    hf_token = os.environ["hf_token"]

    repo_id = "sg2023/Gemma2-2B-IT-Sms-Verification_Code_Extraction"
    tokenizer = AutoTokenizer.from_pretrained(repo_id, token=hf_token)
    tuning_model = Gemma2ForCausalLM.from_pretrained(repo_id, token=hf_token)
    tuning_model.eval()

    df = pd.read_csv("test.csv")

    correct = 0
    incorrect = 0

    for index, row in df.iterrows():
        prompt = row.iloc[0]
        response = str(row.iloc[1])
        request_template = [{"role": "user", "content": prompt}]
        response_template = tokenizer.apply_chat_template(request_template, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(response_template, return_tensors="pt").to(tuning_model.device)
        input_ids = inputs["input_ids"][0]  # Tensor shape: (seq_len,)
        input_len = input_ids.shape[0]

        outputs = tuning_model.generate(**inputs, max_new_tokens=64)[0]
        outputs = outputs[input_len:]
        outputs = tokenizer.decode(outputs, skip_special_tokens=True)

        if response != outputs:
            print(f"Wrong prediction :: \n {prompt} \n\n Expected: {response} / Got: {outputs}")
            incorrect += 1
        else:
            print(f"Correct : {response}")
            correct += 1

    print(f"Correct: {correct} / Incorrect: {incorrect}")
