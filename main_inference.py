from transformers import Gemma2ForCausalLM, AutoTokenizer
import pandas as pd
import os

hf_token = os.environ["hf_token"]

repo_id = "sg2023/Gemma2-2B-IT-Sms-Verification_Code_Extraction"
tokenizer = AutoTokenizer.from_pretrained(repo_id, token=hf_token)

tuned = Gemma2ForCausalLM.from_pretrained(repo_id, token=hf_token)
tuned.eval()

df = pd.read_csv("test.csv")
for index, row in df.iterrows():
    prompt = row.iloc[0]
    response = str(row.iloc[1])
    inputs = tokenizer(prompt + tokenizer.eos_token, return_tensors="pt").to(tuned.device)
    input_ids = inputs["input_ids"][0]  # Tensor shape: (seq_len,)
    input_len = input_ids.shape[0]

    outputs = tuned.generate(**inputs, max_new_tokens=64)[0]
    outputs = outputs[input_len:]
    outputs = tokenizer.decode(outputs, skip_special_tokens=True)

    if response != outputs:
        print(f"Wrong prediction :: \n {prompt} \n\n Expected: {response} / Got: {outputs}")
    else:
        print(f"Correct : {response}")
