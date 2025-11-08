import os

from huggingface_hub import login, create_tag
from transformers import Gemma3ForCausalLM, AutoTokenizer

if __name__ == "__main__":
    hf_token = os.environ["hf_token"]

    login(token=hf_token, add_to_git_credential=True)

    local_model_path = "D:/model/gemma-3-270m-it/checkpoint-40"
    repo_id = "sg2023/gemma3-270m-it-sms-verification_code_extraction"

    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    trained_model = Gemma3ForCausalLM.from_pretrained(local_model_path)

    trained_model.push_to_hub(repo_id, token=True)
    tokenizer.push_to_hub(repo_id, token=True)

    create_tag(repo_id, token=hf_token, revision="main", tag="v1.0.0")
