import os

from huggingface_hub import login, create_tag
from transformers import Gemma2ForCausalLM, AutoTokenizer

if __name__ == "__main__":
    hf_token = os.environ["hf_token"]

    login(token=hf_token, add_to_git_credential=True)

    local_model_path = "D:/model/checkpoint-398"
    repo_id = "sg2023/Gemma2-2B-IT-Sms-Verification_Code_Extraction"

    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    trained_model = Gemma2ForCausalLM.from_pretrained(local_model_path)

    trained_model.push_to_hub(repo_id, use_auth_token=True)
    tokenizer.push_to_hub(repo_id, use_auth_token=True)

    # create_tag(repo_id, token=hf_token, revision="main", tag="v0.1.3")
