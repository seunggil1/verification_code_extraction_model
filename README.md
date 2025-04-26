## Verification code extraction model
- SMS에서 인증번호를 추출 하기 위한 Gemma2 SFT 모델

- `"본인인증번호는 315611 입니다. 정확히 입력해주세요."` -> `315611`
- `"안녕하세요"` -> `0`

```python
from transformers import Gemma2ForCausalLM, AutoTokenizer

repo_id = "sg2023/Gemma2-2B-IT-Sms-Verification_Code_Extraction"
tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = Gemma2ForCausalLM.from_pretrained(repo_id)
model.eval()

prompt = "본인인증번호는 315611 입니다. 정확히 입력해주세요."
request_template = [{"role": "user", "content": prompt}]
response_template = tokenizer.apply_chat_template(request_template, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(response_template, return_tensors="pt").to(model.device)
input_ids = inputs["input_ids"][0]  # Tensor shape: (seq_len,)
input_len = input_ids.shape[0]

outputs = model.generate(**inputs, max_new_tokens=64)[0]
outputs = outputs[input_len:]
outputs = tokenizer.decode(outputs, skip_special_tokens=True)

print(outputs)  # 315611
```


### Huggingface Repo
  - https://huggingface.co/sg2023/Gemma2-2B-IT-Sms-Verification_Code_Extraction
  - Quantization for mobile
    - 모바일 앱에서 사용하기 위한 Quantization 모델
    - https://huggingface.co/sg2023/Gemma2-2B-IT-Sms-Verification_Code_Extraction-int8-tflite

### Install

```shell
pre-commit install

pip install -r requirements.txt
pip install -r requirements_train.txt (required cuda gpu)
```

### Development

#### 1. 환경 변수 설정
- `HF_TOKEN` : Huggingface Token, Gemma2 모델 다운로드, 모델 결과 업로드를 위한 인증 토큰
  - Huggingface에서 Gemma2 권한을 부여 받아야 합니다.
- `OPENAI_API_KEY` : OpenAI API Key, OpenAI API 사용을 위한 인증 키

```shell
export HF_TOKEN=hf_~
export OPENAI_API_KEY=sk-proj-~
```

#### 2. Dataset 준비
- SMS 본문 / 인증번호 데이터셋을 만들기 위해, GPT4o API를 사용하여 SMS 본문과 인증번호 쌍을 생성
  - Body 칼럼에 문자 내용이 기록된 .csv 파일이 필요

| Body                                                  |
|-------------------------------------------------------|
| 본인확인 인증번호 [594757] 입니다 . " 타인 노출 금지" |
| 음식 배달입니다 . 문 앞에 두고 갑니다                 |


```shell
python dataset_make_lable.py \
    --input_csv_file_path sms.csv \
    --output_csv_file_path sms_result.csv
```

#### 3. train / test 데이터 분리
- train / test 데이터 분리
  - train.csv, test.csv 파일 생성

```shell
python dataset_split.py \
    --input_csv_file_path sms_result.csv \
    --output_csv_file_dir .
```


#### 4. 모델 학습
- a100 GPU 1대에서 4~5 시간 정도 소요됩니다.

```shell
python model_train.py \
    --train_dataset train.csv \
    --test_dataset test.csv
```

#### 5. 모델 업로드
- 실행전 코드에서 경로 수정 필요
```shell
python model_upload.py
```

#### 6. 모델 평가
- 실행전 코드에서 경로 수정 필요

```shell
python model_inference.py
```


#### 7. quantization (Only Linux)
```shell
# source .venv/bin/activate
pip install --upgrade "jax==0.4.34" "jaxlib==0.4.34" ai_edge_torch==0.4.* mediapipe==0.10.* sentencepiece --no-cache-dir

python -m ai_edge_torch.generative.examples.gemma.convert_gemma2_to_tflite \
--checkpoint_path ~/.cache/huggingface/hub/models--sg2023--Gemma2-2B-IT-Sms-Verification_Code_Extraction/모델_경로 \
--output_path ./export \
--output_name_prefix gemma2_sft \
--kv_cache_max_len 1024 \
--quantize
```

#### 8. 모델 변환 (Only Linux)
- 모바일용 .task 파일 생성
- 실행전 코드에서 경로 수정 필요

```shell
python model_task_convert.py
```
