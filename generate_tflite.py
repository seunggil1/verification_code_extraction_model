from mediapipe.tasks.python.genai import converter

model_path = "sft-gemma3/checkpoint-228"

config = converter.ConversionConfig(
    input_ckpt=model_path,      # safetensors 파일이 있는 디렉터리
    ckpt_format="safetensors",
    model_type="GEMMA3_1B",                  # special_model 인자로 GEMMA3_1B 사용
    backend="cpu",                           # CPU용 int4 변환
    attention_quant_bits=4,
    feedforward_quant_bits=4,
    embedding_quant_bits=4,
    is_symmetric=True,                       # 대칭 양자화
    output_dir="intermediate/",              # 중간 파일 출력 디렉터리
    vocab_model_file=model_path,# 토크나이저 파일 경로
    output_tflite_file="gemma3-1b-it-int4/gemma3-1b-it-int4.tflite",
)
converter.convert_checkpoint(config)
