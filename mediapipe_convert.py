import platform
import sys

from mediapipe.tasks.python.genai import converter

if platform.system() != "Linux":
    sys.exit("Error: 이 스크립트는 Linux에서만 실행됩니다.")

model_path = "C:/Users/ksgg1/.cache/huggingface/hub/models--sg2023--Gemma2-2B-IT-Sms-Verification_Code_Extraction/snapshots/ddbf539b5220215bddbb843668d33f05b3404969"

config = converter.ConversionConfig(
  input_ckpt=model_path,
  ckpt_format="safetensors",
  model_type="GEMMA2_2B",
  backend="CPU",
  output_dir="google/gemma-2-2b-it-tuning-tf-intermediate",
  combine_file_only=False,
  vocab_model_file=model_path,
  output_tflite_file="google/gemma-2-2b-it-tuning-tf-intermediate",
)

converter.convert_checkpoint(config)