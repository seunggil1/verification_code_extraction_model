from mediapipe.tasks.python.genai import bundler
config = bundler.BundleConfig(
    tflite_model="./models/gemma3-270m-it-sms-verification_code_extraction/sms_verification_code_extraction_fp16_ekv2048.tflite",
    tokenizer_model="./models/gemma3-270m-it-sms-verification_code_extraction/tokenizer.model",
    start_token="<bos>",
    stop_tokens=["<eos>", "<end_of_turn>"],
    output_filename="sms_verification_code_extraction.task",
    prompt_prefix="<start_of_turn>user\n",
    prompt_suffix="<end_of_turn>\n<start_of_turn>model\n",
)
bundler.create_bundle(config)