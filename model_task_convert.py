import argparse

from mediapipe.tasks.python.genai import bundler
from transformers.utils import cached_file


def main(args):
    tflite_model_path = args.tflite_model_path
    sentence_pience_tokenizer_model = cached_file(args.repo_id, "tokenizer.model")
    bundler.create_bundle(
        bundler.BundleConfig(
            tflite_model=tflite_model_path,
            tokenizer_model=sentence_pience_tokenizer_model,
            start_token="<bos>",
            stop_tokens=["<eos>", "<end_of_turn>"],
            output_filename=args.output_dir,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo_id",
        type=str,
        default="sg2023/Gemma2-2B-IT-Sms-Verification_Code_Extraction",
    )
    parser.add_argument(
        "--tflite_model_path", type=str, default="./export/gemma2_sft_q8_ekv1024.tflite"
    )
    parser.add_argument("--output_dir", type=str, default="task_bundle_output")
    parsed_args = parser.parse_args()
    main(parsed_args)
