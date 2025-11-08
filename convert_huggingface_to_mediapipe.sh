#!/bin/bash
set -e

echo "Creating virtual environment: ai-edge-torch"
python -m venv ai-edge-torch
source ai-edge-torch/bin/activate

echo "Installing required packages..."
echo "Installing ai-edge-torch package..."
pip install "ai-edge-torch>=0.6.0"
echo "Installing Hugging Face Hub CLI..."
pip install huggingface_hub[cli]
echo "Package installation completed!"
echo ""

echo "Downloading models from Hugging Face..."
hf download sg2023/gemma3-270m-it-sms-verification_code_extraction --local-dir "./models/gemma3-270m-it-sms-verification_code_extraction"
echo ""

echo "Starting Hugging Face to LiteRT conversion..."
python convert_hf_to_liteRT.py
echo "LiteRT conversion completed!"

echo "Cleaning up ai-edge-torch environment..."
deactivate
rm -r ai-edge-torch
echo ""

echo "Setting up MediaPipe environment..."
python -m venv mediapipe
source mediapipe/bin/activate
echo "Installing MediaPipe package..."
pip install mediapipe
echo "Setup complete!"

echo "Starting MediaPipe model conversion..."
python convert_liteRT_to_mediapipe.py
echo "MediaPipe model conversion completed! : sms_verification_code_extraction.task"

deactivate
rm -r mediapipe
