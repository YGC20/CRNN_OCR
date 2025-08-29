import torch

try:
    print(f"PyTorch version: {torch.__version__}")
    print(f"Is CUDA available? {torch.cuda.is_available()}")
    cuda_version = torch.version.cuda if torch.version.cuda else 'N/A'
    print(f"PyTorch CUDA version: {cuda_version}")
except Exception as e:
    print(f"An error occurred: {e}")
