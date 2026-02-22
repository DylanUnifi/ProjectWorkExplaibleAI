import torch


def get_device():
    """Get device, testing CUDA actually works (not just available)."""
    if torch.cuda.is_available():
        try:
            # Test that CUDA actually works with this GPU
            torch.zeros(1).cuda()
            return torch.device("cuda")
        except RuntimeError:
            print("⚠️  CUDA available but GPU not compatible with this PyTorch version. Using CPU.")
            return torch.device("cpu")
    return torch.device("cpu")
