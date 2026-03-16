import torch
import sys
import os

def check_bede_environment():
    print("=== Bede Grace Hopper Environment Check ===")
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    
    # 1. Check for ARM64 architecture
    arch = os.uname().machine
    print(f"Processor Architecture: {arch} {'(Correct)' if arch == 'aarch64' else '(Unexpected!)'}")

    # 2. Check CUDA Availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if not cuda_available:
        print("\n[!] ERROR: PyTorch cannot see the GPU.")
        print("Note: On aarch64, standard 'pip install torch' often installs the CPU-only version.")
        return

    # 3. Check GPU Identity
    device_count = torch.cuda.device_count()
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    print(f"GPU Device Count: {device_count}")
    print(f"Active GPU: {device_name}")
    
    # 4. Verify Tensor Operation (Triggers actual CUDA initialization)
    try:
        x = torch.rand(5, 5).cuda()
        y = torch.rand(5, 5).cuda()
        z = x @ y
        print("CUDA Tensor Operation: Success")
        print(f"Unified Memory check: {z.device}")
    except Exception as e:
        print(f"\n[!] FATAL: CUDA operation failed: {e}")

if __name__ == "__main__":
    check_bede_environment()
