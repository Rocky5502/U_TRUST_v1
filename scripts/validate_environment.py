from __future__ import annotations

import platform
import sys


def main() -> int:
    print("U-TRUST environment validation")
    print("Python:", sys.version.split()[0])
    print("Platform:", platform.platform())
    if not ((3, 10) <= sys.version_info[:2] < (3, 13)):
        print("WARNING: recommended Python is 3.10-3.12; 3.11 is preferred.")
    try:
        import torch
        print("PyTorch:", torch.__version__)
        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))
            print("CUDA runtime:", torch.version.cuda)
    except ImportError:
        print("PyTorch: not installed (fine for CPU smoke test)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
