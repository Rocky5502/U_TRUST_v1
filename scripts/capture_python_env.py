from __future__ import annotations

import json
import platform


def main() -> None:
    out: dict[str, object] = {"python": platform.python_version()}
    try:
        import torch
        out.update({
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "devices": [
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_mib": torch.cuda.get_device_properties(i).total_memory // (1024**2),
                    "capability": list(torch.cuda.get_device_capability(i)),
                }
                for i in range(torch.cuda.device_count())
            ],
        })
    except Exception as exc:
        out["torch_error"] = repr(exc)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
