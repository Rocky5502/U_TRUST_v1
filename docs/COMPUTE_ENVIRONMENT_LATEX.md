# Recommended manuscript text after machine freeze

```tex
\subsection{Compute Environment}
\label{sec:compute}
Experiments are executed on a Windows-based Intel system with 64~GB of system memory and a Samsung 990 PRO 2~TB SSD. Local GPU inference and benchmark execution use NVIDIA GeForce RTX 5070 Ti hardware. NVIDIA specifies 16~GB of GDDR7 memory for a standard RTX 5070 Ti; therefore, we do not interpret the previously noted 48~GB figure as the capacity of a single card. Before the frozen test campaign, we capture the exact hardware topology with \texttt{nvidia-smi} and a machine-manifest script and report the detected GPU count, memory per device, and aggregate available GPU memory. \TBD{insert frozen Intel CPU model, Windows version/build, NVIDIA driver, PyTorch/CUDA runtime, GPU count, and per-device/aggregate memory from the machine manifest}. The two open-weight 8B backbones are evaluated sequentially rather than held in memory simultaneously. Local LLM inference uses the frozen quantization and decoding configuration reported with the experiment artifacts.
```

If the manifest shows one standard RTX 5070 Ti, write simply "one NVIDIA GeForce RTX 5070 Ti with 16 GB GDDR7" and delete all discussion of 48 GB.
