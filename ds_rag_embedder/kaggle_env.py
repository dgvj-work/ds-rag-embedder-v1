"""Kaggle runtime helpers — GPU checks and training defaults."""

from __future__ import annotations

import gc
import importlib
import subprocess
import sys
from pathlib import Path

from ds_rag_embedder.config import EmbedderConfig


def _run_quiet(cmd: list[str]) -> None:
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def configure_torch_for_kaggle() -> None:
    """
    Fix Kaggle GPU incompatibilities before importing sentence-transformers.

    Tesla P100 (sm_60) fails backward passes with PyTorch cu128 builds that
    dropped sm_60 kernels. Reinstall cu126 wheels when needed.
    """
    import torch

    if not torch.cuda.is_available():
        return

    major, minor = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    print(f"GPU: {name} (sm_{major}{minor}) | torch {torch.__version__}")

    if major >= 7:
        return

    print("Legacy GPU (sm_60/sm_61) detected — installing PyTorch cu126 for CUDA kernel compatibility…")
    _run_quiet([sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"])
    _run_quiet(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--no-cache-dir",
            "torch",
            "torchvision",
            "torchaudio",
            "--index-url",
            "https://download.pytorch.org/whl/cu126",
        ]
    )
    importlib.invalidate_caches()
    import torch as torch_reloaded  # noqa: F401


def cuda_sanity_check() -> None:
    """Raise if CUDA tensors cannot run a backward pass (common Kaggle failure)."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No GPU detected. In Kaggle: Settings → Accelerator → GPU (T4/P100), then re-run."
        )

    x = torch.randn(8, 8, device="cuda", requires_grad=True)
    loss = x.sum()
    loss.backward()
    torch.cuda.synchronize()
    print("CUDA sanity check passed.")


def assert_gpu_ready() -> None:
    configure_torch_for_kaggle()
    cuda_sanity_check()


def kaggle_train_config(output_dir: Path | None = None) -> EmbedderConfig:
    """Memory-safe hyperparameters for Kaggle T4/P100 notebooks."""
    return EmbedderConfig(
        epochs=3,
        batch_size=16,
        max_seq_length=256,
        output_dir=output_dir or Path("models/ds-rag-embedder-v1"),
    )


def ensure_trained_model(output_dir: Path, hub_model_id: str = "waghelad/ds-rag-embedder-v1") -> Path:
    """
    Train locally or fall back to published Hugging Face weights so downstream
    benchmark cells still run if fine-tuning fails.
    """
    from ds_rag_embedder.train import train

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    try:
        path = train(config=kaggle_train_config(output_dir), use_amp=False, use_evaluator=False)
        print(f"Fine-tuning complete → {path}")
        return path
    except Exception as exc:
        print(f"Fine-tuning failed ({type(exc).__name__}: {exc})")
        print(f"Falling back to published weights → {hub_model_id}")

    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=hub_model_id, local_dir=str(output_dir))
    return output_dir
