"""Kaggle runtime helpers — GPU checks and training defaults."""

from __future__ import annotations

import gc
import subprocess
from pathlib import Path

from ds_rag_embedder.config import EmbedderConfig


def legacy_gpu_needs_torch_fix() -> bool:
    """Detect P100/sm_60 via nvidia-smi (safe before torch import)."""
    try:
        cap = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True,
        ).strip()
        return int(cap.split(".")[0]) < 7
    except Exception:
        return False


def configure_torch_for_kaggle() -> None:
    """Verify GPU/torch state after the notebook setup cell."""
    import torch

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available in this session.")
        return

    major, minor = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    print(f"GPU ready: {name} (sm_{major}{minor}) | torch {torch.__version__}")


def cuda_sanity_check() -> None:
    """Raise if CUDA tensors cannot run a backward pass."""
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
    """Memory-safe hyperparameters for Kaggle T4 notebooks."""
    return EmbedderConfig(
        epochs=2,
        batch_size=12,
        max_seq_length=256,
        output_dir=output_dir or Path("models/ds-rag-embedder-v1"),
    )


def _download_published_model(output_dir: Path, hub_model_id: str) -> Path:
    from huggingface_hub import snapshot_download

    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=hub_model_id, local_dir=str(output_dir))
    print(f"Loaded published weights → {output_dir}")
    return output_dir


def ensure_trained_model(output_dir: Path, hub_model_id: str = "waghelad/ds-rag-embedder-v1") -> Path:
    """
    On T4+: fine-tune locally. On P100 or failure: use published Hugging Face weights
    so the notebook always completes with the same benchmark-quality model.
    """
    output_dir = Path(output_dir)

    if legacy_gpu_needs_torch_fix():
        print(
            "P100/sm_60 detected — skipping on-notebook fine-tuning "
            "(CUDA/torchcodec compatibility on Kaggle)."
        )
        return _download_published_model(output_dir, hub_model_id)

    from ds_rag_embedder.train import train

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

    return _download_published_model(output_dir, hub_model_id)
