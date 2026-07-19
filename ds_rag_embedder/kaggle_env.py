"""Kaggle runtime helpers — GPU detection, device selection, model loading."""

from __future__ import annotations

import gc
import os
import subprocess
from pathlib import Path

from ds_rag_embedder.config import EmbedderConfig


def legacy_gpu() -> bool:
    """Detect P100/sm_60 via nvidia-smi (safe before torch import)."""
    try:
        cap = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True,
        ).strip()
        return int(cap.split(".")[0]) < 7
    except Exception:
        return False


def kaggle_device() -> str:
    """Return 'cuda' or 'cpu' based on notebook setup env."""
    forced = os.environ.get("KAGGLE_DEVICE")
    if forced in {"cpu", "cuda"}:
        return forced
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def configure_torch_for_kaggle() -> None:
    """Log runtime device after setup cell."""
    import torch

    device = kaggle_device()
    if device == "cpu":
        print(f"Runtime: CPU inference | torch {torch.__version__}")
        return

    if not torch.cuda.is_available():
        raise RuntimeError("KAGGLE_DEVICE=cuda but CUDA is unavailable.")

    major, minor = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    print(f"Runtime: CUDA | torch {torch.__version__} | {name} sm_{major}{minor}")


def cuda_sanity_check() -> None:
    """Verify the selected runtime can execute embedding workloads."""
    import torch

    device = kaggle_device()
    if device == "cpu":
        x = torch.randn(8, 8)
        x.sum().item()
        print("CPU check passed.")
        return

    x = torch.randn(8, 8, device="cuda", requires_grad=True)
    x.sum().backward()
    torch.cuda.synchronize()
    print("CUDA check passed.")


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
    T4 + CUDA: optional fine-tune. P100 / CPU: load published HF weights (same benchmarks).
    """
    output_dir = Path(output_dir)

    if legacy_gpu() or kaggle_device() == "cpu":
        print("Using published Hugging Face weights (P100/CPU-safe path).")
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


def make_embedder(model_name_or_path: str | Path):
    """Create DSRAGEmbedder with Kaggle-safe device selection."""
    from ds_rag_embedder.model import DSRAGEmbedder

    return DSRAGEmbedder(model_name_or_path, device=kaggle_device())
