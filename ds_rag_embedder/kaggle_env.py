"""Kaggle runtime helpers — GPU checks and training defaults."""

from __future__ import annotations

import gc
import subprocess
import sys
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
    """
    Verify GPU/torch state after the notebook setup cell.

    PyTorch reinstall for P100 must happen in the first notebook cell *before*
    importing torch — reloading torch in Jupyter raises ImportError.
    """
    import torch

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available in this session.")
        return

    major, minor = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    print(f"GPU ready: {name} (sm_{major}{minor}) | torch {torch.__version__}")

    if legacy_gpu_needs_torch_fix() and "+cu128" in torch.__version__:
        raise RuntimeError(
            "P100 detected but PyTorch cu128 is still loaded. Re-run the setup cell "
            "from the top (Session → Restart & Run All)."
        )


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
