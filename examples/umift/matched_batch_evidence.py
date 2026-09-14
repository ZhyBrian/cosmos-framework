"""Record the first matched UMI-FT training microbatches for arm comparison."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from cosmos_framework.utils.callback import Callback


def _tensor_tree_sha256(value: Any) -> str:
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor")
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        elif isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(item):
                visit(key)
                visit(item[key])
        elif item is None:
            digest.update(b"None")
        else:
            digest.update(type(item).__name__.encode())
            digest.update(repr(item).encode())

    visit(value)
    return digest.hexdigest()


def _first_scalar(value: Any) -> Any:
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"matched evidence identity must be scalar, got {tuple(value.shape)}")
        return value.detach().item()
    return value


class MatchedBatchEvidence(Callback):
    """Write detached identity hashes for a bounded number of microbatches."""

    def __init__(self, num_microbatches: int = 8):
        if num_microbatches <= 0:
            raise ValueError("num_microbatches must be positive")
        self.num_microbatches = int(num_microbatches)
        self._count = 0
        self._stream = None
        self._rank = 0

    def on_train_start(self, model, iteration: int = 0) -> None:
        del model, iteration
        self._rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        output_dir = Path(self.config.job.path_local)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"matched_microbatches_rank_{self._rank}.jsonl"
        self._stream = path.open("w", encoding="utf-8")

    def _close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    @torch.no_grad()
    def on_training_step_batch_end(
        self,
        model,
        data_batch: dict[str, Any],
        output_batch: dict[str, Any],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        del model, loss
        if self._count >= self.num_microbatches:
            return
        if self._stream is None:
            raise RuntimeError("MatchedBatchEvidence.on_train_start must run before training")

        row = {
            "schema": "umift-matched-microbatch-v1",
            "rank": self._rank,
            "microbatch_index": self._count,
            "iteration": int(iteration),
            "episode_id": int(_first_scalar(data_batch["episode_id"])),
            "window_start": int(_first_scalar(data_batch["window_start"])),
            "source_id_sha256": _tensor_tree_sha256(data_batch.get("source_id")),
            "source_indices_sha256": _tensor_tree_sha256(data_batch.get("video_source_indices", data_batch.get("source_indices"))),
            "action_sha256": _tensor_tree_sha256(data_batch["action"]),
            "x0_sha256": _tensor_tree_sha256(output_batch["x0"]),
            "xt_sha256": _tensor_tree_sha256(output_batch["xt"]),
            "sigma_sha256": _tensor_tree_sha256(output_batch["sigma"]),
            "condition_mask_vision_sha256": _tensor_tree_sha256(output_batch["condition_mask_vision"]),
        }
        self._stream.write(json.dumps(row, sort_keys=True) + "\n")
        self._stream.flush()
        self._count += 1
        if self._count == self.num_microbatches:
            self._close()

    def on_train_end(self, model, iteration: int = 0) -> None:
        del model, iteration
        self._close()


__all__ = ["MatchedBatchEvidence"]
