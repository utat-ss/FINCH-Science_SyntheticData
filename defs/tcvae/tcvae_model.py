from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .tcvae_backbone import DualPathTransformerConditionalVAE


def _resolve_sidecar(cfg_dir: Path, configured_path: str | None, fallback_name: str) -> Path:
    candidates: list[Path] = []
    if configured_path:
        raw = Path(configured_path)
        candidates.append(raw)
        if not raw.is_absolute():
            candidates.append(cfg_dir / raw)
    candidates.append(cfg_dir / fallback_name)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not resolve sidecar file. Searched: {searched}")


class TCVAESynthesisModel(DualPathTransformerConditionalVAE):
    def __init__(
        self,
        input_dim: int,
        condition_dim: int,
        latent_dim: int,
        dual_path_transformer_params: dict[str, Any],
        sampling_cfg: dict[str, Any],
        normalization_cfg: dict[str, float],
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            cond_dim=condition_dim,
            latent_dim=latent_dim,
            d_model=int(dual_path_transformer_params.get("d_model", 256)),
            n_heads=int(dual_path_transformer_params.get("n_heads", 8)),
            encoder_layers=int(dual_path_transformer_params.get("encoder_layers", 6)),
            decoder_layers=int(dual_path_transformer_params.get("decoder_layers", 2)),
            dropout=float(dual_path_transformer_params.get("dropout", 0.0)),
            latent_fuse_weight=float(dual_path_transformer_params.get("latent_fuse_weight", 0.7)),
            latent_fuse_weight_min=float(dual_path_transformer_params.get("latent_fuse_weight_min", 0.3)),
            gated_film_init=float(dual_path_transformer_params.get("gated_film_init", 0.1)),
            latent_fuse_weight_learnable=bool(dual_path_transformer_params.get("latent_fuse_weight_learnable", True)),
            global_path_dropout=float(dual_path_transformer_params.get("global_path_dropout", 0.2)),
            global_path_hidden_dim=int(dual_path_transformer_params.get("global_path_hidden_dim", 256)),
            global_path_warmup_hold_epochs=int(dual_path_transformer_params.get("global_path_warmup_hold_epochs", 5)),
            global_path_warmup_ramp_epochs=int(dual_path_transformer_params.get("global_path_warmup_ramp_epochs", 10)),
            decoder_logit_gain=float(dual_path_transformer_params.get("decoder_logit_gain", 1.0)),
            decoder_use_film=bool(dual_path_transformer_params.get("decoder_use_film", True)),
        )
        self.latent_dim = int(latent_dim)
        self.condition_dim = int(condition_dim)
        self.sample_temperature = float(sampling_cfg.get("temperature", 1.0))
        self.raw_min = float(normalization_cfg["raw_min"])
        self.raw_max = float(normalization_cfg["raw_max"])
        self.stat_mean = float(normalization_cfg["stat_mean"])
        self.stat_std = max(float(normalization_cfg["stat_std"]), 1e-8)

    @staticmethod
    def normalize_conditions(cond: torch.Tensor) -> torch.Tensor:
        if cond.ndim != 2:
            raise ValueError(f"conditions must be rank-2 but received shape {tuple(cond.shape)}")
        if cond.size(1) == 0:
            return cond
        return cond / cond.sum(dim=1, keepdim=True).clamp_min(1e-6)

    def minmax_to_statistical(self, spectra_01: torch.Tensor) -> torch.Tensor:
        raw = spectra_01 * (self.raw_max - self.raw_min) + self.raw_min
        return (raw - self.stat_mean) / self.stat_std

    @classmethod
    def from_setup_dict(cls, setup_dict: dict[str, Any]) -> "TCVAESynthesisModel":
        tcvae_block = setup_dict.get("cfg_tcvae") or setup_dict.get("cfg_cvae")
        if tcvae_block is None:
            raise ValueError("Expected cfg_tcvae or cfg_cvae in tcVAE setup json.")

        model_cfg = tcvae_block["model"]
        if model_cfg.get("architecture") != "dual_path_transformer":
            raise ValueError(
                f"TCVAESynthesisModel only supports architecture='dual_path_transformer', got {model_cfg.get('architecture')!r}"
            )

        artifacts_cfg = tcvae_block.get("artifacts", {})
        cfg_dir = Path(setup_dict.get("__cfg_dir__", "."))

        minmax_path = _resolve_sidecar(cfg_dir, artifacts_cfg.get("minmax_stats_path"), "minmax_stats.json")
        norm_dict_path = _resolve_sidecar(cfg_dir, artifacts_cfg.get("norm_dict_path"), "norm_dict.json")

        minmax_stats = json.loads(minmax_path.read_text(encoding="utf-8"))
        norm_dict = json.loads(norm_dict_path.read_text(encoding="utf-8"))

        raw_reference = minmax_stats.get("raw_reference", {})
        normalization_cfg = {
            "raw_min": float(raw_reference["min"]),
            "raw_max": float(raw_reference["max"]),
            "stat_mean": float(norm_dict["mean_vals"]),
            "stat_std": float(norm_dict["std_vals"]),
        }

        return cls(
            input_dim=int(model_cfg["input_dim"]),
            condition_dim=int(model_cfg["condition_dim"]),
            latent_dim=int(model_cfg["latent_dim"]),
            dual_path_transformer_params=dict(model_cfg.get("dual_path_transformer_params", {})),
            sampling_cfg=dict(tcvae_block.get("sampling", {})),
            normalization_cfg=normalization_cfg,
        )

    @torch.no_grad()
    def sample_from_conditions(
        self,
        conditions: torch.Tensor,
        temperature: float | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if conditions.ndim != 2 or conditions.size(1) != self.condition_dim:
            raise ValueError(
                f"conditions must have shape (n, {self.condition_dim}) but received {tuple(conditions.shape)}"
            )

        cond = self.normalize_conditions(conditions)
        temp = self.sample_temperature if temperature is None else float(temperature)
        z = torch.randn(cond.size(0), self.latent_dim, device=cond.device, generator=generator)
        if temp != 1.0:
            z = z * temp

        decoded_01 = self.decode(z, cond)
        return self.minmax_to_statistical(decoded_01)
