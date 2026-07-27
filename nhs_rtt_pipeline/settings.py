from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .config import get_paths


@dataclass(frozen=True)
class ProjectSettings:
    schema_version: int
    project_name: str
    random_seed: int
    deterministic_torch: bool
    data_start_month: str
    data_dirs: Mapping[str, str] = field(default_factory=dict)
    output_dirs: Mapping[str, str] = field(default_factory=dict)
    forecasting: Mapping[str, Any] = field(default_factory=dict)
    model: Mapping[str, Any] = field(default_factory=dict)
    baselines: Mapping[str, Any] = field(default_factory=dict)
    shap: Mapping[str, Any] = field(default_factory=dict)
    optimisation: Mapping[str, Any] = field(default_factory=dict)
    smoke_test: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ProjectSettings":
        required = [
            "schema_version",
            "project_name",
            "random_seed",
            "deterministic_torch",
            "data_start_month",
            "forecasting",
            "model",
            "optimisation",
        ]
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"pipeline_config.json is missing required keys: {missing}")
        return cls(
            schema_version=int(data["schema_version"]),
            project_name=str(data["project_name"]),
            random_seed=int(data["random_seed"]),
            deterministic_torch=bool(data["deterministic_torch"]),
            data_start_month=str(data["data_start_month"]),
            data_dirs=dict(data.get("data_dirs", {})),
            output_dirs=dict(data.get("output_dirs", {})),
            forecasting=dict(data.get("forecasting", {})),
            model=dict(data.get("model", {})),
            baselines=dict(data.get("baselines", {})),
            shap=dict(data.get("shap", {})),
            optimisation=dict(data.get("optimisation", {})),
            smoke_test=dict(data.get("smoke_test", {})),
        )

    def layer1_overrides(self) -> dict[str, Any]:
        forecasting = self.forecasting
        model = self.model
        baselines = self.baselines
        return {
            "data_start_month": self.data_start_month,
            "encoder_length": int(forecasting.get("encoder_length", 24)),
            "prediction_length": int(forecasting.get("prediction_length", forecasting.get("forecast_horizon", 12))),
            "validation_months": int(forecasting.get("validation_months", 12)),
            "test_months": int(forecasting.get("test_months", 12)),
            "batch_size": int(model.get("batch_size", 512)),
            "max_epochs": int(model.get("max_epochs", 35)),
            "learning_rate": float(model.get("learning_rate", 1.0e-3)),
            "weight_decay": float(model.get("weight_decay", 1.0e-4)),
            "hidden_channels": int(model.get("hidden_channels", 96)),
            "tcn_levels": int(model.get("tcn_levels", 5)),
            "kernel_size": int(model.get("kernel_size", 3)),
            "dropout": float(model.get("dropout", 0.15)),
            "embedding_dim": int(model.get("embedding_dim", 16)),
            "early_stopping_patience": int(model.get("early_stopping_patience", 7)),
            "gradient_clip_norm": float(model.get("gradient_clip_norm", 1.0)),
            "random_seed": int(self.random_seed),
            "num_workers": int(model.get("num_workers", 2)),
            "enable_hist_gradient_boosting_baseline": bool(baselines.get("enable_hist_gradient_boosting", True)),
            "hist_gradient_boosting_max_train_rows": int(baselines.get("hist_gradient_boosting_max_train_rows", 350000)),
            "hist_gradient_boosting_max_iter": int(baselines.get("hist_gradient_boosting_max_iter", 180)),
        }


def load_pipeline_settings(
    config_path: str | Path | None = None,
    project_root: str | Path | None = None,
) -> ProjectSettings:
    paths = get_paths(project_root)
    resolved = Path(config_path).expanduser().resolve() if config_path is not None else paths.pipeline_config
    if not resolved.exists():
        raise FileNotFoundError(f"Missing central pipeline configuration file: {resolved}")
    with open(resolved, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return ProjectSettings.from_mapping(data)


def write_effective_settings_summary(settings: ProjectSettings, path: str | Path) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": settings.schema_version,
        "project_name": settings.project_name,
        "random_seed": settings.random_seed,
        "deterministic_torch": settings.deterministic_torch,
        "data_start_month": settings.data_start_month,
        "forecasting": dict(settings.forecasting),
        "model": dict(settings.model),
        "optimisation": dict(settings.optimisation),
    }
    with open(resolved, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return resolved
