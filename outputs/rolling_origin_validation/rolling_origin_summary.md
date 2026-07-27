# Rolling-Origin Validation Summary

Configuration: `{"batch_size": 512, "close_to_nominal_tolerance": 0.05, "dropout": 0.15, "early_stopping_patience": 5, "embedding_dim": 16, "encoder_length": 24, "forecast_horizon": 12, "gradient_clip_norm": 1.0, "hidden_channels": 96, "internal_validation_months": 12, "kernel_size": 3, "learning_rate": 0.001, "max_epochs": 18, "min_train_months": 36, "model_name": "TCN", "nominal_interval_alpha": 0.2, "num_workers": 0, "origin_step_months": 6, "quantiles": [0.1, 0.5, 0.9], "random_seed": 42, "requested_origins": 3, "target_column": "incomplete_total", "tcn_levels": 5, "weight_decay": 0.0001}`

Forecast origins: 2024-05-01, 2024-11-01, 2025-05-01
Forecast months evaluated: 2024-06-01 to 2026-05-01
Rows evaluated: 96972
Raw quantile-crossing rate before correction: 0.0000

## Reliability

Nominal P10-P90 coverage: 80.00%
Empirical P10-P90 coverage: 88.72%
The nominal 80% interval is not close to empirical coverage within +/-5%.

Raw P10/P50/P90 values are saved alongside corrected display quantiles. Quantile crossing is detected before correction.
The final production model is trained separately from these validation fold models.