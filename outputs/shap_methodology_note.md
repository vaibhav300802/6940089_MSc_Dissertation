# Custom TCN SHAP Explainability

This explainability layer imports `TCNQuantileRegressor` from `nhs_rtt_pipeline.modeling`, loads the canonical `models/tcn_state_dict.pt`, `models/model_config.json`, and `models/feature_metadata.json`, and reconstructs the same transformed encoder tensor columns used during training.

The explained model output is the inverse-transformed median forecast, P50. The configured forecast horizons are: 1, 6, 12 month(s) ahead.

The implementation uses `shap.KernelExplainer` with a wrapper around the custom PyTorch TCN. The wrapper receives interpretable encoder feature values, reconstructs the scaled tensor inputs using fitted preprocessing metadata, passes Trust and specialty identifier indices to the model embeddings, and returns P50 forecasts for the configured horizons.

Case selection uses model predictions from held-out test-period encoder windows. It does not use actual future target values to select cases for explanation. Missing or non-finite predictions raise an error and are recorded in the audit log rather than being replaced by ground-truth values.

Local explanation consistency is checked as:

```text
base value + sum(feature contributions) approximately equals model P50 output
```

The reported approximation error is the residual from that equality for each explained Trust-specialty context and horizon.

All wording in the generated interpretation tables is associational. SHAP values describe contributions to model predictions; they are not causal claims.

Configuration:

```json
{
  "n_background": 50,
  "n_explain": 100,
  "min_distinct_trusts": 10,
  "selected_trust_count": 5,
  "priority_trust_codes": [
    "RA2"
  ],
  "priority_trust_names": [
    "ROYAL SURREY NHS FOUNDATION TRUST"
  ],
  "recent_lags": 3,
  "nsamples": 192,
  "horizons": [
    1,
    6,
    12
  ],
  "random_seed": 42,
  "max_waterfall_features": 14
}
```
