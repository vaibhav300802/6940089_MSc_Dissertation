# %% Cell 1
import importlib.util
import subprocess
import sys

PIP_PACKAGES = [
    "numpy>=1.23.0",
    "pandas>=2.0.0",
    "pyarrow>=10.0.0",
    "matplotlib>=3.7.0",
    "shap>=0.44.0",
    "torch>=2.1.0",
]

IMPORT_CHECKS = {
    "numpy": "numpy",
    "pandas": "pandas",
    "pyarrow": "pyarrow",
    "matplotlib": "matplotlib",
    "shap": "shap",
    "torch": "torch",
}

missing_packages = []
for package_name, module_name in IMPORT_CHECKS.items():
    if importlib.util.find_spec(module_name) is None:
        missing_packages.append(package_name)

if missing_packages:
    packages_to_install = [
        package_spec
        for package_spec in PIP_PACKAGES
        if package_spec.split(">=")[0] in missing_packages
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages_to_install])

# %% Cell 2
import json
import os
import random
import sys
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from IPython.display import display
except Exception:
    def display(obj: object) -> None:
        if isinstance(obj, pd.DataFrame):
            print(obj.head(25).to_string())
        else:
            print(obj)

PROJECT_ROOT_CANDIDATES = [
    Path.cwd(),
    Path.cwd().parent,
    Path("/content/nhs_rtt_msc_project"),
    Path("/content"),
]
for candidate_root in PROJECT_ROOT_CANDIDATES:
    if (candidate_root / "nhs_rtt_pipeline").exists():
        sys.path.insert(0, str(candidate_root))
        break
else:
    raise FileNotFoundError(
        "Could not find the shared nhs_rtt_pipeline package. "
        "Run this notebook from the project root or upload the complete nhs_rtt_msc_project folder to Colab."
    )

from nhs_rtt_pipeline.config import COLUMNS, ensure_directories, get_paths
from nhs_rtt_pipeline.explainability import (
    CustomTCNShapExplainer,
    ShapLayerConfig,
    aggregate_by_display,
    compute_kernel_shap,
    consistency_report_frame,
    context_index_frame,
    feature_importance_frames,
    load_custom_tcn_explainer_bundle,
    local_trust_specialty_explanations,
    make_trust_interpretations,
    safe_filename,
    shap_values_long_frame,
    write_methodology_note,
)
from nhs_rtt_pipeline.reproducibility import set_global_seed
from nhs_rtt_pipeline.settings import load_pipeline_settings


try:
    PROJECT_SETTINGS = load_pipeline_settings(os.environ.get("NHS_RTT_PIPELINE_CONFIG"))
    SHAP_SETTINGS = PROJECT_SETTINGS.shap
except FileNotFoundError:
    PROJECT_SETTINGS = None
    SHAP_SETTINGS = {}

CONFIG = ShapLayerConfig(
    n_background=int(SHAP_SETTINGS.get("n_background", 50)),
    n_explain=int(SHAP_SETTINGS.get("n_explain", 100)),
    min_distinct_trusts=int(SHAP_SETTINGS.get("min_distinct_trusts", 10)),
    selected_trust_count=int(SHAP_SETTINGS.get("selected_trust_count", 5)),
    priority_trust_codes=tuple(str(value).strip().upper() for value in SHAP_SETTINGS.get("priority_trust_codes", [])),
    priority_trust_names=tuple(str(value).strip() for value in SHAP_SETTINGS.get("priority_trust_names", [])),
    recent_lags=int(SHAP_SETTINGS.get("recent_lags", 3)),
    nsamples=int(SHAP_SETTINGS.get("nsamples", 192)),
    horizons=tuple(SHAP_SETTINGS.get("horizons", (1, 6, 12))),
    random_seed=int(PROJECT_SETTINGS.random_seed) if PROJECT_SETTINGS is not None else 42,
    max_waterfall_features=int(SHAP_SETTINGS.get("max_waterfall_features", 14)),
)

PATHS = get_paths()
ensure_directories(PATHS)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_global_seed(
    CONFIG.random_seed,
    deterministic_torch=bool(PROJECT_SETTINGS.deterministic_torch) if PROJECT_SETTINGS is not None else False,
)

required_files = {
    "TCN state dictionary": PATHS.tcn_state_dict,
    "TCN model configuration": PATHS.model_config,
    "fitted feature metadata": PATHS.feature_metadata,
    "processed RTT data": PATHS.clean_parquet,
}
missing_files = {label: str(path) for label, path in required_files.items() if not Path(path).exists()}
if missing_files:
    raise FileNotFoundError(f"Missing required Layer 2 inputs: {missing_files}")

print(json.dumps(asdict(CONFIG), indent=2))
print(f"Device: {DEVICE}")
print(f"Model state dict: {PATHS.tcn_state_dict}")
print(f"Model config: {PATHS.model_config}")
print(f"Feature metadata: {PATHS.feature_metadata}")
print(f"Clean RTT data: {PATHS.clean_parquet}")

# %% Cell 3
import pandas as pd
import torch

model, model_config, feature_metadata = load_custom_tcn_explainer_bundle(
    state_dict_path=PATHS.tcn_state_dict,
    model_config_path=PATHS.model_config,
    feature_metadata_path=PATHS.feature_metadata,
    device=DEVICE,
)

explainer = CustomTCNShapExplainer(
    model=model,
    model_config=model_config,
    feature_metadata=feature_metadata,
    device=DEVICE,
    horizons=CONFIG.horizons,
)

clean_rtt = pd.read_parquet(PATHS.clean_parquet)
prepared_rtt = explainer.prepare_model_frame(clean_rtt)
feature_specs = explainer.build_feature_specs(CONFIG.recent_lags, include_identifiers=True)
feature_names = [spec.feature_name for spec in feature_specs]

print(f"Loaded model class: {model.__class__.__module__}.{model.__class__.__name__}")
print(f"Model artifact format: {model_config.get('format')}")
print(f"Explained output: P50 median forecast")
print(f"Explained horizons: {explainer.horizons}")
print(f"Prepared rows: {len(prepared_rtt):,}")
print(f"Prepared Trust-specialty series: {prepared_rtt[COLUMNS.series_id].nunique():,}")
print(f"SHAP feature count: {len(feature_specs):,}")

# %% Cell 4
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


def forecast_start_range(start: int, end: int) -> List[int]:
    if end < start:
        return []
    return list(range(int(start), int(end) + 1))


def score_contexts_for_selection(contexts: Sequence[Dict[str, object]]) -> pd.DataFrame:
    if not contexts:
        return pd.DataFrame()
    final_horizon = max(explainer.horizons)

    encoder_matrices = np.stack(
        [
            context["encoder"][explainer.feature_columns].to_numpy(dtype=np.float32)
            for context in contexts
        ],
        axis=0,
    )
    trust_indices = np.asarray([int(context["trust_idx"]) for context in contexts], dtype=np.int64)
    specialty_indices = np.asarray([int(context["specialty_idx"]) for context in contexts], dtype=np.int64)
    predictions = explainer.predict_q50_from_encoder_batch(
        encoder_matrices,
        trust_indices,
        specialty_indices,
        horizons=[final_horizon],
        batch_size=1024,
    ).reshape(-1)
    if not np.isfinite(predictions).all():
        bad_positions = np.where(~np.isfinite(predictions))[0].tolist()
        bad_contexts = [contexts[position]["context_id"] for position in bad_positions[:20]]
        raise RuntimeError(f"Missing model predictions for SHAP context selection: {bad_contexts}")

    rows = []
    for context, prediction in zip(contexts, predictions):
        if not np.isfinite(prediction):
            raise RuntimeError(f"Missing model prediction for SHAP context {context['context_id']}.")
        rows.append(
            {
                "context_id": int(context["context_id"]),
                COLUMNS.series_id: context[COLUMNS.series_id],
                COLUMNS.trust_code: context[COLUMNS.trust_code],
                COLUMNS.trust_name: context[COLUMNS.trust_name],
                COLUMNS.specialty_code: context[COLUMNS.specialty_code],
                COLUMNS.specialty_name: context[COLUMNS.specialty_name],
                COLUMNS.forecast_origin: context[COLUMNS.forecast_origin],
                "selection_horizon": final_horizon,
                "selection_p50": float(prediction),
            }
        )
    return pd.DataFrame(rows)


def select_contexts_for_explanation(
    contexts: Sequence[Dict[str, object]],
    n_explain: int,
    min_distinct_trusts: int,
    selected_trust_count: int,
    priority_trust_codes: Sequence[str] = (),
    priority_trust_names: Sequence[str] = (),
) -> Tuple[List[Dict[str, object]], pd.DataFrame, pd.DataFrame]:
    index = score_contexts_for_selection(contexts)
    if index.empty:
        raise RuntimeError("No model predictions were available for SHAP context selection.")
    top_trusts = (
        index.groupby([COLUMNS.trust_code, COLUMNS.trust_name], as_index=False, observed=True)["selection_p50"]
        .sum()
        .sort_values("selection_p50", ascending=False)
        .head(selected_trust_count)
        .reset_index(drop=True)
    )
    context_by_id = {int(context["context_id"]): context for context in contexts}
    selected_ids: List[int] = []

    priority_codes = {str(code).strip().upper() for code in priority_trust_codes if str(code).strip()}
    priority_names = {
        " ".join(str(name).strip().upper().split())
        for name in priority_trust_names
        if str(name).strip()
    }
    if priority_codes or priority_names:
        normalised_names = index[COLUMNS.trust_name].astype(str).map(lambda value: " ".join(value.strip().upper().split()))
        priority_mask = index[COLUMNS.trust_code].astype(str).str.upper().isin(priority_codes) | normalised_names.isin(priority_names)
        priority_rows = (
            index[priority_mask]
            .sort_values("selection_p50", ascending=False)
            .drop_duplicates([COLUMNS.trust_code])
        )
        for context_id in priority_rows["context_id"].astype(int):
            if int(context_id) not in selected_ids:
                selected_ids.append(int(context_id))

    for trust_name in top_trusts[COLUMNS.trust_name].astype(str):
        candidates = index[index[COLUMNS.trust_name].astype(str).eq(trust_name)].sort_values("selection_p50", ascending=False)
        if not candidates.empty:
            context_id = int(candidates["context_id"].iloc[0])
            if context_id not in selected_ids:
                selected_ids.append(context_id)

    trust_order = (
        index.groupby(COLUMNS.trust_name, observed=True)["selection_p50"]
        .sum()
        .sort_values(ascending=False)
        .index.astype(str)
        .tolist()
    )
    for trust_name in trust_order:
        if index[index["context_id"].isin(selected_ids)][COLUMNS.trust_name].nunique() >= min_distinct_trusts:
            break
        candidates = index[index[COLUMNS.trust_name].astype(str).eq(trust_name)].sort_values("selection_p50", ascending=False)
        for context_id in candidates["context_id"].astype(int):
            if context_id not in selected_ids:
                selected_ids.append(int(context_id))
                break

    for context_id in index.sort_values("selection_p50", ascending=False)["context_id"].astype(int):
        if len(selected_ids) >= int(n_explain):
            break
        if int(context_id) not in selected_ids:
            selected_ids.append(int(context_id))

    selected_ids = selected_ids[: int(n_explain)]
    selected_index = index[index["context_id"].isin(selected_ids)].set_index("context_id").loc[selected_ids].reset_index()
    if selected_index[COLUMNS.trust_name].nunique() < min_distinct_trusts:
        raise RuntimeError(
            f"Selected contexts cover {selected_index[COLUMNS.trust_name].nunique()} Trusts; "
            f"required at least {min_distinct_trusts}."
        )
    return [context_by_id[context_id] for context_id in selected_ids], selected_index, top_trusts


boundaries = feature_metadata.get("boundaries", {})
minimum_time_idx = int(prepared_rtt["time_idx"].min())
validation_start_idx = int(boundaries.get("validation_start_idx", prepared_rtt["time_idx"].quantile(0.75)))
test_start_idx = int(boundaries.get("test_start_idx", prepared_rtt["time_idx"].max() - max(explainer.horizons) + 1))

background_start_indices = forecast_start_range(
    minimum_time_idx + explainer.encoder_length,
    validation_start_idx - 1,
)
test_start_indices = [test_start_idx]

background_contexts = explainer.build_contexts(prepared_rtt, background_start_indices)
candidate_test_contexts = explainer.build_contexts(prepared_rtt, test_start_indices)

if len(background_contexts) < CONFIG.n_background:
    raise RuntimeError(
        f"Only {len(background_contexts)} pre-validation background contexts are available; "
        f"need {CONFIG.n_background}."
    )
if len(candidate_test_contexts) < CONFIG.n_explain:
    raise RuntimeError(
        f"Only {len(candidate_test_contexts)} held-out test contexts are available; "
        f"need {CONFIG.n_explain}."
    )

selected_contexts, selected_context_index, top_predicted_trusts = select_contexts_for_explanation(
    candidate_test_contexts,
    CONFIG.n_explain,
    CONFIG.min_distinct_trusts,
    CONFIG.selected_trust_count,
    CONFIG.priority_trust_codes,
    CONFIG.priority_trust_names,
)

print(f"Background contexts: {len(background_contexts):,}")
print(f"Candidate test contexts: {len(candidate_test_contexts):,}")
print(f"Selected explanation contexts: {len(selected_contexts):,}")
print(f"Distinct selected Trusts: {selected_context_index[COLUMNS.trust_name].nunique():,}")
if CONFIG.priority_trust_codes or CONFIG.priority_trust_names:
    priority_rows = selected_context_index[
        selected_context_index[COLUMNS.trust_code].astype(str).str.upper().isin(CONFIG.priority_trust_codes)
        | selected_context_index[COLUMNS.trust_name].astype(str).isin(CONFIG.priority_trust_names)
    ]
    print(f"Priority Trust explanation rows selected: {len(priority_rows):,}")
    display(priority_rows[[COLUMNS.trust_code, COLUMNS.trust_name, COLUMNS.specialty_code, COLUMNS.specialty_name, "selection_p50"]].head(20))
display(top_predicted_trusts)
display(selected_context_index.head(10))

# %% Cell 5
import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import shap

background_pool = explainer.contexts_to_matrix(background_contexts, feature_specs)
background_sample = shap.sample(
    pd.DataFrame(background_pool, columns=feature_names),
    CONFIG.n_background,
    random_state=CONFIG.random_seed,
)
background_values = np.asarray(background_sample, dtype=np.float64)
test_values = explainer.contexts_to_matrix(selected_contexts, feature_specs)

np.save(PATHS.shap_background, background_values)
np.save(PATHS.shap_test_matrix, test_values)
np.save(PATHS.shap_feature_names_npy, np.asarray(feature_names, dtype=object))
with open(PATHS.shap_feature_names_json, "w", encoding="utf-8") as handle:
    json.dump(feature_names, handle, indent=2)
with open(PATHS.shap_feature_groups_json, "w", encoding="utf-8") as handle:
    json.dump([asdict(spec) for spec in feature_specs], handle, indent=2)

top_predicted_trusts.to_csv(PATHS.shap_top_trusts, index=False)
selected_context_index.to_csv(PATHS.shap_latest_predictions_by_series, index=False)

print(f"Saved background matrix: {PATHS.shap_background}")
print(f"Saved test matrix: {PATHS.shap_test_matrix}")
print(f"Background matrix shape: {background_values.shape}")
print(f"Test matrix shape: {test_values.shape}")

# %% Cell 6
import numpy as np
import pandas as pd

shap_values, expected_values, model_outputs, audit_log = compute_kernel_shap(
    explainer=explainer,
    contexts=selected_contexts,
    specs=feature_specs,
    background_values=background_values,
    test_values=test_values,
    configured_nsamples=CONFIG.nsamples,
    checkpoint_path=PATHS.outputs_dir / "shap_kernel_checkpoint.npz",
    audit_checkpoint_path=PATHS.outputs_dir / "shap_kernel_checkpoint_audit.csv",
    checkpoint_every=1,
    progress_every=1,
)

context_index = context_index_frame(selected_contexts, model_outputs=model_outputs, horizons=explainer.horizons)
values_long = shap_values_long_frame(
    shap_values=shap_values,
    expected_values=expected_values,
    model_outputs=model_outputs,
    feature_values=test_values,
    contexts=selected_contexts,
    specs=feature_specs,
    horizons=explainer.horizons,
)
consistency = consistency_report_frame(
    shap_values=shap_values,
    expected_values=expected_values,
    model_outputs=model_outputs,
    contexts=selected_contexts,
    horizons=explainer.horizons,
)

np.save(PATHS.shap_values, shap_values)
np.save(PATHS.shap_expected_values, expected_values)
np.save(PATHS.shap_model_outputs, model_outputs)
context_index.to_csv(PATHS.shap_context_index, index=False)
values_long.to_parquet(PATHS.shap_values_long, index=False)
consistency.to_csv(PATHS.shap_consistency_report, index=False)
audit_log.to_csv(PATHS.shap_audit_log, index=False)

max_abs_error = float(consistency["absolute_approximation_error"].max())
mean_abs_error = float(consistency["absolute_approximation_error"].mean())

print(f"Saved SHAP values: {PATHS.shap_values}")
print(f"Saved long-format SHAP values: {PATHS.shap_values_long}")
print(f"Saved consistency report: {PATHS.shap_consistency_report}")
print(f"Maximum local approximation error: {max_abs_error:,.6f}")
print(f"Mean local approximation error: {mean_abs_error:,.6f}")
display(consistency.head(10))

# %% Cell 7
import json

import numpy as np
import pandas as pd

aggregated_shap_values, aggregated_feature_values, aggregated_feature_names, aggregated_feature_groups = aggregate_by_display(
    shap_values,
    test_values,
    feature_specs,
)
np.save(PATHS.shap_values_aggregated, aggregated_shap_values)
with open(PATHS.shap_aggregated_feature_names_json, "w", encoding="utf-8") as handle:
    json.dump(aggregated_feature_names, handle, indent=2)

global_importance, horizon_importance, group_importance = feature_importance_frames(values_long)
local_explanations = local_trust_specialty_explanations(values_long)

global_importance.to_csv(PATHS.shap_global_importance, index=False)
horizon_importance.to_csv(PATHS.shap_horizon_importance, index=False)
group_importance.to_csv(PATHS.shap_group_importance, index=False)
local_explanations.to_parquet(PATHS.shap_local_explanations, index=False)

print(f"Saved global feature importance: {PATHS.shap_global_importance}")
print(f"Saved horizon feature importance: {PATHS.shap_horizon_importance}")
print(f"Saved grouped feature importance: {PATHS.shap_group_importance}")
print(f"Saved local trust-specialty explanations: {PATHS.shap_local_explanations}")
display(global_importance.head(20))
display(group_importance.head(20))

# %% Cell 8
import matplotlib.pyplot as plt
import numpy as np
import shap

summary_shap = aggregated_shap_values.reshape(-1, aggregated_shap_values.shape[-1])
summary_features = np.repeat(aggregated_feature_values, len(explainer.horizons), axis=0)

plt.figure(figsize=(10, 6))
shap.summary_plot(
    summary_shap,
    features=summary_features,
    feature_names=aggregated_feature_names,
    plot_type="bar",
    show=False,
    max_display=len(aggregated_feature_names),
)
plt.tight_layout()
plt.savefig(PATHS.shap_global_summary, dpi=150, bbox_inches="tight")
plt.close()

print(f"Saved SHAP global summary plot: {PATHS.shap_global_summary}")

# %% Cell 9
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import shap


def make_trust_waterfall_plot(trust_name: str, output_path: Path, horizon: int) -> Dict[str, object]:
    context_frame = context_index.reset_index(drop=True)
    row_indices = context_frame.index[context_frame[COLUMNS.trust_name].astype(str).eq(str(trust_name))].to_numpy(dtype=int)
    if len(row_indices) == 0:
        raise ValueError(f"No selected explanation contexts for Trust: {trust_name}")
    trust_code = str(context_frame.loc[row_indices, COLUMNS.trust_code].dropna().astype(str).iloc[0])
    horizon_position = list(explainer.horizons).index(int(horizon))
    trust_shap = aggregated_shap_values[row_indices, horizon_position, :].mean(axis=0)
    trust_features = aggregated_feature_values[row_indices, :].mean(axis=0)
    trust_base_value = float(expected_values[row_indices, horizon_position].mean())
    trust_model_output = float(model_outputs[row_indices, horizon_position].mean())
    explanation = shap.Explanation(
        values=trust_shap,
        base_values=trust_base_value,
        data=trust_features,
        feature_names=aggregated_feature_names,
    )
    shap.plots.waterfall(explanation, max_display=min(CONFIG.max_waterfall_features, len(aggregated_feature_names)), show=False)
    fig = plt.gcf()
    fig.set_size_inches(10, 6)
    fig.suptitle(f"{trust_name} | P50 horizon {horizon}", y=1.02, fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    contributions = dict(zip(aggregated_feature_names, trust_shap))
    return {
        COLUMNS.trust_code: trust_code,
        COLUMNS.trust_name: trust_name,
        COLUMNS.horizon: int(horizon),
        "plot_path": str(output_path),
        "base_value": trust_base_value,
        "model_output": trust_model_output,
        "sum_feature_contributions": float(trust_shap.sum()),
        "base_plus_sum": float(trust_base_value + trust_shap.sum()),
        "total_waiting_shap": float(contributions.get("total_waiting", 0.0)),
        "total_completed_pathways_shap": float(contributions.get("total_completed_pathways", 0.0)),
        "new_rtt_periods_shap": float(contributions.get("new_rtt_periods", 0.0)),
        "unreported_removals_shap": float(contributions.get("unreported_removals", 0.0)),
    }


def trust_names_for_waterfall_outputs() -> List[str]:
    names: List[str] = []
    for trust_name in top_predicted_trusts[COLUMNS.trust_name].astype(str).head(CONFIG.selected_trust_count):
        if trust_name not in names:
            names.append(trust_name)

    priority_codes = {str(code).strip().upper() for code in CONFIG.priority_trust_codes if str(code).strip()}
    priority_names = {
        " ".join(str(name).strip().upper().split())
        for name in CONFIG.priority_trust_names
        if str(name).strip()
    }
    if priority_codes or priority_names:
        normalised_names = selected_context_index[COLUMNS.trust_name].astype(str).map(lambda value: " ".join(value.strip().upper().split()))
        priority_rows = selected_context_index[
            selected_context_index[COLUMNS.trust_code].astype(str).str.upper().isin(priority_codes)
            | normalised_names.isin(priority_names)
        ].sort_values("selection_p50", ascending=False)
        for trust_name in priority_rows[COLUMNS.trust_name].astype(str):
            if trust_name not in names:
                names.append(trust_name)
    return names


waterfall_records = []
waterfall_horizon = max(explainer.horizons)
waterfall_trust_names = trust_names_for_waterfall_outputs()
for trust_name in waterfall_trust_names:
    output_path = PATHS.outputs_dir / f"shap_trust_{safe_filename(trust_name)}.png"
    waterfall_records.append(make_trust_waterfall_plot(trust_name, output_path, horizon=waterfall_horizon))

waterfall_index = pd.DataFrame(waterfall_records)
waterfall_index.to_csv(PATHS.shap_waterfall_index, index=False)
display(waterfall_index)
print(f"Saved SHAP waterfall index: {PATHS.shap_waterfall_index}")

# %% Cell 10
import pandas as pd

interpretations = make_trust_interpretations(
    local_explanations=local_explanations,
    trust_names=waterfall_trust_names,
    horizon=max(explainer.horizons),
)
interpretations.to_csv(PATHS.shap_interpretations, index=False)

write_methodology_note(PATHS.shap_methodology_note, CONFIG, explainer.horizons)

summary = {
    "model_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
    "model_format": model_config.get("format"),
    "state_dict_path": str(PATHS.tcn_state_dict),
    "model_config_path": str(PATHS.model_config),
    "feature_metadata_path": str(PATHS.feature_metadata),
    "explained_output": "inverse-transformed P50 median forecast",
    "explained_horizons": list(explainer.horizons),
    "selected_contexts": int(len(selected_contexts)),
    "background_contexts_sampled": int(len(background_values)),
    "local_consistency_max_abs_error": float(consistency["absolute_approximation_error"].max()),
    "local_consistency_mean_abs_error": float(consistency["absolute_approximation_error"].mean()),
}

print(json.dumps(summary, indent=2))
print(f"Saved interpretations: {PATHS.shap_interpretations}")
print(f"Saved methodology note: {PATHS.shap_methodology_note}")
display(interpretations)
