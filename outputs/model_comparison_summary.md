# Model Comparison Summary

Configuration: `{"covid_end": "2021-09-01", "covid_start": "2020-03-01", "enable_hist_gradient_boosting": true, "expensive_baseline_name": "hist_gradient_boosting", "hist_gradient_boosting_learning_rate": 0.05, "hist_gradient_boosting_max_iter": 180, "hist_gradient_boosting_max_train_rows": 350000, "hist_gradient_boosting_min_samples_leaf": 30, "random_seed": 42, "require_complete_core_baselines": true, "seasonal_period": 12, "target_column": "incomplete_total"}`

Strongest model by overall MAE: **naive_last_value** with MAE 274.1613.
TCN outperforms seasonal naive by 9.5949 MAE (2.34% relative improvement).

## Paired Error Analysis

- TCN_vs_naive_last_value: mean comparator-minus-TCN absolute-error difference -126.8297; TCN better on 36.98% of paired rows; sign-test p-value 0.
- TCN_vs_seasonal_naive_12m: mean comparator-minus-TCN absolute-error difference 9.4962; TCN better on 51.76% of paired rows; sign-test p-value 3.664e-12.
- TCN_vs_historical_seasonal_mean: mean comparator-minus-TCN absolute-error difference 294.3789; TCN better on 66.56% of paired rows; sign-test p-value 0.
- TCN_vs_hist_gradient_boosting: mean comparator-minus-TCN absolute-error difference -92.7665; TCN better on 37.99% of paired rows; sign-test p-value 0.

The TCN should only be described as superior for groups or horizons where these metrics support it.