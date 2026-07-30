# Model Comparison Summary

Configuration: `{"covid_end": "2021-09-01", "covid_start": "2020-03-01", "enable_hist_gradient_boosting": true, "expensive_baseline_name": "hist_gradient_boosting", "hist_gradient_boosting_learning_rate": 0.05, "hist_gradient_boosting_max_iter": 180, "hist_gradient_boosting_max_train_rows": 350000, "hist_gradient_boosting_min_samples_leaf": 30, "random_seed": 42, "require_complete_core_baselines": true, "seasonal_period": 12, "target_column": "incomplete_decision_to_admit"}`

Strongest model by overall MAE: **naive_last_value** with MAE 86.6206.
TCN does not outperform seasonal naive on overall MAE; seasonal naive is better by 1.1728 MAE.

## Paired Error Analysis

- TCN_vs_naive_last_value: mean comparator-minus-TCN absolute-error difference -37.8104; TCN better on 38.93% of paired rows; sign-test p-value 3.743e-182.
- TCN_vs_seasonal_naive_12m: mean comparator-minus-TCN absolute-error difference -1.1728; TCN better on 52.22% of paired rows; sign-test p-value 9.159e-09.
- TCN_vs_historical_seasonal_mean: mean comparator-minus-TCN absolute-error difference 72.1012; TCN better on 62.46% of paired rows; sign-test p-value 6.155e-231.
- TCN_vs_hist_gradient_boosting: mean comparator-minus-TCN absolute-error difference -31.1054; TCN better on 38.82% of paired rows; sign-test p-value 6.845e-186.

The TCN should only be described as superior for groups or horizons where these metrics support it.