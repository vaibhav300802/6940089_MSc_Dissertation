# Ablation Study Methodology

## Purpose

This experiment measures how much each of three model components contributes to forecast
accuracy, by retraining the TCN with that component removed and comparing it against the
full production model on the same held-out test period.

## Variants

- **Full model (baseline)**: The production model exactly as trained in Layer 1; not retrained here, its existing saved backtest predictions are reused so the comparison is against the real deployed model.
- **No Trust/specialty identity**: Retrained with every Trust and specialty index forced to the same constant value, so the entity embedding layer cannot distinguish one Trust or specialty from another.
- **No lagged features**: Retrained with every 1/3/6-month lagged referral, completion and net-inflow feature removed from the input list.
- **No missingness/imputation flags**: Retrained with every missingness indicator and imputation flag removed from the input list, so the model can no longer tell an observed month from a filled one.

## Protocol

Every variant uses the identical encoder length, forecast horizon, train/validation/test
split, model architecture (hidden channels, TCN levels, kernel size, dropout, embedding
dimension) and random seed as the production Layer 1 model. Only the single named component
differs, so any change in accuracy can be attributed to that component rather than to a
confound. The full-model row is not retrained; it is the existing saved production model's
backtest predictions, recomputed through the same metric functions used for every variant
so the comparison is on equal footing.

## Result

                          label  mae_median  mae_median_pct_change_vs_full_model
          Full model (baseline)  399.422549                             0.000000
    No Trust/specialty identity  313.476940                           -21.517465
             No lagged features  364.207682                            -8.816444
No missingness/imputation flags  432.638077                             8.315887
