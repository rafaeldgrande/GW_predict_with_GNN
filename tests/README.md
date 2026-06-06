# tests — Test Suite

Automated tests for the three main scripts. Tests use a small pre-built HDF5 dataset (`data_for_tests.h5`, 432 samples) so they run quickly without GPU.

---

## Running the tests

```bash
cd tests/
pytest test_train_model.py test_predictions.py test_optimize_hyperparameters.py -v
```

Or run all at once with the provided script:

```bash
bash run_tests.bash
```

---

## Test files

### `test_train_model.py`

Runs `train_model.py` end-to-end with minimal epochs and checks that:
- `out_weights.pth` is created and non-empty
- `logs/` directory and a timestamped log file are created

### `test_optimize_hyperparameters.py`

Runs `optimize_hyperparameters.py` with a small number of trials and checks that:
- `best_params.json` is created with all expected keys (`n_layers`, `dropout_rate`, `mlp_hidden_dim`, `hidden_dim_layer_*`, `learning_rate`)
- `optimization_history.html`, `param_importances.html`, `hyperparameter_slices.html` are created
- `logs/` directory and a timestamped log file are created

### `test_predictions.py`

Runs `predict.py` using the weights produced by `test_train_model.py` and checks that:
- `predictions.npz` is created with `y_pred` and `y_true` keys
- Predictions array is non-empty
- `logs/` directory and a timestamped log file are created

---

## Test data

| File | Description |
|------|-------------|
| `data_for_tests.h5` | Small HDF5 dataset (432 samples, 1PC MoS2). **Do not delete** — required for all tests |
| `data_list.txt` | Points to `data_for_tests.h5` |
| `best_params.json` | Example hyperparameter file used as input to `test_predictions.py` |

---

## Generated files (not committed)

The following are produced by the test runs and ignored by `.gitignore`:

- `out_weights.pth` — model weights from training test
- `predictions.npz` — predictions from prediction test
- `*.html` — Optuna visualization plots
- `logs/` — timestamped log files
