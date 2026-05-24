# OpenADMET PXR Integrated Modeling Pipeline

This package provides an integrated modeling pipeline for the OpenADMET PXR Activity Track. It generates molecular features from SMILES, adds PXR-aware descriptors and auxiliary assay-derived signals, trains cross-validated classical regression models, and exports a Hugging Face-compatible submission file.

**Important:** this package intentionally does not include leaderboard-based result optimization. It does not implement post-submission shrink/shift calibration, multiple-submission score tuning, or manual leaderboard feedback adjustment. The output is a direct model prediction.

## Files

```text
pxr_modeling.py                  # modeling script
requirements.txt                 # Python dependencies
README.md                        # this file
```

## Expected data directory

Place the challenge CSV files in one folder, for example `data/`:

```text
data/
  pxr-challenge_TRAIN.csv
  pxr-challenge_TEST_BLINDED.csv
  pxr-challenge_counter-assay_TRAIN.csv
  pxr-challenge_single_concentration_TRAIN.csv
  pxr-challenge_structure_TEST_BLINDED.csv
```

The structure test file is kept for completeness. The integrated activity model does not use blinded labels, because no labels are available for the test set.

## Installation

```bash
pip install -r requirements.txt
```

If RDKit installation through pip fails on your machine, use conda:

```bash
conda install -c conda-forge rdkit
```

## Run

From the package directory:

```bash
python pxr_modeling.py --data_dir data --out_dir pxr_integrated_outputs
```

For a lighter local test run:

```bash
python pxr_modeling.py --data_dir data --out_dir pxr_integrated_outputs --quick
```

## Output

The final submission file is:

```text
pxr_integrated_outputs/submissions/submission_integrated_no_opt.csv
```

It contains exactly the required columns:

```text
SMILES,Molecule Name,pEC50
```

Additional diagnostic files:

```text
pxr_integrated_outputs/oof_metrics.csv
pxr_integrated_outputs/oof_predictions.csv
pxr_integrated_outputs/feature_summary.json
pxr_integrated_outputs/kept_feature_names.csv
```

## Method summary

The model uses four feature blocks:

1. Morgan radius-2 and radius-3 fingerprints, MACCS keys, RDKit descriptors, and PXR-aware SMARTS features.
2. Exact-match aggregate features from single-concentration screening data.
3. Auxiliary molecular predictors trained on counter-assay and single-concentration assay endpoints.
4. Out-of-fold Tanimoto nearest-neighbor features computed without leakage for training rows.

The final predictor is a simple mean ensemble of cross-validated regression models, including Ridge, ExtraTrees, RandomForest, HistGradientBoosting, and optional LightGBM/XGBoost/CatBoost when installed.

No Molecule Name, OCNT_ID, OCNT Batch, plate_id, experiment_name, or source columns are used as direct chemical model features.
