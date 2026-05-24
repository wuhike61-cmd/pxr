#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Integrated OpenADMET PXR Activity Modeling Pipeline
==================================================

This script trains an integrated, reproducible PXR pEC50 prediction model using:
1) primary PXR dose-response training data,
2) RDKit/Morgan/MACCS molecular representations,
3) PXR-aware physicochemical and SMARTS features,
4) counter-assay and single-concentration auxiliary information,
5) scaffold-aware cross validation,
6) a simple ensemble of classical regression models.

Important: this script intentionally does NOT include leaderboard-based result
optimization such as post-submission shrink/shift calibration, manual score-based
selection, or phase-specific result tuning. The output is a direct model prediction.

Expected data directory:
    data/
      pxr-challenge_TRAIN.csv
      pxr-challenge_TEST_BLINDED.csv
      pxr-challenge_counter-assay_TRAIN.csv
      pxr-challenge_single_concentration_TRAIN.csv
      pxr-challenge_structure_TEST_BLINDED.csv   # optional, not used as labels

Example:
    python pxr_modeling.py --data_dir ../../data --out_dir pxr_integrated_outputs

Output:
    pxr_integrated_outputs/submissions/submission_integrated_no_opt.csv
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    from scipy.stats import spearmanr, kendalltau
except Exception:
    spearmanr = None
    kendalltau = None

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, MACCSkeys, Descriptors, rdMolDescriptors, Lipinski, Crippen, MolSurf
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDKIT_AVAILABLE = True
except Exception as e:
    RDKIT_AVAILABLE = False
    Chem = None
    DataStructs = None
    AllChem = None
    MACCSkeys = None
    Descriptors = None
    rdMolDescriptors = None
    Lipinski = None
    Crippen = None
    MolSurf = None
    MurckoScaffold = None

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
except Exception:
    HAS_CAT = False


PXR_SMARTS: Dict[str, str] = {
    # R region / aromatic stacking
    "fused_aromatic": "c1ccc2ccccc2c1",
    "triphenyl_like": "c1ccc(-c2ccc(-c3ccccc3)cc2)cc1",
    "biaryl": "c1ccc(-c2ccccc2)cc1",
    # L/D hydrophobic motifs
    "tert_butyl": "[CX4]([CH3])([CH3])[CH3]",
    "isopropyl": "[CX4H]([CH3])[CH3]",
    "trifluoromethyl": "C(F)(F)F",
    "tert_butyl_phenol": "Oc1cc([CX4]([CH3])([CH3])[CH3])cc([CX4]([CH3])([CH3])[CH3])c1",
    # hydrogen-bond groups
    "sulfonamide": "S(=O)(=O)N",
    "phosphonate": "P(=O)(O)O",
    "amide": "C(=O)N",
    "carbamate": "OC(=O)N",
    "urea": "NC(=O)N",
    # macrocycles
    "macrocycle_9": "[r9]",
    "macrocycle_10": "[r10]",
    "macrocycle_12_plus": "[r;!r3;!r4;!r5;!r6;!r7;!r8;!r9;!r10;!r11]",
    # heteroaromatics
    "pyridine": "n1ccccc1",
    "imidazole": "c1ncn[c,n]1",
    "pyrimidine": "n1cnccc1",
}

PXR_SMARTS_MOLS = None


def require_rdkit() -> None:
    if not RDKIT_AVAILABLE:
        raise ImportError(
            "RDKit is required for molecular featurization. Install with `pip install rdkit` "
            "or `conda install -c conda-forge rdkit`."
        )


def safe_mol_from_smiles(smiles: str):
    if pd.isna(smiles):
        return None
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        return mol
    except Exception:
        return None


def canonical_smiles(smiles: str) -> Optional[str]:
    mol = safe_mol_from_smiles(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def get_scaffold(smiles: str) -> str:
    mol = safe_mol_from_smiles(smiles)
    if mol is None:
        return "invalid"
    try:
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        return scaf if scaf else "no_scaffold"
    except Exception:
        return "scaffold_error"


def bitvect_to_array(fp, n_bits: Optional[int] = None) -> np.ndarray:
    if n_bits is None:
        n_bits = fp.GetNumBits()
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def morgan_bit_fp(mol, radius: int, n_bits: int):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)


def maccs_fp(mol):
    return MACCSkeys.GenMACCSKeys(mol)


def rdkit_pxr_features_for_mol(mol) -> Dict[str, float]:
    """17 physicochemical descriptors + 36 SMARTS count/flag features."""
    global PXR_SMARTS_MOLS
    if PXR_SMARTS_MOLS is None:
        PXR_SMARTS_MOLS = {name: Chem.MolFromSmarts(sma) for name, sma in PXR_SMARTS.items()}

    feats: Dict[str, float] = {}
    ring_info = mol.GetRingInfo()

    n_rings = rdMolDescriptors.CalcNumRings(mol)
    n_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)

    feats["mw"] = float(Descriptors.MolWt(mol))
    feats["logp"] = float(Crippen.MolLogP(mol))
    feats["tpsa"] = float(MolSurf.TPSA(mol))
    feats["n_heavy_atoms"] = float(mol.GetNumHeavyAtoms())
    feats["n_rotatable"] = float(Lipinski.NumRotatableBonds(mol))
    feats["n_rings"] = float(n_rings)
    feats["n_aromatic_rings"] = float(n_aromatic_rings)
    feats["n_aromatic_carbo"] = float(rdMolDescriptors.CalcNumAromaticCarbocycles(mol))
    feats["n_aromatic_hetero"] = float(rdMolDescriptors.CalcNumAromaticHeterocycles(mol))
    feats["frac_aromatic"] = float(n_aromatic_rings / n_rings) if n_rings else 0.0
    feats["n_HBD"] = float(Lipinski.NumHDonors(mol))
    feats["n_HBA"] = float(Lipinski.NumHAcceptors(mol))
    feats["n_HBD_plus_HBA"] = feats["n_HBD"] + feats["n_HBA"]
    feats["fcsp3"] = float(rdMolDescriptors.CalcFractionCSP3(mol))
    feats["n_stereo"] = float(rdMolDescriptors.CalcNumAtomStereoCenters(mol))
    feats["n_chiral_centers"] = float(len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)))
    feats["formal_charge"] = float(sum(atom.GetFormalCharge() for atom in mol.GetAtoms()))

    # SMARTS pharmacophore features: count + binary flag
    for name, smarts_mol in PXR_SMARTS_MOLS.items():
        if smarts_mol is None:
            count = 0
        else:
            try:
                count = len(mol.GetSubstructMatches(smarts_mol))
            except Exception:
                count = 0
        feats[f"n_{name}"] = float(count)
        feats[f"has_{name}"] = float(count > 0)

    return feats


def featurize_smiles(smiles_list: Sequence[str], n_bits: int = 2048) -> Tuple[np.ndarray, List[str], List[Optional[object]]]:
    """Generate Morgan r2/r3, MACCS, and PXR-aware descriptor features."""
    require_rdkit()
    rows: List[np.ndarray] = []
    feature_names: List[str] = []
    mols: List[Optional[object]] = []

    desc_names = None
    for smi in smiles_list:
        mol = safe_mol_from_smiles(smi)
        mols.append(mol)
        if mol is None:
            # Build a zero row using a dummy methane molecule to infer feature length.
            mol_tmp = Chem.MolFromSmiles("C")
            f_m2 = np.zeros(n_bits, dtype=np.float32)
            f_m3 = np.zeros(n_bits, dtype=np.float32)
            f_maccs = np.zeros(167, dtype=np.float32)
            desc_dict = rdkit_pxr_features_for_mol(mol_tmp)
            desc = np.zeros(len(desc_dict), dtype=np.float32)
            if desc_names is None:
                desc_names = list(desc_dict.keys())
        else:
            fp2 = morgan_bit_fp(mol, radius=2, n_bits=n_bits)
            fp3 = morgan_bit_fp(mol, radius=3, n_bits=n_bits)
            f_m2 = bitvect_to_array(fp2, n_bits=n_bits).astype(np.float32)
            f_m3 = bitvect_to_array(fp3, n_bits=n_bits).astype(np.float32)
            f_maccs = bitvect_to_array(maccs_fp(mol), n_bits=167).astype(np.float32)
            desc_dict = rdkit_pxr_features_for_mol(mol)
            if desc_names is None:
                desc_names = list(desc_dict.keys())
            desc = np.array([desc_dict.get(k, 0.0) for k in desc_names], dtype=np.float32)

        rows.append(np.concatenate([f_m2, f_m3, f_maccs, desc], axis=0))

    if desc_names is None:
        desc_names = []
    feature_names = (
        [f"morgan_r2_{i}" for i in range(n_bits)]
        + [f"morgan_r3_{i}" for i in range(n_bits)]
        + [f"maccs_{i}" for i in range(167)]
        + desc_names
    )
    return np.vstack(rows).astype(np.float32), feature_names, mols


def prepare_primary_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(data_dir / "pxr-challenge_TRAIN.csv")
    test = pd.read_csv(data_dir / "pxr-challenge_TEST_BLINDED.csv")
    for df in [train, test]:
        df["canonical_smiles"] = df["SMILES"].map(canonical_smiles)
        df["valid_smiles"] = df["canonical_smiles"].notna()
    before = len(train)
    train = train[train["valid_smiles"]].copy().reset_index(drop=True)
    if len(train) < before:
        print(f"[Clean] Dropped {before - len(train)} invalid primary training SMILES.")
    if not test["valid_smiles"].all():
        n_bad = (~test["valid_smiles"]).sum()
        print(f"[Warning] {n_bad} invalid test SMILES found; zero features will be used for those rows.")
    test = test.copy().reset_index(drop=True)
    return train, test


def load_auxiliary_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    counter_path = data_dir / "pxr-challenge_counter-assay_TRAIN.csv"
    single_path = data_dir / "pxr-challenge_single_concentration_TRAIN.csv"
    counter = pd.read_csv(counter_path) if counter_path.exists() else pd.DataFrame()
    single = pd.read_csv(single_path) if single_path.exists() else pd.DataFrame()
    if not counter.empty:
        counter["canonical_smiles"] = counter["SMILES"].map(canonical_smiles)
        counter = counter[counter["canonical_smiles"].notna()].copy()
    if not single.empty:
        single["canonical_smiles"] = single["SMILES"].map(canonical_smiles)
        single = single[single["canonical_smiles"].notna()].copy()
    return counter, single


def aggregate_single_concentration(single: pd.DataFrame) -> pd.DataFrame:
    if single.empty:
        return pd.DataFrame(columns=["canonical_smiles"])
    df = single.copy()
    for col in ["log2_fc_estimate", "median_log2_fc", "neg_log10_fdr", "concentration_M", "n_replicates", "log2_fc_stderr"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "fdr_bh" in df.columns:
        df["is_fdr05"] = (pd.to_numeric(df["fdr_bh"], errors="coerce") < 0.05).astype(float)
    else:
        df["is_fdr05"] = 0.0
    agg = df.groupby("canonical_smiles").agg(
        single_n_records=("canonical_smiles", "size"),
        single_log2fc_mean=("log2_fc_estimate", "mean"),
        single_log2fc_max=("log2_fc_estimate", "max"),
        single_log2fc_min=("log2_fc_estimate", "min"),
        single_median_log2fc_mean=("median_log2_fc", "mean"),
        single_neglogfdr_max=("neg_log10_fdr", "max"),
        single_stderr_mean=("log2_fc_stderr", "mean"),
        single_conc_mean=("concentration_M", "mean"),
        single_conc_max=("concentration_M", "max"),
        single_reps_sum=("n_replicates", "sum"),
        single_frac_fdr05=("is_fdr05", "mean"),
    ).reset_index()
    return agg


def merge_exact_aux_features(base: pd.DataFrame, single_agg: pd.DataFrame) -> pd.DataFrame:
    out = base[["canonical_smiles"]].copy()
    if single_agg.empty or len(single_agg.columns) <= 1:
        out["single_has_exact"] = 0.0
        return out.drop(columns=["canonical_smiles"])
    merged = out.merge(single_agg, on="canonical_smiles", how="left")
    merged["single_has_exact"] = merged["single_n_records"].notna().astype(float)
    for col in merged.columns:
        if col == "canonical_smiles":
            continue
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    return merged.drop(columns=["canonical_smiles"])


def train_auxiliary_predictor(
    aux_df: pd.DataFrame,
    target_col: str,
    target_smiles: Sequence[str],
    n_bits: int,
    seed: int,
    feature_label: str,
) -> np.ndarray:
    """Train a molecular predictor on an auxiliary target and predict it for target_smiles."""
    if aux_df.empty or target_col not in aux_df.columns:
        return np.full(len(target_smiles), np.nan, dtype=np.float32)
    df = aux_df[["canonical_smiles", target_col]].copy()
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=["canonical_smiles", target_col])
    if df.empty or df["canonical_smiles"].nunique() < 30:
        return np.full(len(target_smiles), np.nan, dtype=np.float32)
    df = df.groupby("canonical_smiles", as_index=False)[target_col].mean()
    X_aux, _, _ = featurize_smiles(df["canonical_smiles"].tolist(), n_bits=n_bits)
    y_aux = df[target_col].values.astype(np.float32)
    X_target, _, _ = featurize_smiles(list(target_smiles), n_bits=n_bits)
    model = ExtraTreesRegressor(
        n_estimators=300,
        max_features="sqrt",
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
    )
    pipe = make_pipeline(SimpleImputer(strategy="median"), model)
    print(f"[Aux] Training auxiliary predictor for {feature_label}: {len(df)} unique molecules")
    pipe.fit(X_aux, y_aux)
    return pipe.predict(X_target).astype(np.float32)


def build_auxiliary_prediction_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    counter: pd.DataFrame,
    single: pd.DataFrame,
    n_bits: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    target_smiles = pd.concat([train["canonical_smiles"], test["canonical_smiles"]], axis=0).tolist()
    n_train = len(train)
    feats_all: Dict[str, np.ndarray] = {}

    # Counter-assay predictive signatures.
    if not counter.empty:
        for target_col, name in [
            ("pEC50", "aux_counter_pEC50_pred"),
            ("Emax_estimate (log2FC vs. baseline)", "aux_counter_Emax_pred"),
            ("Emax.vs.pos.ctrl_estimate (dimensionless)", "aux_counter_Emax_posctrl_pred"),
        ]:
            if target_col in counter.columns:
                feats_all[name] = train_auxiliary_predictor(counter, target_col, target_smiles, n_bits, seed, name)

    # Single concentration predictive signatures.
    single_agg = aggregate_single_concentration(single)
    if not single_agg.empty and len(single_agg.columns) > 1:
        single_aux = single_agg.rename(columns={"single_log2fc_max": "target_single_log2fc_max"})
        if "target_single_log2fc_max" in single_aux.columns:
            feats_all["aux_single_log2fc_max_pred"] = train_auxiliary_predictor(
                single_aux[["canonical_smiles", "target_single_log2fc_max"]],
                "target_single_log2fc_max",
                target_smiles,
                n_bits,
                seed,
                "aux_single_log2fc_max_pred",
            )
        if "single_neglogfdr_max" in single_agg.columns:
            single_aux2 = single_agg.rename(columns={"single_neglogfdr_max": "target_single_neglogfdr_max"})
            feats_all["aux_single_neglogfdr_pred"] = train_auxiliary_predictor(
                single_aux2[["canonical_smiles", "target_single_neglogfdr_max"]],
                "target_single_neglogfdr_max",
                target_smiles,
                n_bits,
                seed,
                "aux_single_neglogfdr_pred",
            )

    if not feats_all:
        train_df = pd.DataFrame(index=np.arange(n_train))
        test_df = pd.DataFrame(index=np.arange(len(test)))
        return train_df, test_df, []

    aux_all = pd.DataFrame(feats_all)
    # Missing aux predictions happen only when a target was unavailable; fill later by imputer.
    return aux_all.iloc[:n_train].reset_index(drop=True), aux_all.iloc[n_train:].reset_index(drop=True), list(aux_all.columns)


def generate_cv_indices(train: pd.DataFrame, n_splits: int, seed: int) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Optional[np.ndarray]]:
    scaffolds = train["canonical_smiles"].map(get_scaffold).astype(str).values
    unique_groups = len(set(scaffolds))
    indices = np.arange(len(train))
    if unique_groups >= n_splits:
        splitter = GroupKFold(n_splits=n_splits)
        splits = list(splitter.split(indices, train["pEC50"].values, groups=scaffolds))
        print(f"[CV] Using GroupKFold by Murcko scaffold: {n_splits} folds, {unique_groups} scaffolds")
        return splits, scaffolds
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(splitter.split(indices))
    print(f"[CV] Using shuffled KFold: {n_splits} folds")
    return splits, None


def compute_knn_features_for_query(
    query_fps: Sequence[object],
    ref_fps: Sequence[object],
    ref_y: np.ndarray,
    top_k: int = 10,
) -> np.ndarray:
    rows = []
    for qfp in query_fps:
        sims = np.array(DataStructs.BulkTanimotoSimilarity(qfp, list(ref_fps)), dtype=np.float32)
        if len(sims) == 0:
            rows.append(np.zeros(10, dtype=np.float32))
            continue
        order = np.argsort(-sims)[: min(top_k, len(sims))]
        top_s = sims[order]
        top_y = ref_y[order]
        weights = top_s + 1e-6
        rows.append(
            np.array(
                [
                    top_s[0],
                    top_y[0],
                    np.mean(top_s[:3]),
                    np.mean(top_y[:3]),
                    np.mean(top_s[:5]),
                    np.mean(top_y[:5]),
                    np.mean(top_s),
                    np.mean(top_y),
                    np.std(top_y),
                    np.sum(weights * top_y) / np.sum(weights),
                ],
                dtype=np.float32,
            )
        )
    return np.vstack(rows)


def build_tanimoto_knn_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    top_k: int = 10,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    require_rdkit()
    y = train["pEC50"].values.astype(np.float32)
    train_mols = [safe_mol_from_smiles(s) for s in train["canonical_smiles"].tolist()]
    test_mols = [safe_mol_from_smiles(s) for s in test["canonical_smiles"].tolist()]
    train_fps = [morgan_bit_fp(mol, radius=2, n_bits=2048) if mol is not None else morgan_bit_fp(Chem.MolFromSmiles("C"), 2, 2048) for mol in train_mols]
    test_fps = [morgan_bit_fp(mol, radius=2, n_bits=2048) if mol is not None else morgan_bit_fp(Chem.MolFromSmiles("C"), 2, 2048) for mol in test_mols]

    names = [
        "knn_top1_sim",
        "knn_top1_pEC50",
        "knn_top3_sim_mean",
        "knn_top3_pEC50_mean",
        "knn_top5_sim_mean",
        "knn_top5_pEC50_mean",
        "knn_top10_sim_mean",
        "knn_top10_pEC50_mean",
        "knn_top10_pEC50_std",
        "knn_top10_pEC50_weighted",
    ]
    oof = np.zeros((len(train), len(names)), dtype=np.float32)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        print(f"[KNN] Fold {fold}: reference={len(tr_idx)}, query={len(va_idx)}")
        ref_fps = [train_fps[i] for i in tr_idx]
        ref_y = y[tr_idx]
        query_fps = [train_fps[i] for i in va_idx]
        oof[va_idx] = compute_knn_features_for_query(query_fps, ref_fps, ref_y, top_k=top_k)
    test_knn = compute_knn_features_for_query(test_fps, train_fps, y, top_k=top_k)
    return oof, test_knn, names


def get_model_pool(seed: int, quick: bool = False) -> Dict[str, object]:
    models: Dict[str, object] = {}
    models["ridge"] = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(with_mean=False), Ridge(alpha=10.0))
    models["extratrees"] = make_pipeline(
        SimpleImputer(strategy="median"),
        ExtraTreesRegressor(
            n_estimators=250 if quick else 500,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        ),
    )
    models["rf"] = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(
            n_estimators=150 if quick else 300,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=seed + 1,
            n_jobs=-1,
        ),
    )
    models["histgb"] = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingRegressor(max_iter=150 if quick else 300, learning_rate=0.05, random_state=seed),
    )
    if HAS_LGBM:
        models["lgbm"] = make_pipeline(
            SimpleImputer(strategy="median"),
            LGBMRegressor(
                n_estimators=300 if quick else 700,
                learning_rate=0.03,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.7,
                reg_alpha=0.05,
                reg_lambda=1.0,
                random_state=seed,
                n_jobs=-1,
                verbose=-1,
            ),
        )
    if HAS_XGB:
        models["xgb"] = make_pipeline(
            SimpleImputer(strategy="median"),
            XGBRegressor(
                n_estimators=250 if quick else 600,
                learning_rate=0.03,
                max_depth=5,
                subsample=0.85,
                colsample_bytree=0.75,
                objective="reg:squarederror",
                random_state=seed,
                n_jobs=-1,
            ),
        )
    if HAS_CAT and not quick:
        models["catboost"] = make_pipeline(
            SimpleImputer(strategy="median"),
            CatBoostRegressor(
                iterations=500,
                learning_rate=0.03,
                depth=6,
                loss_function="RMSE",
                random_seed=seed,
                verbose=False,
            ),
        )
    return models


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    out = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }
    if spearmanr is not None:
        out["spearman"] = float(spearmanr(y_true, y_pred).correlation)
    if kendalltau is not None:
        out["kendall"] = float(kendalltau(y_true, y_pred).correlation)
    return out


def train_cv_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    seed: int,
    quick: bool,
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    models = get_model_pool(seed=seed, quick=quick)
    print(f"[Model] Training model pool: {list(models.keys())}")
    oof_pred = pd.DataFrame(index=np.arange(len(y)))
    test_pred = pd.DataFrame(index=np.arange(X_test.shape[0]))
    metrics_rows = []

    for name, model in models.items():
        print(f"\n[Model] {name}")
        oof = np.zeros(len(y), dtype=np.float32)
        test_fold_preds = []
        for fold, (tr_idx, va_idx) in enumerate(splits):
            est = clone(model)
            est.fit(X[tr_idx], y[tr_idx])
            pred_va = est.predict(X[va_idx]).astype(np.float32)
            pred_te = est.predict(X_test).astype(np.float32)
            oof[va_idx] = pred_va
            test_fold_preds.append(pred_te)
            m = regression_metrics(y[va_idx], pred_va)
            print(f"  fold={fold} mae={m['mae']:.4f} rmse={m['rmse']:.4f} r2={m['r2']:.4f}")
        oof_pred[name] = oof
        test_pred[name] = np.mean(np.vstack(test_fold_preds), axis=0)
        overall = regression_metrics(y, oof)
        overall["model"] = name
        metrics_rows.append(overall)
        print(f"  OOF {name}: MAE={overall['mae']:.4f}, RMSE={overall['rmse']:.4f}, R2={overall['r2']:.4f}")

    # Direct model ensemble. No leaderboard-based calibration, shrink/shift, or post-hoc tuning is applied.
    oof_pred["ensemble_mean"] = oof_pred[list(models.keys())].mean(axis=1)
    test_pred["ensemble_mean"] = test_pred[list(models.keys())].mean(axis=1)
    ens = regression_metrics(y, oof_pred["ensemble_mean"].values)
    ens["model"] = "ensemble_mean"
    metrics_rows.append(ens)
    print(f"\n[Ensemble] OOF MAE={ens['mae']:.4f}, RMSE={ens['rmse']:.4f}, R2={ens['r2']:.4f}")
    return oof_pred, test_pred["ensemble_mean"].values.astype(np.float32), pd.DataFrame(metrics_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Integrated OpenADMET PXR pEC50 modeling pipeline without result optimization.")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing challenge CSV files.")
    parser.add_argument("--out_dir", type=str, default="pxr_integrated_outputs", help="Output directory.")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of CV folds.")
    parser.add_argument("--n_bits", type=int, default=2048, help="Morgan fingerprint length.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed.")
    parser.add_argument("--quick", action="store_true", help="Use a lighter model pool for local testing.")
    args = parser.parse_args()

    require_rdkit()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    feat_dir = out_dir / "features"
    sub_dir = out_dir / "submissions"
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_dir.mkdir(exist_ok=True)
    sub_dir.mkdir(exist_ok=True)

    print("[Load] Reading primary data")
    train, test = prepare_primary_data(data_dir)
    counter, single = load_auxiliary_data(data_dir)
    y = pd.to_numeric(train["pEC50"], errors="coerce").values.astype(np.float32)

    splits, scaffolds = generate_cv_indices(train, args.n_splits, args.seed)

    print("[Feature] Molecular fingerprints + RDKit/PXR descriptors")
    X_train_base, feature_names, _ = featurize_smiles(train["canonical_smiles"].tolist(), n_bits=args.n_bits)
    X_test_base, _, _ = featurize_smiles(test["canonical_smiles"].tolist(), n_bits=args.n_bits)

    print("[Feature] Exact-match single-concentration aggregate features")
    single_agg = aggregate_single_concentration(single)
    train_single_exact = merge_exact_aux_features(train, single_agg)
    test_single_exact = merge_exact_aux_features(test, single_agg)
    exact_aux_names = train_single_exact.columns.tolist()

    print("[Feature] Auxiliary assay prediction features")
    train_aux_pred, test_aux_pred, aux_pred_names = build_auxiliary_prediction_features(
        train, test, counter, single, n_bits=min(args.n_bits, 1024), seed=args.seed
    )

    print("[Feature] Tanimoto nearest-neighbor features")
    X_train_knn, X_test_knn, knn_names = build_tanimoto_knn_features(train, test, splits=splits, top_k=10)

    X_train = np.hstack([
        X_train_base,
        train_single_exact.values.astype(np.float32),
        train_aux_pred.values.astype(np.float32),
        X_train_knn,
    ]).astype(np.float32)
    X_test = np.hstack([
        X_test_base,
        test_single_exact.values.astype(np.float32),
        test_aux_pred.values.astype(np.float32),
        X_test_knn,
    ]).astype(np.float32)
    all_feature_names = feature_names + exact_aux_names + aux_pred_names + knn_names

    print(f"[Feature] X_train={X_train.shape}, X_test={X_test.shape}")

    # Remove constant columns learned from training set only.
    finite_mask = np.isfinite(X_train).any(axis=0)
    # For NaN columns, imputer can handle them, but fully non-finite columns are dropped.
    std = np.nanstd(np.where(np.isfinite(X_train), X_train, np.nan), axis=0)
    var_mask = np.nan_to_num(std, nan=0.0) > 0
    keep = finite_mask & var_mask
    X_train = X_train[:, keep]
    X_test = X_test[:, keep]
    kept_names = [n for n, k in zip(all_feature_names, keep) if k]
    print(f"[Feature] Kept non-constant features: {len(kept_names)}")

    print("[Train] Cross-validated training")
    oof_pred, test_pred, metrics_df = train_cv_ensemble(X_train, y, X_test, splits, seed=args.seed, quick=args.quick)

    # Save artifacts.
    metrics_df.to_csv(out_dir / "oof_metrics.csv", index=False)
    oof_out = train[["Molecule Name", "SMILES", "canonical_smiles", "pEC50"]].copy()
    oof_out = pd.concat([oof_out, oof_pred], axis=1)
    oof_out.to_csv(out_dir / "oof_predictions.csv", index=False)

    feature_summary = {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_features_before_filter": int(len(all_feature_names)),
        "n_features_after_filter": int(len(kept_names)),
        "feature_blocks": {
            "base_morgan_rdkit_pxr": int(len(feature_names)),
            "single_exact_aggregate": int(len(exact_aux_names)),
            "auxiliary_predictions": int(len(aux_pred_names)),
            "tanimoto_knn": int(len(knn_names)),
        },
        "models": list(get_model_pool(args.seed, args.quick).keys()),
        "note": "No leaderboard-based result optimization, post-hoc shrink/shift calibration, or manual score tuning is applied.",
    }
    with open(out_dir / "feature_summary.json", "w", encoding="utf-8") as f:
        json.dump(feature_summary, f, ensure_ascii=False, indent=2)
    pd.Series(kept_names).to_csv(out_dir / "kept_feature_names.csv", index=False, header=["feature"])

    sub = test[["SMILES", "Molecule Name"]].copy()
    sub["pEC50"] = test_pred
    # Safety clipping to broad physicochemical range, not leaderboard tuning.
    sub["pEC50"] = sub["pEC50"].clip(1.0, 9.0)
    sub_path = sub_dir / "submission_integrated_no_opt.csv"
    sub.to_csv(sub_path, index=False)
    print(f"[Done] Submission saved: {sub_path}")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
