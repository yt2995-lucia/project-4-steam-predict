"""
Supervised model fitting pipeline for Steam game success prediction.

This script loads engineered features and target labels, splits the data into
train / validation / test sets, tunes Logistic Regression, Random Forest, and
XGBoost models, and saves fitted models plus modeling summaries.

Expected inputs:
- data/features.csv
- data/target.csv
- data/unsupervised_output.csv

Generated outputs:
- models/*.joblib
- outputs/modeling/*.csv
- outputs/modeling/modeling_run_metadata.json
"""


from pathlib import Path
import json
import itertools
import joblib

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from xgboost import XGBClassifier


# ============================================================
# 0. Global Settings
# ============================================================

RANDOM_STATE = 42

TRAIN_SIZE = 0.60
VALIDATION_SIZE = 0.20
TEST_SIZE = 0.20

N_SPLITS = 5
SCORING_METRIC = "roc_auc"


# ============================================================
# 1. Path Setup
# ============================================================

def get_project_root():
    """
    Return project root assuming this file is placed under src/.
    """
    return Path(__file__).resolve().parents[1]


def make_paths():
    """
    Define all input and output paths.
    """
    project_root = get_project_root()

    paths = {
        "features": project_root / "data" / "features.csv",
        "target": project_root / "data" / "target.csv",
        "unsupervised_output": project_root / "data" / "unsupervised_output.csv",
        "models_dir": project_root / "models",
        "outputs_dir": project_root / "outputs" / "modeling",
    }

    paths["models_dir"].mkdir(parents=True, exist_ok=True)
    paths["outputs_dir"].mkdir(parents=True, exist_ok=True)

    return paths


# ============================================================
# 2. Data Loading and Preparation
# ============================================================

def load_data(paths):
    """
    Load feature, target, and unsupervised output files.

    The unsupervised file is loaded for record keeping, but features.csv
    should already contain UMAP, topic, and cluster features.
    """
    features = pd.read_csv(paths["features"])
    target = pd.read_csv(paths["target"])
    unsupervised_output = pd.read_csv(paths["unsupervised_output"])

    return features, target, unsupervised_output


def prepare_modeling_data(features, target):
    """
    Merge features and target by appid, then define X and y.
    """
    df = features.merge(target, on="appid", how="inner")

    y = df["is_successful"].astype(int)
    X = df.drop(columns=["appid", "is_successful"])

    return df, X, y


def split_data(df, X, y):
    """
    Split data into train, validation, and test sets.

    Final ratio:
    - Train: 60%
    - Validation: 20%
    - Test: 20%

    The split is stratified so that class distribution is preserved.
    """
    appids = df["appid"]

    X_temp, X_test, y_temp, y_test, appid_temp, appid_test = train_test_split(
        X,
        y,
        appids,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    validation_ratio_within_temp = VALIDATION_SIZE / (TRAIN_SIZE + VALIDATION_SIZE)

    X_train, X_val, y_train, y_val, appid_train, appid_val = train_test_split(
        X_temp,
        y_temp,
        appid_temp,
        test_size=validation_ratio_within_temp,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    split_info = {
        "train": {
            "X": X_train,
            "y": y_train,
            "appid": appid_train,
        },
        "validation": {
            "X": X_val,
            "y": y_val,
            "appid": appid_val,
        },
        "test": {
            "X": X_test,
            "y": y_test,
            "appid": appid_test,
        },
    }

    return split_info


def save_split_appids(split_info, output_path):
    """
    Save appid split assignment for reproducibility.
    """
    rows = []

    for split_name in ["train", "validation", "test"]:
        appids = split_info[split_name]["appid"]

        for appid in appids:
            rows.append({
                "appid": appid,
                "split": split_name,
            })

    split_df = pd.DataFrame(rows)
    split_df.to_csv(output_path, index=False)


# ============================================================
# 3. Validation Metrics
# ============================================================

def compute_validation_metrics(model, X_val, y_val, model_name):
    """
    Compute validation-set metrics after hyperparameter tuning.

    This validation set is used as a quality check, not as final evaluation.
    """
    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = model.predict(X_val)

    metrics = {
        "model": model_name,
        "validation_auc": roc_auc_score(y_val, y_prob),
        "validation_accuracy": accuracy_score(y_val, y_pred),
        "validation_precision": precision_score(y_val, y_pred, zero_division=0),
        "validation_recall": recall_score(y_val, y_pred, zero_division=0),
        "validation_f1": f1_score(y_val, y_pred, zero_division=0),
    }

    return metrics


# ============================================================
# 4. Model 1: Logistic Regression
# ============================================================

def tune_logistic_regression(X_train, y_train, cv):
    """
    Tune Logistic Regression.

    Logistic Regression uses:
    - median imputation
    - standard scaling
    - L1 / L2 regularization search
    """
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="liblinear",
            max_iter=3000,
            random_state=RANDOM_STATE,
        )),
    ])

    param_grid = {
        "clf__C": [0.005, 0.01, 0.03, 0.1, 1],
        "clf__penalty": ["l1", "l2"],
        "clf__class_weight": [None, "balanced"],
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=SCORING_METRIC,
        cv=cv,
        n_jobs=-1,
    )

    search.fit(X_train, y_train)

    return search


# ============================================================
# 5. Model 2: Random Forest
# ============================================================

def tune_random_forest(X_train, y_train, cv):
    """
    Tune Random Forest.

    The tuning grid is refined around the previous best region:
    - relatively shallow trees
    - 300 to 800 trees
    - log2 / sqrt feature sampling
    """
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])

    param_grid = {
        "clf__n_estimators": [300, 500, 800],
        "clf__max_depth": [4, 5, 6, 8],
        "clf__min_samples_leaf": [1, 2, 3],
        "clf__max_features": ["log2", "sqrt"],
        "clf__class_weight": [None, "balanced"],
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=SCORING_METRIC,
        cv=cv,
        n_jobs=-1,
    )

    search.fit(X_train, y_train)

    return search


# ============================================================
# 6. Model 3: XGBoost
# ============================================================

def generate_param_combinations(param_grid):
    """
    Convert parameter grid dictionary into a list of parameter dictionaries.
    """
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    combinations = []

    for combo in itertools.product(*param_values):
        params = dict(zip(param_names, combo))
        combinations.append(params)

    return combinations


def tune_xgboost_manual(X_train, y_train, cv):
    """
    Manually tune XGBoost using 5-fold CV.

    This avoids sklearn / xgboost compatibility issues with some versions.
    The tuning grid is refined around the previous best region:
    - shallow trees
    - moderate learning rate
    - feature subsampling
    """
    param_grid = {
        "n_estimators": [150, 200],
        "max_depth": [2, 3, 4],
        "learning_rate": [0.03, 0.05, 0.07],
        "subsample": [0.9, 1.0],
        "colsample_bytree": [0.8, 0.9],
    }

    param_combinations = generate_param_combinations(param_grid)

    best_auc = -1
    best_params = None
    all_results = []

    for params in param_combinations:
        fold_aucs = []

        for train_index, valid_index in cv.split(X_train, y_train):
            X_tr = X_train.iloc[train_index]
            X_cv = X_train.iloc[valid_index]
            y_tr = y_train.iloc[train_index]
            y_cv = y_train.iloc[valid_index]

            imputer = SimpleImputer(strategy="median")
            X_tr_imputed = imputer.fit_transform(X_tr)
            X_cv_imputed = imputer.transform(X_cv)

            model = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                **params,
            )

            model.fit(X_tr_imputed, y_tr)

            y_cv_prob = model.predict_proba(X_cv_imputed)[:, 1]
            fold_auc = roc_auc_score(y_cv, y_cv_prob)
            fold_aucs.append(fold_auc)

        mean_cv_auc = float(np.mean(fold_aucs))
        std_cv_auc = float(np.std(fold_aucs))

        result_row = params.copy()
        result_row["mean_cv_auc"] = mean_cv_auc
        result_row["std_cv_auc"] = std_cv_auc
        all_results.append(result_row)

        if mean_cv_auc > best_auc:
            best_auc = mean_cv_auc
            best_params = params.copy()

    final_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **best_params,
        )),
    ])

    final_pipeline.fit(X_train, y_train)

    results_df = pd.DataFrame(all_results).sort_values(
        "mean_cv_auc",
        ascending=False,
    )

    return final_pipeline, best_params, best_auc, results_df


# ============================================================
# 7. Summary and Metadata
# ============================================================

def build_summary(logreg_search, rf_search, xgb_best_auc, xgb_best_params,
                  logreg_metrics, rf_metrics, xgb_metrics):
    """
    Build model fitting summary table.
    """
    rows = [
        {
            "model": "Logistic Regression",
            "best_cv_auc_on_training": logreg_search.best_score_,
            "validation_auc": logreg_metrics["validation_auc"],
            "validation_accuracy": logreg_metrics["validation_accuracy"],
            "validation_precision": logreg_metrics["validation_precision"],
            "validation_recall": logreg_metrics["validation_recall"],
            "validation_f1": logreg_metrics["validation_f1"],
            "best_params": json.dumps(logreg_search.best_params_),
        },
        {
            "model": "Random Forest",
            "best_cv_auc_on_training": rf_search.best_score_,
            "validation_auc": rf_metrics["validation_auc"],
            "validation_accuracy": rf_metrics["validation_accuracy"],
            "validation_precision": rf_metrics["validation_precision"],
            "validation_recall": rf_metrics["validation_recall"],
            "validation_f1": rf_metrics["validation_f1"],
            "best_params": json.dumps(rf_search.best_params_),
        },
        {
            "model": "XGBoost",
            "best_cv_auc_on_training": xgb_best_auc,
            "validation_auc": xgb_metrics["validation_auc"],
            "validation_accuracy": xgb_metrics["validation_accuracy"],
            "validation_precision": xgb_metrics["validation_precision"],
            "validation_recall": xgb_metrics["validation_recall"],
            "validation_f1": xgb_metrics["validation_f1"],
            "best_params": json.dumps(xgb_best_params),
        },
    ]

    summary = pd.DataFrame(rows)

    return summary


def build_metadata(features, target, unsupervised_output, df, X, y, split_info):
    """
    Build metadata dictionary for reproducibility.
    """
    metadata = {
        "random_state": RANDOM_STATE,
        "train_size": TRAIN_SIZE,
        "validation_size": VALIDATION_SIZE,
        "test_size": TEST_SIZE,
        "n_splits_cv": N_SPLITS,
        "scoring_metric": SCORING_METRIC,
        "features_shape": list(features.shape),
        "target_shape": list(target.shape),
        "unsupervised_output_shape": list(unsupervised_output.shape),
        "merged_data_shape": list(df.shape),
        "X_shape": list(X.shape),
        "target_distribution": y.value_counts().to_dict(),
        "target_ratio": y.value_counts(normalize=True).to_dict(),
        "split_shapes": {
            "train": list(split_info["train"]["X"].shape),
            "validation": list(split_info["validation"]["X"].shape),
            "test": list(split_info["test"]["X"].shape),
        },
        "split_target_ratios": {
            "train": split_info["train"]["y"].value_counts(normalize=True).to_dict(),
            "validation": split_info["validation"]["y"].value_counts(normalize=True).to_dict(),
            "test": split_info["test"]["y"].value_counts(normalize=True).to_dict(),
        },
        "note": (
            "Train set is used for 5-fold CV hyperparameter tuning. "
            "Validation set is used for quality checking after tuning. "
            "Test set is kept untouched for final model evaluation."
        ),
    }

    return metadata


def save_outputs(paths, logreg_search, rf_search, xgb_model,
                 xgb_results_df, summary, validation_metrics_df,
                 metadata, split_info):
    """
    Save fitted models and modeling output tables.
    """
    models_dir = paths["models_dir"]
    outputs_dir = paths["outputs_dir"]

    joblib.dump(
        logreg_search.best_estimator_,
        models_dir / "logistic_regression_best.joblib",
    )

    joblib.dump(
        rf_search.best_estimator_,
        models_dir / "random_forest_best.joblib",
    )

    joblib.dump(
        xgb_model,
        models_dir / "xgboost_best.joblib",
    )

    summary.to_csv(
        outputs_dir / "model_fitting_summary.csv",
        index=False,
    )

    validation_metrics_df.to_csv(
        outputs_dir / "validation_metrics.csv",
        index=False,
    )

    xgb_results_df.to_csv(
        outputs_dir / "xgboost_cv_results.csv",
        index=False,
    )

    save_split_appids(
        split_info,
        outputs_dir / "split_appids.csv",
    )

    with open(outputs_dir / "modeling_run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)


# ============================================================
# 8. Main Pipeline
# ============================================================

def main():
    paths = make_paths()

    features, target, unsupervised_output = load_data(paths)

    df, X, y = prepare_modeling_data(features, target)

    split_info = split_data(df, X, y)

    X_train = split_info["train"]["X"]
    y_train = split_info["train"]["y"]

    X_val = split_info["validation"]["X"]
    y_val = split_info["validation"]["y"]

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    logreg_search = tune_logistic_regression(X_train, y_train, cv)
    best_logreg_model = logreg_search.best_estimator_

    rf_search = tune_random_forest(X_train, y_train, cv)
    best_rf_model = rf_search.best_estimator_

    best_xgb_model, xgb_best_params, xgb_best_auc, xgb_results_df = tune_xgboost_manual(
        X_train,
        y_train,
        cv,
    )

    logreg_metrics = compute_validation_metrics(
        best_logreg_model,
        X_val,
        y_val,
        "Logistic Regression",
    )

    rf_metrics = compute_validation_metrics(
        best_rf_model,
        X_val,
        y_val,
        "Random Forest",
    )

    xgb_metrics = compute_validation_metrics(
        best_xgb_model,
        X_val,
        y_val,
        "XGBoost",
    )

    validation_metrics_df = pd.DataFrame([
        logreg_metrics,
        rf_metrics,
        xgb_metrics,
    ])

    summary = build_summary(
        logreg_search,
        rf_search,
        xgb_best_auc,
        xgb_best_params,
        logreg_metrics,
        rf_metrics,
        xgb_metrics,
    )

    metadata = build_metadata(
        features,
        target,
        unsupervised_output,
        df,
        X,
        y,
        split_info,
    )

    save_outputs(
        paths,
        logreg_search,
        rf_search,
        best_xgb_model,
        xgb_results_df,
        summary,
        validation_metrics_df,
        metadata,
        split_info,
    )


if __name__ == "__main__":
    main()
