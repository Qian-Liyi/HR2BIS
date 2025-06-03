import os
import pickle

import numpy as np
import pandas as pd

import lightgbm as lgb
from sklearn.model_selection import GroupKFold, GridSearchCV, train_test_split

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    roc_curve,
)


class RobustSigmoidScaler:
    def __init__(self):
        self.median_ = None
        self.iqr_ = None

    def _sigmoid(self, x):
        x = np.clip(x, -500, 500)  # prevent overflow
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

    def fit(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[:, np.newaxis]

        self.median_ = np.median(X, axis=0)
        self.iqr_ = np.subtract(*np.percentile(X, [75, 25], axis=0))
        return self

    def transform(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[:, np.newaxis]

        if self.median_ is None or self.iqr_ is None:
            raise ValueError("Must call fit before transform.")
        scale = self.iqr_ * 1.35 + 1e-9
        sigmoid_X = self._sigmoid((X - self.median_) / scale)
        return sigmoid_X

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def nested_lgbm(X, Y, group, cv, random_state=42):
    outer_cv = GroupKFold(
        n_splits=cv[0],
        shuffle=True,
    )
    inner_cv = GroupKFold(
        n_splits=cv[1],
        shuffle=True,
    )

    scores_acc = []
    scores_auc = []
    scores_f1 = []
    scores_prec = []
    scores_rec = []
    confusion_matrices = []
    roc_curves = []

    importances_split = []
    importances_gain = []

    for train_idx, test_idx in outer_cv.split(X, Y, group):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

        scaler = RobustSigmoidScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        base_model = lgb.LGBMClassifier(
            random_state=random_state,
            learning_rate=0.1,
            n_estimators=200,
            verbose=0,
        )

        inner_splits = list(inner_cv.split(X_train, Y_train, group[train_idx]))
        stage1_param_grid = {
            "learning_rate": [0.01, 0.05, 0.1],
            "n_estimators": [100, 300, 500, 1000],
        }
        stage1_search = GridSearchCV(
            estimator=base_model,
            param_grid=stage1_param_grid,
            scoring="roc_auc",
            cv=inner_splits,
            n_jobs=-1,
            verbose=0,
        )
        stage1_search.fit(X_train, Y_train)
        best_stage1_params = stage1_search.best_params_
        print(f"Best parameters from stage 1: {best_stage1_params}")

        base_model.set_params(**best_stage1_params)
        stage2_param_grid = {
            "num_leaves": [15, 31, 63],
            "min_child_samples": [10, 20, 30],
        }
        stage2_search = GridSearchCV(
            estimator=base_model,
            param_grid=stage2_param_grid,
            scoring="roc_auc",
            cv=inner_splits,
            n_jobs=1,
            verbose=0,
        )
        stage2_search.fit(X_train, Y_train)
        best_params = stage2_search.best_params_
        best_model = stage2_search.best_estimator_
        print(f"Best parameters from stage 2: {best_params}")

        # Prediction
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]

        # Metrics
        acc = accuracy_score(Y_test, y_pred)
        try:
            auc = roc_auc_score(Y_test, y_prob)
        except ValueError:
            auc = np.nan
        f1 = f1_score(Y_test, y_pred, zero_division=0)
        prec = precision_score(Y_test, y_pred, zero_division=0)
        rec = recall_score(Y_test, y_pred, zero_division=0)

        cm = confusion_matrix(Y_test, y_pred)

        scores_acc.append(acc)
        scores_auc.append(auc)
        scores_f1.append(f1)
        scores_prec.append(prec)
        scores_rec.append(rec)
        confusion_matrices.append(cm)

        if len(np.unique(Y_test)) == 2:
            fpr, tpr, _ = roc_curve(Y_test, y_prob)
        else:
            fpr, tpr = np.nan, np.nan
        roc_curves.append((fpr, tpr))

        importances_split.append(best_model.feature_importances_)
        importances_gain.append(
            best_model.booster_.feature_importance(importance_type="gain")
        )

        print(
            f"Outer CV fold: {len(scores_acc)}, "
            f"Accuracy: {acc:.4f}, "
            f"AUC: {auc:.4f}, "
            f"F1: {f1:.4f}, "
            f"Precision: {prec:.4f}, "
            f"Recall: {rec:.4f}"
        )
        print(classification_report(Y_test, y_pred))

    results = {
        "accuracy": scores_acc,
        "auc": scores_auc,
        "f1": scores_f1,
        "precision": scores_prec,
        "recall": scores_rec,
        "confusion_matrices": confusion_matrices,
        "roc_curves": roc_curves,
    }

    importances = {
        "split": np.array(importances_split),
        "gain": np.array(importances_gain),
    }

    return results, importances


group_pairs = [
    ["Negative", "Pre-0min"],
    ["Negative", "Pre-5min"],
    ["Negative", "Pre-10min"],
    ["Negative", "Pre-15min"],
]
for group2analysis in group_pairs:
    hctsa_operations = pd.read_csv("../data/meta/hctsa_operations.csv")
    hctsa_dir = "../data/derivatives/hctsa/BIS_prediction/hctsa_npy/"
    hctsa_paths = [hctsa_dir + f for f in os.listdir(hctsa_dir) if f.endswith(".npy")]
    hctsa_meta_paths = [path.replace(".npy", "_label.txt") for path in hctsa_paths]
    hctsa = [np.load(path) for path in hctsa_paths]
    hctsa = np.concatenate(hctsa, axis=0)
    hctsa_meta = pd.concat(
        [pd.read_csv(path, sep="\t", header=None) for path in hctsa_meta_paths], axis=0
    )

    sub = [label.split("_")[0] for label in hctsa_meta.iloc[:, 0].values]
    group = [label.split("_")[1] for label in hctsa_meta.iloc[:, 0].values]
    hctsa_label = pd.DataFrame(
        {
            "sub": sub,
            "group": group,
            "label": hctsa_meta.iloc[:, 0].values,
        }
    )
    idx2keep = np.isin(hctsa_label["group"], group2analysis)
    print(
        f"Number of epochs in {group2analysis[1]} group: {np.sum(idx2keep)}, total: {len(hctsa_label)}"
    )
    hctsa_label = hctsa_label[idx2keep]
    hctsa = hctsa[idx2keep, :]
    # Remove NaN rows
    rows_with_nan = np.unique(np.where(np.isnan(hctsa))[0])
    hctsa = np.delete(hctsa, rows_with_nan, axis=0)
    hctsa_label = hctsa_label.drop(index=rows_with_nan)

    # Remove constant columns
    IQR = np.percentile(hctsa, 75, axis=0) - np.percentile(hctsa, 25, axis=0)
    hctsa = hctsa[:, IQR > 1e-3]
    hctsa_col_label = hctsa_operations.loc[IQR > 1e-3, :]
    hctsa_col_label.to_csv(
        f"../output/HR_epochs/prediction/hctsa_col_label_{group2analysis[1]}.csv",
        index=False,
    )
    X = hctsa
    y = hctsa_label["group"].values
    y_merged = np.where(np.isin(y, ["Negative"]), 0, 1)
    subs = hctsa_label["sub"].values
    print("Original dataset shape:", X.shape, y_merged.shape, subs.shape)
    # Model training
    results = nested_lgbm(X, y_merged, subs, cv=(10, 10), random_state=42)

    os.makedirs("../output/HR_epochs/prediction", exist_ok=True)
    with open(
        f"../output/HR_epochs/prediction/lightgbm_{group2analysis[1]}.pkl",
        "wb",
    ) as f:
        pickle.dump(results, f)
