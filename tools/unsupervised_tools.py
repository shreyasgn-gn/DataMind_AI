"""
DataMind AI - Unsupervised Learning Engine

Supports:
- autonomous clustering
- anomaly detection
- mixed tabular data
- numerical, low-cardinality categorical, and datetime features
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from tools.profile_tools import detect_id_columns


# ============================================================
# COMMON FEATURE PREPARATION
# ============================================================

def prepare_unsupervised_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Prepare a numeric matrix for clustering/anomaly detection.

    Steps:
    - remove identifier columns
    - convert date-like columns
    - create calendar features
    - keep numeric columns
    - encode low-cardinality categoricals
    - skip very high-cardinality text
    - remove constant columns
    - impute missing values
    """

    if df.empty:
        raise ValueError("Dataset is empty.")

    data = df.copy()

    report: dict[str, Any] = {
        "original_rows": len(data),
        "original_columns": len(data.columns),
        "identifier_columns_removed": [],
        "datetime_features_created": [],
        "categorical_columns_encoded": [],
        "high_cardinality_columns_skipped": [],
        "constant_columns_removed": [],
        "final_features": 0,
    }

    # --------------------------------------------------------
    # Identifier removal
    # --------------------------------------------------------

    try:
        id_columns = detect_id_columns(data)
    except Exception:
        id_columns = []

    id_columns = [
        column
        for column in id_columns
        if column in data.columns
    ]

    if id_columns:
        data = data.drop(
            columns=id_columns
        )

    report[
        "identifier_columns_removed"
    ] = id_columns

    if data.empty:
        raise ValueError(
            "No usable columns remain after identifier removal."
        )

    # --------------------------------------------------------
    # Datetime conversion and features
    # --------------------------------------------------------

    datetime_columns: list[str] = []

    for column in list(data.columns):

        if pd.api.types.is_datetime64_any_dtype(
            data[column]
        ):
            datetime_columns.append(column)
            continue

        column_lower = column.lower()

        if not any(
            token in column_lower
            for token in (
                "date",
                "datetime",
                "timestamp",
                "time",
            )
        ):
            continue

        try:
            converted = pd.to_datetime(
                data[column],
                errors="coerce",
                format="mixed",
                dayfirst=True,
            )

            if converted.notna().mean() >= 0.8:
                data[column] = converted
                datetime_columns.append(column)

        except Exception:
            continue

    for column in datetime_columns:

        prefix = column.lower()

        new_features = {
            f"{prefix}_year": data[column].dt.year,
            f"{prefix}_month": data[column].dt.month,
            f"{prefix}_day": data[column].dt.day,
            f"{prefix}_day_of_week": (
                data[column].dt.dayofweek
            ),
            f"{prefix}_quarter": (
                data[column].dt.quarter
            ),
            f"{prefix}_is_weekend": (
                data[column].dt.dayofweek >= 5
            ).astype(int),
        }

        for feature_name, values in new_features.items():
            data[feature_name] = values

        report[
            "datetime_features_created"
        ].extend(
            list(new_features.keys())
        )

    if datetime_columns:
        data = data.drop(
            columns=datetime_columns
        )

    # --------------------------------------------------------
    # Boolean values
    # --------------------------------------------------------

    bool_columns = data.select_dtypes(
        include=["bool"]
    ).columns.tolist()

    for column in bool_columns:
        data[column] = data[column].astype(int)

    # --------------------------------------------------------
    # Numeric data
    # --------------------------------------------------------

    numeric_columns = data.select_dtypes(
        include=["number"]
    ).columns.tolist()

    numeric_data = data[
        numeric_columns
    ].copy()

    # --------------------------------------------------------
    # Low-cardinality categorical data
    # --------------------------------------------------------

    categorical_columns = data.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    categorical_data: pd.DataFrame | None = None

    usable_categoricals: list[str] = []

    for column in categorical_columns:

        unique_count = data[column].nunique(
            dropna=True
        )

        if 2 <= unique_count <= 30:
            usable_categoricals.append(
                column
            )
        else:
            report[
                "high_cardinality_columns_skipped"
            ].append(column)

    if usable_categoricals:
        categorical_data = pd.get_dummies(
            data[
                usable_categoricals
            ],
            columns=usable_categoricals,
            dummy_na=True,
            dtype=float,
        )

        report[
            "categorical_columns_encoded"
        ] = usable_categoricals

    # --------------------------------------------------------
    # Combine features
    # --------------------------------------------------------

    feature_parts: list[pd.DataFrame] = []

    if not numeric_data.empty:
        feature_parts.append(
            numeric_data
        )

    if (
        categorical_data is not None
        and not categorical_data.empty
    ):
        feature_parts.append(
            categorical_data
        )

    if not feature_parts:
        raise ValueError(
            "No usable numeric or low-cardinality "
            "categorical features remain."
        )

    features = pd.concat(
        feature_parts,
        axis=1,
    )

    features = features.apply(
        pd.to_numeric,
        errors="coerce",
    )

    # --------------------------------------------------------
    # Remove constant columns
    # --------------------------------------------------------

    constant_columns = [
        column
        for column in features.columns
        if features[column].nunique(
            dropna=True
        ) <= 1
    ]

    if constant_columns:
        features = features.drop(
            columns=constant_columns
        )

    report[
        "constant_columns_removed"
    ] = constant_columns

    if features.empty:
        raise ValueError(
            "No variable features remain after preprocessing."
        )

    # --------------------------------------------------------
    # Imputation
    # --------------------------------------------------------

    imputer = SimpleImputer(
        strategy="median"
    )

    values = imputer.fit_transform(
        features
    )

    features = pd.DataFrame(
        values,
        columns=features.columns,
        index=features.index,
    )

    report[
        "final_features"
    ] = len(features.columns)

    return features, report


# ============================================================
# CLUSTER QUALITY
# ============================================================

def _cluster_quality(
    silhouette: float,
) -> str:
    """
    Translate silhouette score into a conservative
    human-readable quality label.
    """

    if silhouette >= 0.50:
        return "strong"

    if silhouette >= 0.30:
        return "reasonable"

    if silhouette >= 0.10:
        return "weak"

    return "very_weak"


# ============================================================
# CLUSTERING
# ============================================================

def run_clustering(
    df: pd.DataFrame,
    random_state: int = 42,
    max_clusters: int = 8,
    min_cluster_fraction: float = 0.02,
) -> dict[str, Any]:
    """
    Automatically select a KMeans configuration.

    Cluster counts producing extremely tiny clusters are rejected
    when a better-balanced alternative exists.
    """

    features, preparation = (
        prepare_unsupervised_features(df)
    )

    row_count = len(features)

    if row_count < 10:
        raise ValueError(
            "At least 10 usable rows are required for clustering."
        )

    scaler = StandardScaler()

    X = scaler.fit_transform(
        features
    )

    # At least 5 rows or 2% of the dataset.
    minimum_cluster_size = max(
        5,
        int(
            np.ceil(
                row_count
                * min_cluster_fraction
            )
        ),
    )

    upper_k = min(
        max_clusters,
        row_count - 1,
    )

    if upper_k < 2:
        raise ValueError(
            "Not enough rows to evaluate clusters."
        )

    valid_candidates: list[dict[str, Any]] = []
    all_candidates: list[dict[str, Any]] = []

    for k in range(2, upper_k + 1):

        model = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=10,
        )

        labels = model.fit_predict(X)

        unique_labels = np.unique(labels)

        if len(unique_labels) < 2:
            continue

        counts = (
            pd.Series(labels)
            .value_counts()
            .sort_index()
        )

        min_size = int(
            counts.min()
        )

        score = float(
            silhouette_score(
                X,
                labels,
            )
        )

        candidate = {
            "k": int(k),
            "silhouette": score,
            "minimum_cluster_size": min_size,
            "cluster_sizes": [
                int(value)
                for value in counts.tolist()
            ],
        }

        all_candidates.append(
            candidate
        )

        if min_size >= minimum_cluster_size:
            valid_candidates.append(
                candidate
            )

    if not all_candidates:
        raise ValueError(
            "Unable to find a valid clustering configuration."
        )

    # Prefer balanced candidates.
    if valid_candidates:
        best = max(
            valid_candidates,
            key=lambda item: item["silhouette"],
        )
        balanced_selection = True
    else:
        # If every K produces a small cluster, use the best
        # silhouette but explicitly mark the result as fragile.
        best = max(
            all_candidates,
            key=lambda item: item["silhouette"],
        )
        balanced_selection = False

    best_k = best["k"]

    model = KMeans(
        n_clusters=best_k,
        random_state=random_state,
        n_init=10,
    )

    labels = model.fit_predict(X)

    cluster_counts = (
        pd.Series(labels)
        .value_counts()
        .sort_index()
        .to_dict()
    )

    cluster_counts = {
        str(int(cluster)): int(count)
        for cluster, count
        in cluster_counts.items()
    }

    silhouette = round(
        float(best["silhouette"]),
        4,
    )

    quality = _cluster_quality(
        silhouette
    )

    warning = None

    if quality == "very_weak":
        warning = (
            "The dataset does not show strong natural "
            "cluster separation. Cluster assignments should "
            "be treated as exploratory rather than definitive."
        )

    if not balanced_selection:
        warning = (
            "All tested cluster counts produced at least "
            "one very small cluster. The selected result "
            "should be treated cautiously."
        )

    return {
        "success": True,
        "method": "KMeans",
        "rows_analyzed": row_count,
        "features_used": len(features.columns),
        "cluster_count": int(best_k),
        "silhouette_score": silhouette,
        "cluster_quality": quality,
        "cluster_balance_guard": balanced_selection,
        "minimum_cluster_size_required": (
            minimum_cluster_size
        ),
        "cluster_sizes": cluster_counts,
        "cluster_scores": {
            str(item["k"]): round(
                float(item["silhouette"]),
                4,
            )
            for item in all_candidates
        },
        "warning": warning,
        "cluster_labels": [
            int(label)
            for label in labels
        ],
        "preparation": preparation,
    }


# ============================================================
# ANOMALY DETECTION
# ============================================================

def run_anomaly_detection(
    df: pd.DataFrame,
    random_state: int = 42,
    contamination: float | str = "adaptive",
) -> dict[str, Any]:
    """
    Detect unusual observations with Isolation Forest.

    'adaptive' uses a conservative 5% expected-anomaly rate
    for medium/large tabular datasets and avoids the zero-anomaly
    behavior that can occur with an unconstrained 'auto' threshold.
    """

    features, preparation = (
        prepare_unsupervised_features(df)
    )

    row_count = len(features)

    if row_count < 10:
        raise ValueError(
            "At least 10 usable rows are required "
            "for anomaly detection."
        )

    scaler = StandardScaler()

    X = scaler.fit_transform(
        features
    )

    # --------------------------------------------------------
    # Choose effective contamination
    # --------------------------------------------------------

    if isinstance(contamination, str):

        if contamination.lower() != "adaptive":
            raise ValueError(
                "contamination must be a float between "
                "0.01 and 0.20, or 'adaptive'."
            )

        # Conservative default for generic tabular data.
        effective_contamination = 0.05

    else:

        effective_contamination = float(
            contamination
        )

        if not 0.01 <= effective_contamination <= 0.20:
            raise ValueError(
                "contamination must be between "
                "0.01 and 0.20."
            )

    model = IsolationForest(
        n_estimators=200,
        contamination=effective_contamination,
        random_state=random_state,
        n_jobs=-1,
    )

    predictions = model.fit_predict(
        X
    )

    decision_scores = model.decision_function(
        X
    )

    anomaly_mask = (
        predictions == -1
    )

    anomaly_scores = (
        -decision_scores
    )

    anomaly_indices = np.where(
        anomaly_mask
    )[0]

    anomaly_count = int(
        len(anomaly_indices)
    )

    anomaly_rate = (
        anomaly_count / row_count
    )

    # --------------------------------------------------------
    # Rank observations by anomaly strength
    # --------------------------------------------------------

    ranking = np.argsort(
        anomaly_scores
    )[::-1]

    top_n = min(
        20,
        row_count,
    )

    top_anomalies = []

    for position in ranking[:top_n]:

        top_anomalies.append(
            {
                "row_index": int(position),
                "anomaly_score": round(
                    float(
                        anomaly_scores[position]
                    ),
                    6,
                ),
                "is_anomaly": bool(
                    anomaly_mask[position]
                ),
            }
        )

    warning = None

    if anomaly_count == 0:
        warning = (
            "No observations crossed the Isolation Forest "
            "anomaly threshold."
        )

    return {
        "success": True,
        "method": "Isolation Forest",
        "rows_analyzed": row_count,
        "features_used": len(features.columns),
        "contamination_strategy": (
            contamination
        ),
        "effective_contamination": round(
            float(effective_contamination),
            4,
        ),
        "anomaly_count": anomaly_count,
        "anomaly_rate": round(
            float(anomaly_rate),
            4,
        ),
        "normal_count": int(
            row_count - anomaly_count
        ),
        "top_anomalies": top_anomalies,
        "warning": warning,
        "preparation": preparation,
    }