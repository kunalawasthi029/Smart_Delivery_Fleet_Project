'''ml_model.py
Train on ecommerce_logistics_delivery.csv and test on delivery_data.csv.

The two files do not share the same schema. This script:
  1. Prints a feature comparison.
  2. Maps overlapping / convertible columns onto a shared feature space.
  3. Trains a Random Forest on the ecommerce dataset.
  4. Evaluates on delivery_data.csv (held-out test set).
  5. Also reports an in-domain 80/20 split on ecommerce (full feature set).
'''

from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pymongo import MongoClient

BASE_DIR = Path(__file__).resolve().parent
TRAIN_CSV = BASE_DIR / "ecommerce_logistics_delivery.csv"
TEST_CSV = BASE_DIR / "delivery_data.csv"

# Weather categories in delivery_data.csv → minutes, matching the
# ecommerce Weather_Delay_Minutes scale (mostly 0–10, mean ~3.8).
WEATHER_TO_DELAY_MINUTES = {
    "Sunny": 0.0,
    "Windy": 2.0,
    "Fog": 4.0,
    "Rainy": 6.0,
    "Snow": 10.0,
}

# traffic_index in delivery_data.csv is ~0.5–2.5 (unitless).
# Map onto Traffic_Delay_Minutes (~0–60 typical, mean ~22).
TRAFFIC_INDEX_TO_MINUTES = 15.0

SHARED_FEATURES = [
    "distance_km",
    "package_weight_kg",
    "traffic_delay_minutes",
    "weather_delay_minutes",
]


def compare_datasets(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Print column overlap and how each test column is used."""
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)

    print("\n" + "=" * 64)
    print("  DATASET COMPARISON")
    print("=" * 64)
    print(f"  Train : {TRAIN_CSV.name}  ({len(train_df):,} rows, {len(train_df.columns)} cols)")
    print(f"  Test  : {TEST_CSV.name}  ({len(test_df):,} rows, {len(test_df.columns)} cols)")

    exact = sorted(c for c in test_cols if c in train_cols)
    print(f"\n  Exact same column names : {exact if exact else '(none)'}")

    print("\n  Test column              Train counterpart                 Status")
    print("  " + "-" * 70)
    mapping_rows = [
        ("order_id", "Order_ID", "IDs only - not a feature"),
        ("driver_id", "Delivery_Partner", "different IDs - dropped"),
        ("distance_km", "Distance_KM", "MATCH - renamed"),
        ("package_weight_kg", "Package_Weight_KG", "MATCH - renamed (scale differs)"),
        ("traffic_index", "Traffic_Delay_Minutes", "MAPPED - index * 15 -> minutes"),
        ("weather_condition", "Weather_Delay_Minutes", "MAPPED - category -> minutes"),
        ("delivery_status", "Delivery_Status", "different enums - not used as feature"),
        ("is_delayed", "On_Time_Delivery_Flag", "MATCH - inverted flag is the target"),
    ]
    for test_col, train_col, status in mapping_rows:
        print(f"  {test_col:<24} {train_col:<32} {status}")

    unused_train = sorted(
        train_cols
        - {
            "Order_ID",
            "Distance_KM",
            "Package_Weight_KG",
            "Traffic_Delay_Minutes",
            "Weather_Delay_Minutes",
            "On_Time_Delivery_Flag",
            "Delivery_Status",
            "Delivery_Partner",
        }
    )
    print("\n  Train-only columns (cannot be used on delivery_data.csv):")
    print("   ", ", ".join(unused_train))

    print("\n  Shared model features:")
    for name in SHARED_FEATURES:
        print(f"    - {name}")
    print("  Target: is_delayed  (0 = on time, 1 = delayed)")
    print("=" * 64)


def prepare_train(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Align ecommerce columns to the shared feature names."""
    out = pd.DataFrame({
        "distance_km": df["Distance_KM"].astype(float),
        "package_weight_kg": df["Package_Weight_KG"].astype(float),
        "traffic_delay_minutes": df["Traffic_Delay_Minutes"].astype(float),
        "weather_delay_minutes": df["Weather_Delay_Minutes"].astype(float),
        # 1 = delayed when the order was NOT on time
        "is_delayed": (df["On_Time_Delivery_Flag"].astype(int) == 0).astype(int),
    })
    return out[SHARED_FEATURES], out["is_delayed"]


def prepare_test(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Map delivery_data.csv onto the same shared features as training."""
    weather_delay = (
        df["weather_condition"]
        .map(WEATHER_TO_DELAY_MINUTES)
        .fillna(WEATHER_TO_DELAY_MINUTES["Sunny"])
        .astype(float)
    )
    out = pd.DataFrame({
        "distance_km": df["distance_km"].astype(float),
        "package_weight_kg": df["package_weight_kg"].astype(float),
        "traffic_delay_minutes": df["traffic_index"].astype(float) * TRAFFIC_INDEX_TO_MINUTES,
        "weather_delay_minutes": weather_delay,
        "is_delayed": df["is_delayed"].astype(int),
    })
    return out[SHARED_FEATURES], out["is_delayed"]


def print_metrics(y_true, preds, title: str, feature_cols: List[str], model=None) -> None:
    print("\n" + "=" * 50)
    print(f"  {title}")
    print("=" * 50)
    print(f"  Accuracy    : {accuracy_score(y_true, preds):.4f}")
    print(f"  Precision   : {precision_score(y_true, preds, zero_division=0):.4f}")
    print(f"  Recall      : {recall_score(y_true, preds, zero_division=0):.4f}")
    print(f"  F1-Score    : {f1_score(y_true, preds, zero_division=0):.4f}")
    print("=" * 50)

    print("\nClassification Report:")
    print(classification_report(y_true, preds, target_names=["On-Time", "Delayed"], zero_division=0))

    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    print("Confusion Matrix:")
    print("              Predicted")
    print("              On-Time  Delayed")
    print(f"Actual On-Time   {cm[0][0]:>5}   {cm[0][1]:>5}")
    print(f"Actual Delayed   {cm[1][0]:>5}   {cm[1][1]:>5}")

    if model is not None and hasattr(model, "feature_importances_"):
        print("\nFeature Importances:")
        importances = sorted(
            zip(feature_cols, model.feature_importances_),
            key=lambda x: x[1],
            reverse=True,
        )
        for name, imp in importances:
            bar = "#" * int(imp * 80)
            print(f"  {name:<28} {imp:.4f} {bar}")


def train_shared_and_test_delivery(train_raw: pd.DataFrame, test_raw: pd.DataFrame) -> None:
    X_train, y_train = prepare_train(train_raw)
    X_test, y_test = prepare_test(test_raw)

    missing = [c for c in SHARED_FEATURES if c not in X_train.columns or c not in X_test.columns]
    if missing:
        raise ValueError(f"Aligned features missing after mapping: {missing}")

    print("\n" + "=" * 64)
    print("  CROSS-DATASET RUN  (train=ecommerce, test=delivery_data.csv)")
    print("=" * 64)
    print(f"  Training samples : {len(X_train)}")
    print(f"  Testing samples  : {len(X_test)}")
    print(f"  Train classes    : On-Time={(y_train == 0).sum()} | Delayed={(y_train == 1).sum()}")
    print(f"  Test classes     : On-Time={(y_test == 0).sum()} | Delayed={(y_test == 1).sum()}")
    print("\n  Feature check (train vs test means):")
    for col in SHARED_FEATURES:
        print(
            f"    {col:<24} train_mean={X_train[col].mean():8.2f}   "
            f"test_mean={X_test[col].mean():8.2f}"
        )

    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    print("\nTraining Random Forest on shared features...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print_metrics(
        y_test,
        preds,
        "EVALUATION ON delivery_data.csv (TEST)",
        SHARED_FEATURES,
        model,
    )


def train_from_mongodb(
    db_name: str = "delivery_fleet",
    collection_name: str = "orders",
) -> None:
    """Train on the ecommerce rows ingested into MongoDB (new dataset)."""
    client = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=5000)
    docs = list(client[db_name][collection_name].find({}, {"_id": 0}))
    if not docs:
        raise ValueError("No data in MongoDB. Run database.py first.")

    df = pd.DataFrame(docs)
    print("\n" + "=" * 64)
    print("  MONGODB RUN  (ecommerce_logistics_delivery ingested into orders)")
    print("=" * 64)
    print(f"  Documents loaded : {len(df)}")

    feature_cols = [
        "distance_km",
        "traffic_delay_minutes",
        "weather_delay_minutes",
        "package_weight_kg",
        "warehouse_processing_h",
        "courier_rating",
        "courier_experience_yrs",
        "warehouse_capacity_pct",
        "weekend_flag",
        "holiday_flag",
        "promotion_flag",
        "order_hour",
        "day_of_week",
        "month",
    ]
    missing = [c for c in feature_cols + ["is_delayed", "shipping_method"] if c not in df.columns]
    if missing:
        raise ValueError(f"MongoDB orders missing expected fields: {missing}")

    le = LabelEncoder()
    df["shipping_method_enc"] = le.fit_transform(df["shipping_method"].astype(str))
    feature_cols = feature_cols + ["shipping_method_enc"]

    X = df[feature_cols]
    y = df["is_delayed"].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Training samples : {len(X_tr)}")
    print(f"  Testing samples  : {len(X_te)}")
    print(f"  Train classes    : On-Time={(y_tr == 0).sum()} | Delayed={(y_tr == 1).sum()}")

    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    print("\nTraining Random Forest on MongoDB ecommerce features...")
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    print_metrics(
        y_te,
        preds,
        "EVALUATION ON MongoDB ecommerce hold-out (is_delayed)",
        feature_cols,
        model,
    )


def train_ecommerce_indomain(train_raw: pd.DataFrame) -> None:
    """Full ecommerce feature set, 80/20 split — in-domain baseline."""
    target = "On_Time_Delivery_Flag"
    drop = [
        "Order_ID",
        "Order_Date",
        "Delivery_Status",
        "Delivery_Delay_Hours",
        "Cancellation_Flag",
        "Return_Flag",
        "On_Time_Delivery_Flag",
    ]
    X = train_raw.drop(columns=[c for c in drop if c in train_raw.columns])
    y = (train_raw[target].astype(int) == 0).astype(int)

    for col in X.select_dtypes(include="object").columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    feature_cols = list(X.columns)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n" + "=" * 64)
    print("  IN-DOMAIN RUN  (ecommerce_logistics_delivery.csv 80/20)")
    print("=" * 64)
    print(f"  Training samples : {len(X_tr)}")
    print(f"  Testing samples  : {len(X_te)}")
    print(f"  Features used    : {len(feature_cols)} (ecommerce-only; not in delivery_data.csv)")

    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    print("\nTraining Random Forest on full ecommerce features...")
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    print_metrics(
        y_te,
        preds,
        "EVALUATION ON ecommerce hold-out (is_delayed)",
        feature_cols,
        model,
    )


if __name__ == "__main__":
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(f"Training file not found: {TRAIN_CSV}")
    if not TEST_CSV.exists():
        raise FileNotFoundError(f"Testing file not found: {TEST_CSV}")

    train_raw = pd.read_csv(TRAIN_CSV)
    test_raw = pd.read_csv(TEST_CSV)

    compare_datasets(train_raw, test_raw)
    train_from_mongodb()
    train_shared_and_test_delivery(train_raw, test_raw)
    train_ecommerce_indomain(train_raw)
