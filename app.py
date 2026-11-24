from pathlib import Path
import re
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans


ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "customers.csv"

REQUIRED_COLUMNS = {
    "customer_id",
    "age",
    "gender",
    "region",
    "product_id",
    "product_name",
    "product_category",
    "browsing_minutes",
    "pages_viewed",
    "rating",
    "purchase_count",
    "purchase_amount",
    "purchase",
}
COLUMN_ALIASES = {
    "customerid": "customer_id",
    "customer": "customer_id",
    "productid": "product_id",
    "itemid": "product_id",
    "product": "product_name",
    "item": "product_name",
    "productcategory": "product_category",
    "category": "product_category",
    "itemcategory": "product_category",
    "browsingminutes": "browsing_minutes",
    "sessiondurationminutes": "browsing_minutes",
    "sessionduration": "browsing_minutes",
    "pagesviewed": "pages_viewed",
    "pageviews": "pages_viewed",
    "productrating": "rating",
    "customerrating": "rating",
    "totalpurchases": "purchase_count",
    "numberofpurchases": "purchase_count",
    "purchasefrequency": "purchase_count",
    "purchaseamount": "purchase_amount",
    "amountspent": "purchase_amount",
    "spent": "purchase_amount",
    "purchased": "purchase",
    "ispurchased": "purchase",
    "purchaseflag": "purchase",
    "location": "region",
    "city": "region",
}
NUMERIC_FEATURES = [
    "age",
    "browsing_minutes",
    "pages_viewed",
    "rating",
    "purchase_count",
]
CATEGORICAL_FEATURES = ["gender", "region", "product_category"]
CUSTOMER_CATEGORICAL_FEATURES = ["gender", "region"]

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024


def _normalize_column_name(column):
    normalized = re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_")
    compact = normalized.replace("_", "")
    return COLUMN_ALIASES.get(compact, normalized)


@app.after_request
def allow_local_live_server(response):
    origin = request.headers.get("Origin")
    if origin:
        try:
            parsed_origin = urlparse(origin)
            port = parsed_origin.port
        except ValueError:
            port = None
            parsed_origin = None

        if (
            parsed_origin is not None
            and parsed_origin.scheme == "http"
            and parsed_origin.hostname in {"localhost", "127.0.0.1"}
            and port is not None
            and 5500 <= port <= 5599
        ):
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type"
            response.headers["Vary"] = "Origin"
    return response


def read_dataset(file_storage=None):
    source = file_storage if file_storage is not None else DATA_FILE
    try:
        frame = pd.read_csv(source, sep=None, engine="python", encoding="utf-8-sig")
    except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        raise ValueError("The uploaded file could not be read as a CSV.") from error

    frame.columns = [_normalize_column_name(column) for column in frame.columns]
    if frame.columns.duplicated().any():
        duplicates = sorted(set(frame.columns[frame.columns.duplicated()].tolist()))
        raise ValueError(
            "The CSV has multiple columns that map to the same field: "
            + ", ".join(duplicates)
        )

    missing_columns = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing_columns:
        found_columns = ", ".join(map(str, frame.columns)) or "(none)"
        raise ValueError(
            "CSV columns do not match the expected customer dataset. "
            f"Missing: {', '.join(missing_columns)}. "
            f"Found: {found_columns}. "
            "Use the sample CSV as a template."
        )
    if frame.empty:
        raise ValueError("The CSV must contain at least one data row.")

    original_rows = len(frame)
    frame = frame.drop_duplicates().copy()
    duplicate_rows = original_rows - len(frame)
    cleaned_missing_values = int(frame[list(REQUIRED_COLUMNS)].isna().sum().sum())

    frame["customer_id"] = frame["customer_id"].astype("string").str.strip()
    invalid_customer_rows = int(
        frame["customer_id"].isna().sum() + (frame["customer_id"] == "").sum()
    )
    frame = frame.dropna(subset=["customer_id"])
    frame = frame[frame["customer_id"] != ""].copy()
    if frame.empty:
        raise ValueError("The CSV must contain at least one non-empty customer_id.")

    numeric_columns = NUMERIC_FEATURES + ["purchase_amount", "purchase"]
    for column in numeric_columns:
        original_values = frame[column]
        frame[column] = pd.to_numeric(original_values, errors="coerce")
        non_finite = frame[column].notna() & ~np.isfinite(frame[column])
        cleaned_missing_values += int(
            (frame[column].isna() & original_values.notna()).sum()
        ) + int(non_finite.sum())
        frame.loc[non_finite, column] = np.nan
        if column in {"purchase_amount", "purchase"}:
            frame[column] = frame[column].fillna(0)
        else:
            median = frame[column].median()
            frame[column] = frame[column].fillna(0 if pd.isna(median) else median)

    for column in ["gender", "region", "product_id", "product_name", "product_category"]:
        cleaned_missing_values += int(
            (
                frame[column].notna()
                & frame[column].astype(str).str.strip().eq("")
            ).sum()
        )
        frame[column] = frame[column].fillna("Unknown").astype(str).str.strip()
        frame.loc[frame[column] == "", column] = "Unknown"
    if "purchase_date" in frame.columns:
        frame["purchase_date"] = pd.to_datetime(frame["purchase_date"], errors="coerce")
    else:
        frame["purchase_date"] = pd.NaT

    frame["purchase"] = frame["purchase"].clip(0, 1).round().astype(int)
    frame["purchase_amount"] = frame["purchase_amount"].clip(lower=0)
    frame["purchase_count"] = frame["purchase_count"].clip(lower=0)
    frame["rating"] = frame["rating"].clip(0, 5)
    frame["browsing_minutes"] = frame["browsing_minutes"].clip(lower=0)
    frame["pages_viewed"] = frame["pages_viewed"].clip(lower=0)

    return frame, {
        "duplicate_rows_removed": duplicate_rows,
        "invalid_customer_rows_removed": invalid_customer_rows,
        "missing_values_cleaned": cleaned_missing_values,
    }


def _mode(values):
    modes = values.mode()
    return str(modes.iloc[0]) if not modes.empty else "Unknown"


def _json_number(value, decimals=2):
    if pd.isna(value) or not np.isfinite(value):
        return None
    return round(float(value), decimals)


def _customer_features(frame):
    grouped = frame.groupby("customer_id", sort=True)
    profiles = grouped[NUMERIC_FEATURES].mean().reset_index()
    profiles["gender"] = grouped["gender"].apply(_mode).values
    profiles["region"] = grouped["region"].apply(_mode).values
    profiles["total_spend"] = grouped["purchase_amount"].sum().values
    return profiles


def _recommendations(frame, profiles, profile_index, profile_transformer):
    if len(profiles) < 2:
        return []

    feature_columns = NUMERIC_FEATURES + CUSTOMER_CATEGORICAL_FEATURES
    encoded_profiles = profile_transformer.fit_transform(profiles[feature_columns])
    neighbors_count = min(6, len(profiles))
    neighbor_model = NearestNeighbors(n_neighbors=neighbors_count, metric="euclidean")
    neighbor_model.fit(encoded_profiles)
    distances, indices = neighbor_model.kneighbors(
        encoded_profiles[profile_index : profile_index + 1]
    )

    customer_id = profiles.iloc[profile_index]["customer_id"]
    already_bought = set(
        frame.loc[
            (frame["customer_id"] == customer_id) & (frame["purchase"] == 1),
            "product_id",
        ]
    )
    product_scores = {}
    for distance, neighbor_index in zip(distances[0], indices[0]):
        if neighbor_index == profile_index:
            continue
        neighbor_id = profiles.iloc[neighbor_index]["customer_id"]
        interactions = frame[
            (frame["customer_id"] == neighbor_id) & (frame["purchase"] == 1)
        ]
        if interactions.empty:
            interactions = frame[
                (frame["customer_id"] == neighbor_id) & (frame["rating"] >= 3.5)
            ]
        weight = 1.0 / (1.0 + float(distance))
        for _, row in interactions.iterrows():
            product_id = row["product_id"]
            if product_id in already_bought:
                continue
            entry = product_scores.setdefault(
                product_id,
                {
                    "product_id": product_id,
                    "product_name": row["product_name"],
                    "category": row["product_category"],
                    "weighted_rating": 0.0,
                    "weight": 0.0,
                },
            )
            entry["weighted_rating"] += float(row["rating"]) * weight
            entry["weight"] += weight

    if not product_scores:
        candidates = frame[~frame["product_id"].isin(already_bought)]
        if candidates.empty:
            candidates = frame
        popular = (
            candidates.groupby(["product_id", "product_name", "product_category"])
            .agg(rating=("rating", "mean"), purchases=("purchase", "sum"))
            .reset_index()
            .sort_values(["purchases", "rating"], ascending=False)
            .head(3)
        )
        return [
            {
                "product_id": str(row.product_id),
                "product_name": str(row.product_name),
                "category": str(row.product_category),
                "rating": _json_number(row.rating),
            }
            for row in popular.itertuples(index=False)
        ]

    ranked = []
    for entry in product_scores.values():
        ranked.append(
            {
                "product_id": str(entry["product_id"]),
                "product_name": str(entry["product_name"]),
                "category": str(entry["category"]),
                "rating": _json_number(
                    entry["weighted_rating"] / entry["weight"]
                ),
            }
        )
    return sorted(ranked, key=lambda item: item["rating"] or 0, reverse=True)[:3]


def analyze(frame, cleaning, selected_customer_id=None):
    profiles = _customer_features(frame)
    if len(profiles) < 2:
        raise ValueError("At least two unique customers are needed to train the models.")
    if frame["purchase"].nunique() < 2:
        raise ValueError("The purchase column must contain both 0 and 1 examples.")

    customer_x = profiles[NUMERIC_FEATURES]
    spending_model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )
    spending_model.fit(customer_x, profiles["total_spend"])
    profiles["predicted_spend"] = spending_model.predict(customer_x).clip(min=0)

    customer_count = len(profiles)
    cluster_count = min(3, customer_count)
    segment_model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clusterer", KMeans(n_clusters=cluster_count, random_state=42, n_init=10)),
        ]
    )
    cluster_labels = segment_model.fit_predict(customer_x)
    cluster_spend = profiles.assign(cluster=cluster_labels).groupby("cluster")[
        "total_spend"
    ].mean()
    ordered_clusters = list(cluster_spend.sort_values().index)
    tier_names = ["Occasional", "Regular", "Premium"]
    tier_map = {}
    for rank, cluster in enumerate(ordered_clusters):
        tier_position = (
            round(rank * (len(tier_names) - 1) / (len(ordered_clusters) - 1))
            if len(ordered_clusters) > 1
            else 0
        )
        tier_map[cluster] = tier_names[tier_position]
    profiles["segment"] = [tier_map[label] for label in cluster_labels]

    row_transformer = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    purchase_model = Pipeline(
        [
            ("features", row_transformer),
            (
                "classifier",
                LogisticRegression(max_iter=1000, class_weight="balanced"),
            ),
        ]
    )
    purchase_model.fit(frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES], frame["purchase"])

    profile_transformer = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CUSTOMER_CATEGORICAL_FEATURES,
            ),
        ]
    )
    profile_transformer.fit(
        profiles[NUMERIC_FEATURES + CUSTOMER_CATEGORICAL_FEATURES]
    )
    purchase_input = profiles[NUMERIC_FEATURES].copy()
    purchase_input["product_category"] = profiles["product_category"] = profiles[
        "customer_id"
    ].map(
        frame.groupby("customer_id")["product_category"].apply(_mode)
    )
    purchase_input["gender"] = profiles["gender"]
    purchase_input["region"] = profiles["region"]
    purchase_probabilities = purchase_model.predict_proba(
        purchase_input[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    )[:, 1]

    positive_count = int((frame["purchase"] == 1).sum())
    negative_count = int((frame["purchase"] == 0).sum())
    purchase_cv_accuracy = None
    if min(positive_count, negative_count) >= 2:
        folds = min(5, positive_count, negative_count)
        scores = cross_val_score(
            purchase_model,
            frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES],
            frame["purchase"],
            cv=folds,
            scoring="accuracy",
        )
        purchase_cv_accuracy = _json_number(scores.mean() * 100, 1)

    spending_cv_mae = None
    if customer_count >= 3:
        folds = min(5, customer_count)
        scores = cross_val_score(
            spending_model,
            customer_x,
            profiles["total_spend"],
            cv=KFold(n_splits=folds, shuffle=True, random_state=42),
            scoring="neg_mean_absolute_error",
        )
        spending_cv_mae = _json_number(-scores.mean())

    profiles["purchase_probability"] = purchase_probabilities
    profiles["customer_id"] = profiles["customer_id"].astype(str)
    selected_index = 0
    if selected_customer_id is not None:
        matches = profiles.index[
            profiles["customer_id"] == str(selected_customer_id)
        ].tolist()
        if not matches:
            raise ValueError(f"Customer '{selected_customer_id}' was not found.")
        selected_index = matches[0]

    selected = profiles.iloc[selected_index]
    customer_rows = [
        {
            "customer_id": row.customer_id,
            "spending": _json_number(row.total_spend),
            "predicted_spending": _json_number(row.predicted_spend),
            "purchase_probability": _json_number(row.purchase_probability * 100, 1),
            "segment": row.segment,
        }
        for row in profiles.itertuples(index=False)
    ]

    category_summary = (
        frame.groupby("product_category")
        .agg(revenue=("purchase_amount", "sum"), purchases=("purchase", "sum"))
        .sort_values("revenue", ascending=False)
    )
    monthly_or_category = [
        {
            "label": str(category),
            "revenue": _json_number(row.revenue),
            "purchases": int(row.purchases),
        }
        for category, row in category_summary.iterrows()
    ]
    dated_purchases = frame.dropna(subset=["purchase_date"]).copy()
    if dated_purchases.empty:
        sales_trend = []
    else:
        dated_purchases["period"] = dated_purchases["purchase_date"].dt.to_period("M")
        monthly_sales = dated_purchases.groupby("period")["purchase_amount"].sum()
        sales_trend = [
            {"label": str(period), "revenue": _json_number(revenue)}
            for period, revenue in monthly_sales.items()
        ]

    return {
        "overview": {
            "customer_count": customer_count,
            "interaction_count": int(len(frame)),
            "total_revenue": _json_number(frame["purchase_amount"].sum()),
            "purchase_rate": _json_number(frame["purchase"].mean() * 100, 1),
            "average_rating": _json_number(frame["rating"].mean(), 1),
            **cleaning,
        },
        "model_metrics": {
            "spending_cv_mae": spending_cv_mae,
            "purchase_cv_accuracy": purchase_cv_accuracy,
            "spending_training_customers": customer_count,
            "purchase_training_interactions": int(len(frame)),
        },
        "categories": monthly_or_category,
        "sales_trend": sales_trend,
        "customers": customer_rows,
        "selected_customer": {
            "customer_id": str(selected["customer_id"]),
            "predicted_spending": _json_number(selected["predicted_spend"]),
            "purchase_probability": _json_number(
                selected["purchase_probability"] * 100, 1
            ),
            "segment": str(selected["segment"]),
            "recommendations": _recommendations(
                frame, profiles, selected_index, profile_transformer
            ),
        },
    }


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/sample.csv")
def sample_csv():
    return send_file(DATA_FILE, mimetype="text/csv", as_attachment=True, download_name="customers.csv")


@app.get("/api/analysis")
def get_analysis():
    return _analysis_response(None, request.args.get("customer_id"))


@app.post("/api/analysis")
def upload_analysis():
    uploaded_file = request.files.get("file")
    if uploaded_file is None or not uploaded_file.filename:
        return jsonify({"error": "Choose a CSV file to upload."}), 400
    return _analysis_response(uploaded_file, request.form.get("customer_id"))


def _analysis_response(file_storage, selected_customer_id):
    try:
        frame, cleaning = read_dataset(file_storage)
        return jsonify(analyze(frame, cleaning, selected_customer_id))
    except ValueError as error:
        return jsonify({"error": str(error)}), 400


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({"error": "CSV uploads must be 5 MB or smaller."}), 413


if __name__ == "__main__":
    app.run(debug=True)
