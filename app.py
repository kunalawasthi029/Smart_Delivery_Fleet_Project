'''app.py
Flask API + dashboard for the ecommerce logistics MongoDB dataset.
'''

from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from pymongo import MongoClient
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=str(BASE_DIR), static_url_path="")

FEATURE_COLS = [
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
    "shipping_method_enc",
]

MODEL_STATE = {
    "model": None,
    "encoder": None,
    "metrics": {},
    "ready": False,
    "error": None,
}


def get_db():
    client = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=4000)
    return client["delivery_fleet"]


def risk_from_status(status: str, traffic_min: float) -> str:
    if status == "Severely Delayed":
        return "Critical"
    if status == "Delayed":
        return "High"
    if status == "Cancelled":
        return "Medium"
    if traffic_min > 50:
        return "Medium"
    return "Low"


def pretty_driver(driver_id: str, name: str | None = None) -> str:
    if name:
        return name
    return driver_id.replace("_", " ").title().replace("In-House", "In-House")


def train_model() -> None:
    try:
        db = get_db()
        docs = list(db["orders"].find({}, {"_id": 0, "checkpoint_logs": 0}))
        if not docs:
            MODEL_STATE["error"] = "No orders in MongoDB. Run database.py first."
            return
        df = pd.DataFrame(docs)
        encoder = LabelEncoder()
        df["shipping_method_enc"] = encoder.fit_transform(df["shipping_method"].astype(str))
        X = df[FEATURE_COLS]
        y = df["is_delayed"].astype(int)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        MODEL_STATE["model"] = model
        MODEL_STATE["encoder"] = encoder
        MODEL_STATE["metrics"] = {
            "accuracy": round(float(accuracy_score(y_te, preds)) * 100, 1),
            "precision": round(float(precision_score(y_te, preds, zero_division=0)) * 100, 1),
            "recall": round(float(recall_score(y_te, preds, zero_division=0)) * 100, 1),
            "f1": round(float(f1_score(y_te, preds, zero_division=0)) * 100, 1),
            "train_rows": int(len(X_tr)),
            "test_rows": int(len(X_te)),
        }
        MODEL_STATE["ready"] = True
        MODEL_STATE["error"] = None
    except Exception as exc:
        MODEL_STATE["error"] = str(exc)
        MODEL_STATE["ready"] = False


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "dashboard.html")


@app.route("/api/health")
def health():
    try:
        db = get_db()
        db.command("ping")
        orders = db["orders"].count_documents({})
        drivers = db["drivers"].count_documents({})
        return jsonify({
            "ok": True,
            "mongodb": True,
            "orders": orders,
            "drivers": drivers,
            "model_ready": MODEL_STATE["ready"],
            "model_error": MODEL_STATE["error"],
        })
    except Exception as exc:
        return jsonify({"ok": False, "mongodb": False, "error": str(exc)}), 503


@app.route("/api/stats")
def stats():
    db = get_db()
    orders = db["orders"]
    total = orders.count_documents({})
    if total == 0:
        return jsonify({"error": "No orders in MongoDB"}), 404

    status_counts = {
        d["_id"]: d["n"]
        for d in orders.aggregate([{"$group": {"_id": "$delivery_status", "n": {"$sum": 1}}}])
    }
    on_time = status_counts.get("On Time", 0)
    delayed = status_counts.get("Delayed", 0)
    severe = status_counts.get("Severely Delayed", 0)
    cancelled = status_counts.get("Cancelled", 0)

    avg_delay = list(orders.aggregate([
        {"$group": {"_id": None, "avg": {"$avg": "$delivery_delay_hours"}}}
    ]))
    avg_transit = round(float(avg_delay[0]["avg"]), 1) if avg_delay else 0

    volume = [0] * 7
    for d in orders.aggregate([
        {"$group": {"_id": "$day_of_week", "n": {"$sum": 1}}}
    ]):
        dow = int(d["_id"])
        if 0 <= dow <= 6:
            volume[dow] = d["n"]

    return jsonify({
        "totalOrders": total,
        "onTime": on_time,
        "delayed": delayed,
        "severelyDelayed": severe,
        "cancelled": cancelled,
        "onTimeRate": round(100.0 * on_time / total, 1),
        "highRiskCount": severe,
        "avgTransitHours": avg_transit,
        "statusCounts": [on_time, delayed, severe, cancelled],
        "volumeByDow": volume,
        "model": MODEL_STATE["metrics"],
    })


@app.route("/api/orders")
def list_orders():
    db = get_db()
    limit = min(int(request.args.get("limit", 80)), 200)
    status = request.args.get("status")
    query = {}
    if status and status != "All":
        query["delivery_status"] = status

    drivers = {d["driver_id"]: d.get("name", d["driver_id"]) for d in db["drivers"].find({})}
    cursor = db["orders"].find(query, {"_id": 0, "checkpoint_logs": 0}).sort("order_id", 1).limit(limit)

    rows = []
    for o in cursor:
        status_val = o.get("delivery_status", "")
        traffic = float(o.get("traffic_delay_minutes") or 0)
        driver_id = o.get("driver_id", "")
        rows.append({
            "id": o.get("order_id"),
            "driver": pretty_driver(driver_id, drivers.get(driver_id)),
            "driver_id": driver_id,
            "dist": round(float(o.get("distance_km") or 0), 2),
            "trafficMin": round(traffic, 1),
            "weatherMin": round(float(o.get("weather_delay_minutes") or 0), 1),
            "region": o.get("customer_region", ""),
            "category": o.get("product_category", ""),
            "shipping": o.get("shipping_method", ""),
            "weight": round(float(o.get("package_weight_kg") or 0), 2),
            "delayHours": round(float(o.get("delivery_delay_hours") or 0), 2),
            "status": status_val,
            "risk": risk_from_status(status_val, traffic),
        })
    return jsonify({"orders": rows, "count": len(rows)})


@app.route("/api/aggregations")
def aggregations():
    db = get_db()
    orders = db["orders"]

    pipeline_a = list(orders.aggregate([
        {"$group": {
            "_id": "$delivery_status",
            "order_count": {"$sum": 1},
            "avg_distance_km": {"$avg": "$distance_km"},
            "avg_delay_hours": {"$avg": "$delivery_delay_hours"},
            "avg_satisfaction": {"$avg": "$customer_satisfaction"},
            "total_logistics_cost": {"$sum": "$logistics_cost"},
        }},
        {"$sort": {"avg_delay_hours": -1}},
    ]))

    status_order = ["On Time", "Delayed", "Severely Delayed", "Cancelled"]
    by_status = {d["_id"]: d for d in pipeline_a}
    avg_distance = [round(float(by_status.get(s, {}).get("avg_distance_km") or 0), 2) for s in status_order]
    avg_delay = [round(float(by_status.get(s, {}).get("avg_delay_hours") or 0), 2) for s in status_order]

    shipping = list(orders.aggregate([
        {"$group": {
            "_id": "$shipping_method",
            "total_orders": {"$sum": 1},
            "on_time_count": {"$sum": {"$cond": [{"$eq": ["$is_delayed", 0]}, 1, 0]}},
        }},
    ]))
    ship_order = ["Economy", "Standard", "Express", "Same Day"]
    ship_map = {d["_id"]: d for d in shipping}
    on_time_rates = []
    for method in ship_order:
        d = ship_map.get(method, {"total_orders": 1, "on_time_count": 0})
        total = d["total_orders"] or 1
        on_time_rates.append(round(100.0 * d["on_time_count"] / total, 1))

    drivers_lookup = {d["driver_id"]: d.get("name", d["driver_id"]) for d in db["drivers"].find({})}
    driver_perf = list(orders.aggregate([
        {"$group": {
            "_id": "$driver_id",
            "total_orders": {"$sum": 1},
            "delayed_orders": {"$sum": "$is_delayed"},
        }},
        {"$project": {
            "driver_id": "$_id",
            "total_orders": 1,
            "delayed_orders": 1,
            "delay_rate": {"$divide": ["$delayed_orders", "$total_orders"]},
            "_id": 0,
        }},
        {"$sort": {"delay_rate": -1}},
    ]))
    for d in driver_perf:
        d["name"] = pretty_driver(d["driver_id"], drivers_lookup.get(d["driver_id"]))
        d["delay_rate"] = round(float(d["delay_rate"]), 3)

    return jsonify({
        "pipelineA": [
            {
                "status": d["_id"],
                "order_count": d["order_count"],
                "avg_distance_km": round(float(d["avg_distance_km"]), 2),
                "avg_delay_hours": round(float(d["avg_delay_hours"]), 2),
                "avg_satisfaction": round(float(d["avg_satisfaction"]), 2),
                "total_logistics_cost": round(float(d["total_logistics_cost"]), 2),
            }
            for d in pipeline_a
        ],
        "avgDistance": avg_distance,
        "avgDelay": avg_delay,
        "shippingMethods": ship_order,
        "onTimeRates": on_time_rates,
        "drivers": driver_perf,
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    if not MODEL_STATE["ready"]:
        return jsonify({"error": MODEL_STATE["error"] or "Model not ready"}), 503

    body = request.get_json(force=True) or {}
    shipping = body.get("shipping_method", "Standard")
    encoder = MODEL_STATE["encoder"]
    if shipping not in set(encoder.classes_):
        shipping = encoder.classes_[0]
    shipping_enc = int(encoder.transform([shipping])[0])

    row = {
        "distance_km": float(body.get("distance_km", 150)),
        "traffic_delay_minutes": float(body.get("traffic_delay_minutes", 20)),
        "weather_delay_minutes": float(body.get("weather_delay_minutes", 3)),
        "package_weight_kg": float(body.get("package_weight_kg", 1.5)),
        "warehouse_processing_h": float(body.get("warehouse_processing_h", 4)),
        "courier_rating": float(body.get("courier_rating", 4.2)),
        "courier_experience_yrs": float(body.get("courier_experience_yrs", 4)),
        "warehouse_capacity_pct": float(body.get("warehouse_capacity_pct", 75)),
        "weekend_flag": int(body.get("weekend_flag", 0)),
        "holiday_flag": int(body.get("holiday_flag", 0)),
        "promotion_flag": int(body.get("promotion_flag", 0)),
        "order_hour": int(body.get("order_hour", 12)),
        "day_of_week": int(body.get("day_of_week", 3)),
        "month": int(body.get("month", 6)),
        "shipping_method_enc": shipping_enc,
    }
    X = pd.DataFrame([row])[FEATURE_COLS]
    model = MODEL_STATE["model"]
    delayed = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    classes = list(model.classes_)
    delay_prob = float(proba[classes.index(1)]) if 1 in classes else float(delayed)
    importances = sorted(
        zip(FEATURE_COLS, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )[:6]
    return jsonify({
        "delayed": bool(delayed),
        "delay_probability": round(delay_prob * 100, 1),
        "on_time_probability": round((1 - delay_prob) * 100, 1),
        "importances": [{"name": n, "val": round(float(v) * 100, 1)} for n, v in importances],
    })


if __name__ == "__main__":
    print("Connecting to MongoDB and training model...")
    train_model()
    if MODEL_STATE["ready"]:
        print("Model ready:", MODEL_STATE["metrics"])
    else:
        print("Model not ready:", MODEL_STATE["error"])
    app.run(host="127.0.0.1", port=5000, debug=False)
