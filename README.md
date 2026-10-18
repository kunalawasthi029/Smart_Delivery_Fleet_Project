# Smart Delivery Fleet Delay & Risk Predictor

## Introduction
This repository implements an end‑to‑end data pipeline that showcases **MongoDB** capabilities (schema validation, embedding, referencing, and aggregation) together with a **Scikit‑Learn** machine‑learning model for predicting delivery delays.

## Project Objective
- Generate realistic synthetic delivery order data.
- Store the data in MongoDB using best‑practice schema design.
- Perform advanced analytical queries with aggregation pipelines.
- Train and evaluate a classification model that predicts whether an order will be delayed.

## Dataset Description
The synthetic CSV (`delivery_data.csv`) contains the following columns:
| Column | Description |
|--------|-------------|
| `order_id` | Unique order identifier (e.g., `ORD-123456`). |
| `driver_id` | Identifier that references a document in the `drivers` collection. |
| `distance_km` | Route distance in kilometres. |
| `traffic_index` | Multiplier representing traffic congestion (higher = more traffic). |
| `weather_condition` | Categorical weather description (`Sunny`, `Rainy`, `Snow`, `Fog`, `Windy`). |
| `package_weight_kg` | Weight of the package. |
| `delivery_status` | Enum: `Delivered`, `In‑Transit`, `Delayed`. |
| `is_delayed` | Target label (`0` = on‑time, `1` = delayed). |
| `checkpoint_logs` | Embedded array of checkpoint sub‑documents (timestamp, location, status). |

## MongoDB Schema Design & Validation Rules
- **Collections**: `drivers` (simple reference collection) and `orders` (main dataset).
- **JSON Schema Validation** on `orders` ensures:
  - `package_weight_kg` is a non‑negative double.
  - `delivery_status` matches the allowed enum values.
  - `is_delayed` is an integer in `{0,1}`.
  - `checkpoint_logs` is an array of sub‑documents each containing a `timestamp`, `location`, and `status`.
- **Referencing**: `orders.driver_id` → `drivers.driver_id`.
- **Embedding**: `checkpoint_logs` are stored directly inside each order document.

## PyMongo CRUD Operations (`database.py`)
1. **Connection** – `MongoClient` to a local MongoDB instance.
2. **Collection setup** – creates `drivers` and `orders` with the JSON schema validator.
3. **CSV ingestion** –
   - Upserts driver documents (one per unique `driver_id`).
   - Inserts order documents with embedded checkpoint logs.
4. **Indexes** – unique index on `order_id` and an index on `driver_id` for fast look‑ups.

## Aggregation Pipelines (`queries.py`)
1. **Metrics per delivery status** – `$group` to compute average distance, total weight, and order count.
2. **Driver enrichment** – `$lookup` + `$unwind` to join `orders` with the `drivers` collection.
3. **Faceted analysis** – `$facet` to simultaneously compute:
   - Top drivers by delay rate.
   - Weather conditions that cause the most delays.

## Machine Learning Implementation (`ml_model.py`)
- Loads numeric features (`distance_km`, `traffic_index`, `package_weight_kg`) and the target `is_delayed` directly from MongoDB.
- Splits data (80/20) with stratification.
- Supports two classifiers:
  - **Random Forest** (default, 200 trees).
  - **Logistic Regression** (optional via `model_type` argument).
- Prints standard classification metrics: Accuracy, Precision, Recall, and F1‑Score.

## Installation
```bash
# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate

# Install required packages
pip install pymongo pandas scikit-learn
```
> **Note:** The scripts use only standard library modules besides the three dependencies above.

## Usage
```bash
# 1. Generate synthetic data
python generate_data.py

# 2. Ingest data into MongoDB (ensure MongoDB is running locally)
python database.py

# 3. Run aggregation examples
python queries.py

# 4. Train & evaluate the ML model
python ml_model.py   # use "--model_type logistic_regression" to switch models
```
All scripts print clear console messages indicating success.

## Project Structure
```
mongo bd project/
├── generate_data.py      # synthetic CSV generator
├── database.py           # MongoDB connection & ingestion
├── queries.py            # aggregation pipeline examples
├── ml_model.py           # ML training & evaluation
├── delivery_data.csv     # generated dataset (created by generate_data.py)
└── README.md             # **this** document
```

## Extending the Project
- Add more features (e.g., one‑hot encode `weather_condition`).
- Replace the synthetic data generator with real order data.
- Deploy a lightweight API (FastAPI/Flask) that serves real‑time delay predictions.
- Automate nightly data refresh using a cron job or the `/schedule` slash command.

---
*This repository is intended for educational purposes and demonstrates how MongoDB and machine learning can be combined for logistics analytics.*
