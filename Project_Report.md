# Smart Delivery Fleet Delay & Risk Predictor

## Project Report

---

## 1. Introduction

The rapid growth of e-commerce has placed immense pressure on logistics and last-mile delivery operations. Delivery delays directly impact customer satisfaction, brand reputation, and operational costs. Predicting whether a delivery will be delayed — and understanding the factors behind delays — is therefore a critical business need.

This project, **Smart Delivery Fleet Delay & Risk Predictor**, implements an end-to-end data pipeline that combines **MongoDB** (a NoSQL document database) with **Scikit-Learn** machine learning to build a logistics intelligence platform. The system ingests a realistic e-commerce logistics dataset into MongoDB, performs advanced analytical queries using aggregation pipelines, trains a Random Forest classifier to predict delivery delays, and presents all insights through a live web-based dashboard built with Flask.

The project demonstrates key MongoDB concepts — including schema validation via `$jsonSchema`, document embedding, collection referencing, indexing, and aggregation operators (`$group`, `$lookup`, `$facet`) — alongside practical machine learning workflows for classification.

---

## 2. Project Objective

The primary objectives of this project are:

1. **Data Ingestion & Storage**: Ingest a large-scale e-commerce logistics dataset (15,000 orders, 36 features) into MongoDB using best-practice schema design, including JSON Schema validation, document embedding, and collection referencing.

2. **Advanced Querying**: Perform server-side analytical queries using MongoDB aggregation pipelines to derive business insights — including delivery performance by status, driver delay rates, shipping method comparisons, and regional analysis.

3. **Machine Learning**: Train and evaluate a Random Forest classification model that predicts whether a delivery order will be delayed (`is_delayed`: 0 = on-time, 1 = delayed), using features extracted from MongoDB.

4. **Cross-Dataset Evaluation**: Validate the trained model against a secondary synthetic dataset (`delivery_data.csv`) with a different schema, demonstrating real-world model portability challenges and feature alignment techniques.

5. **Interactive Dashboard**: Serve a real-time web dashboard via Flask that visualises fleet statistics, aggregation results, and allows users to run live ML predictions through a form-based interface.

---

## 3. Dataset Description

### 3.1 Primary Dataset — `ecommerce_logistics_delivery.csv`

This is a synthetic dataset of **15,000 e-commerce delivery orders** with **36 features**, generated using controlled statistical distributions and business relationships. It serves as the primary training data.

| Column | Type | Description |
|--------|------|-------------|
| `Order_ID` | String | Unique order identifier (e.g., ORD-00001) |
| `Order_Date` | Date | Date the order was placed |
| `Product_Category` | String | Category of the product ordered |
| `Customer_Region` | String | Geographic region of the customer |
| `Warehouse_Region` | String | Region of the fulfilling warehouse |
| `Shipping_Method` | Categorical | Economy, Standard, Express, or Same Day |
| `Payment_Method` | String | Payment type used |
| `Delivery_Partner` | String | Name of the delivery driver/partner |
| `Order_Value` | Float | Monetary value of the order |
| `Order_Quantity` | Integer | Number of items in the order |
| `Package_Weight_KG` | Float | Weight of the package in kilograms |
| `Package_Volume_CM3` | Float | Volume of the package in cubic centimetres |
| `Distance_KM` | Float | Delivery distance in kilometres |
| `Warehouse_Processing_Hours` | Float | Hours spent processing at the warehouse |
| `Traffic_Delay_Minutes` | Float | Minutes of traffic-related delay |
| `Weather_Delay_Minutes` | Float | Minutes of weather-related delay |
| `Courier_Rating` | Float | Rating of the courier (0–5 scale) |
| `Courier_Experience_Years` | Float | Years of experience of the courier |
| `Warehouse_Capacity_Utilization` | Float | Percentage of warehouse capacity in use |
| `Inventory_Availability` | Float | Inventory availability metric |
| `Weekend_Flag` | Binary | 1 if the order was placed on a weekend |
| `Holiday_Flag` | Binary | 1 if the order was placed on a holiday |
| `Promotion_Flag` | Binary | 1 if a promotion was applied |
| `Estimated_Delivery_Hours` | Float | Estimated delivery time in hours |
| `Promised_Delivery_Hours` | Integer | Promised delivery window |
| `Delivery_Delay_Hours` | Float | Actual delay in hours |
| `Delivery_Status` | Categorical | On Time, Delayed, Severely Delayed, Cancelled |
| `On_Time_Delivery_Flag` | Binary | 1 = delivered on time, 0 = delayed |
| `Cancellation_Flag` | Binary | 1 if the order was cancelled |
| `Return_Flag` | Binary | 1 if the order was returned |
| `Shipping_Cost` | Float | Cost of shipping |
| `Logistics_Cost` | Float | Total logistics cost |
| `Customer_Satisfaction_Score` | Float | Customer satisfaction rating |
| `Order_Hour` | Integer | Hour of the day the order was placed |
| `Day_of_Week` | Integer | Day of the week (0 = Monday, 6 = Sunday) |
| `Month` | Integer | Month of the year |

### 3.2 Secondary Dataset — `delivery_data.csv`

A smaller synthetic dataset of **500 orders** with **8 columns**, generated by `generate_data.py`. Used for cross-dataset model evaluation.

| Column | Description |
|--------|-------------|
| `order_id` | Unique order identifier |
| `driver_id` | Driver identifier |
| `distance_km` | Route distance in kilometres |
| `traffic_index` | Traffic congestion multiplier (0.5–2.5) |
| `weather_condition` | Categorical: Sunny, Rainy, Snow, Fog, Windy |
| `package_weight_kg` | Package weight in kg |
| `delivery_status` | Delivered, In-Transit, or Delayed |
| `is_delayed` | Target label (0 = on-time, 1 = delayed) |

### 3.3 Data Provenance

Both datasets are **fully synthetic** — no real customer, company, address, or personally identifiable information is included. The primary dataset's delivery performance is influenced by shipping method, distance, warehouse processing, traffic, weather, capacity, inventory, courier experience, and service quality.

---

## 4. MongoDB / PyMongo Implementation

### 4.1 Database Architecture

The project uses a MongoDB database named **`delivery_fleet`** with two collections:

```
delivery_fleet (Database)
├── drivers    — Reference collection (unique driver profiles)
└── orders     — Main collection (delivery orders with embedded checkpoint logs)
```

**Design patterns used:**

- **Referencing**: `orders.driver_id` references `drivers.driver_id` — linked via `$lookup` aggregation.
- **Embedding**: Each order document contains an embedded array `checkpoint_logs` with sub-documents (timestamp, location, status).

### 4.2 JSON Schema Validation (`$jsonSchema`)

The `orders` collection is protected by a `$jsonSchema` validator that enforces:

```javascript
{
  "$jsonSchema": {
    "bsonType": "object",
    "required": [
      "order_id", "driver_id", "distance_km", "traffic_delay_minutes",
      "weather_delay_minutes", "package_weight_kg", "shipping_method",
      "delivery_status", "is_delayed", "checkpoint_logs"
    ],
    "properties": {
      "package_weight_kg":     { "bsonType": "double", "minimum": 0 },
      "delivery_status":       { "enum": ["On Time", "Delayed", "Severely Delayed", "Cancelled"] },
      "is_delayed":            { "bsonType": "int", "minimum": 0, "maximum": 1 },
      "courier_rating":        { "bsonType": "double", "minimum": 0, "maximum": 5 },
      "checkpoint_logs": {
        "bsonType": "array",
        "items": {
          "bsonType": "object",
          "required": ["timestamp", "location", "status"],
          "properties": {
            "timestamp": { "bsonType": "date" },
            "location":  { "bsonType": "string" },
            "status":    { "bsonType": "string" }
          }
        }
      }
    }
  }
}
```

This ensures data integrity at the database level — invalid documents are rejected on insertion.

### 4.3 Indexes

| Collection | Field | Index Type | Purpose |
|------------|-------|------------|---------|
| `orders` | `order_id` | Unique | Prevent duplicate order insertion |
| `orders` | `driver_id` | Standard | Fast lookups for `$lookup` aggregation |
| `orders` | `customer_region` | Standard | Fast regional queries |
| `orders` | `delivery_status` | Standard | Fast status-based filtering |
| `drivers` | `driver_id` | Unique | Reference integrity and fast joins |

### 4.4 CSV Ingestion (`database.py`)

The ingestion process:

1. **Connects** to a local MongoDB instance using `MongoClient`.
2. **Drops** existing collections for a clean run.
3. **Creates** `drivers` and `orders` collections with the JSON Schema validator.
4. **Reads** the CSV file row by row using Python's `csv.DictReader`.
5. **Upserts** driver documents — one per unique `Delivery_Partner` — into the `drivers` collection (referencing pattern).
6. **Inserts** order documents into `orders`, each containing an embedded `checkpoint_logs` array of 3 checkpoint sub-documents (embedding pattern).
7. By default, ingests **5,000 rows** (configurable up to all 15,000).

### 4.5 Aggregation Pipelines (`queries.py`)

Three aggregation pipelines are implemented:

**Pipeline A — `$group` by Delivery Status:**
Groups all orders by `delivery_status` and computes: order count, average distance, average delay hours, average customer satisfaction, and total logistics cost.

**Pipeline B — `$lookup` Join:**
Performs a left join between `orders` and `drivers` using `$lookup` on `driver_id`, followed by `$unwind` and `$project` to enrich order records with the driver's name and region.

**Pipeline C — `$facet` Multi-Analysis:**
Runs three sub-analyses in a single aggregation call:
- **Driver Performance**: Delay rate per delivery partner (top 5 by delay rate).
- **Region Analysis**: Average delay hours and customer satisfaction per customer region.
- **Shipping Analysis**: Average delay, shipping cost, and on-time rate per shipping method.

---

## 5. Data Preprocessing

### 5.1 Preprocessing for MongoDB Training (`app.py` and `ml_model.py`)

When training from MongoDB-ingested data:

- **Feature Selection**: 14 numeric features are selected: `distance_km`, `traffic_delay_minutes`, `weather_delay_minutes`, `package_weight_kg`, `warehouse_processing_h`, `courier_rating`, `courier_experience_yrs`, `warehouse_capacity_pct`, `weekend_flag`, `holiday_flag`, `promotion_flag`, `order_hour`, `day_of_week`, `month`.
- **Label Encoding**: The `shipping_method` column (categorical: Economy, Standard, Express, Same Day) is encoded to integers using `sklearn.preprocessing.LabelEncoder`, resulting in a 15th feature `shipping_method_enc`.
- **Target Variable**: `is_delayed` (0 = on-time, 1 = delayed), derived from `On_Time_Delivery_Flag` in the original CSV.
- **Train-Test Split**: 80/20 stratified split using `train_test_split` with `random_state=42`.

### 5.2 Preprocessing for Cross-Dataset Evaluation (`ml_model.py`)

Since `ecommerce_logistics_delivery.csv` and `delivery_data.csv` have different schemas, a mapping step aligns them onto 4 shared features:

| Shared Feature | Training Source (ecommerce) | Test Source (delivery_data) | Mapping |
|---|---|---|---|
| `distance_km` | `Distance_KM` | `distance_km` | Direct rename |
| `package_weight_kg` | `Package_Weight_KG` | `package_weight_kg` | Direct rename |
| `traffic_delay_minutes` | `Traffic_Delay_Minutes` | `traffic_index × 15` | Index → minutes conversion |
| `weather_delay_minutes` | `Weather_Delay_Minutes` | Category → minutes lookup | Sunny=0, Windy=2, Fog=4, Rainy=6, Snow=10 |

### 5.3 Preprocessing for In-Domain Evaluation (`ml_model.py`)

For the full ecommerce in-domain run:

- **Dropped columns**: `Order_ID`, `Order_Date`, `Delivery_Status`, `Delivery_Delay_Hours`, `Cancellation_Flag`, `Return_Flag`, `On_Time_Delivery_Flag` (to prevent data leakage).
- **Categorical encoding**: All remaining object-type columns are label-encoded.
- **Target**: `is_delayed` = 1 when `On_Time_Delivery_Flag == 0`.

---

## 6. Model Implementation

### 6.1 Algorithm Selection

The project uses **Random Forest Classifier** (`sklearn.ensemble.RandomForestClassifier`) as the primary model, configured with:

- `n_estimators = 200` (200 decision trees)
- `random_state = 42` (reproducibility)
- `n_jobs = -1` (parallel training across all CPU cores)

Random Forest was chosen because:
- It handles both numerical and encoded categorical features well.
- It is robust against overfitting due to ensemble averaging.
- It provides built-in feature importance scores.
- It does not require feature scaling.

### 6.2 Three Training Runs

The `ml_model.py` script performs three distinct training and evaluation runs:

| Run | Training Data | Test Data | Features |
|-----|---------------|-----------|----------|
| **MongoDB Run** | 5,000 orders from MongoDB `orders` collection | 20% hold-out from same data | 15 features (full ecommerce + shipping encoding) |
| **Cross-Dataset Run** | All 15,000 rows from `ecommerce_logistics_delivery.csv` | All 500 rows from `delivery_data.csv` | 4 shared features (mapped) |
| **In-Domain Run** | 80% of `ecommerce_logistics_delivery.csv` | 20% hold-out (stratified) | ~29 features (all non-leaky columns, label-encoded) |

### 6.3 Flask API Integration (`app.py`)

The Flask backend trains the model at startup:
- Loads all order documents from MongoDB.
- Encodes `shipping_method`, selects 15 features, splits 80/20, trains Random Forest.
- Stores the trained model, encoder, and metrics in a global `MODEL_STATE` dictionary.
- Exposes a `POST /api/predict` endpoint that accepts delivery parameters as JSON, runs inference, and returns: predicted class (delayed/on-time), delay probability, on-time probability, and top 6 feature importances.

---

## 7. Evaluation Results

### 7.1 Metrics Used

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Accuracy** | (TP + TN) / Total | Overall correct predictions |
| **Precision** | TP / (TP + FP) | Of predicted delays, how many were actual delays |
| **Recall** | TP / (TP + FN) | Of actual delays, how many were correctly predicted |
| **F1-Score** | 2 × (Precision × Recall) / (Precision + Recall) | Harmonic mean balancing precision and recall |

### 7.2 Expected Results

**MongoDB Run (15 features, 5,000 orders, 80/20 split):**
- Uses the richest feature set including courier rating, warehouse processing, traffic/weather delays, temporal features, and shipping method.
- Expected to achieve **high accuracy** due to the strong correlation between features like `traffic_delay_minutes`, `warehouse_processing_h`, and the delay target.

**In-Domain Run (full ecommerce, ~29 features, 80/20 split):**
- Uses the maximum number of features available from the ecommerce dataset.
- Expected to achieve the **highest overall performance** as it has the most informative feature set with no domain shift.

**Cross-Dataset Run (4 shared features, train on ecommerce, test on delivery_data.csv):**
- Limited to only 4 overlapping features with approximate mappings.
- Expected to show **lower performance** due to:
  - Reduced feature space (4 vs 15+ features).
  - Approximate unit conversions (`traffic_index × 15 → minutes`).
  - Different data distributions between synthetic generators.
- This run demonstrates a realistic model portability challenge.

### 7.3 Evaluation Outputs

Each run produces:
- **Classification Report**: Per-class precision, recall, F1-score for both "On-Time" and "Delayed" classes.
- **Confusion Matrix**: 2×2 matrix showing TP, TN, FP, FN counts.
- **Feature Importances**: Ranked list showing each feature's contribution to the model's decisions (Random Forest `feature_importances_` attribute).

### 7.4 Dashboard Model Metrics

The web dashboard displays the MongoDB-run model's metrics live:
- Accuracy, Precision, Recall, and F1-Score shown as percentage values in the ML Predictions view.
- Feature importance bars rendered for each prediction output.
- Training sample count and test sample count displayed.

---

## 8. Conclusion

This project successfully demonstrates an end-to-end data science pipeline integrating MongoDB with machine learning for logistics delay prediction:

1. **MongoDB Mastery**: The project showcases key MongoDB features — `$jsonSchema` validation for data integrity, document embedding (`checkpoint_logs`), collection referencing (`drivers` ↔ `orders`), indexing for performance, and powerful aggregation pipelines (`$group`, `$lookup`, `$facet`) for server-side analytics.

2. **PyMongo Proficiency**: The Python-MongoDB integration via PyMongo covers connection management, collection setup, CSV ingestion with upserts, and programmatic aggregation — all following best practices.

3. **ML Pipeline**: A complete machine learning workflow is implemented — from feature selection and encoding through stratified train-test splitting to model training, evaluation, and live inference via a REST API.

4. **Cross-Dataset Generalization**: By evaluating the model across two datasets with different schemas, the project highlights real-world challenges of model portability, feature alignment, and domain adaptation.

5. **Full-Stack Delivery**: The Flask-powered dashboard with Chart.js visualizations ties everything together into a professional, interactive application — making the analytics accessible beyond the command line.

**Potential future enhancements** include:
- Adding more ML models (e.g., Logistic Regression, XGBoost) for comparison.
- One-hot encoding categorical features like `weather_condition` for richer feature representation.
- Deploying the application to a cloud platform for production use.
- Implementing automated data refresh pipelines.

---

*This project is intended for educational and academic purposes, demonstrating the integration of MongoDB database operations with machine learning for logistics analytics.*
