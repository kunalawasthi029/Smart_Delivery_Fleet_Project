# 🚚 Smart Delivery Fleet — Delay & Risk Predictor

[![Live Demo](https://img.shields.io/badge/🌐_Live_Demo-Render-brightgreen?style=for-the-badge)](https://smart-delivery-fleet-project.onrender.com/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-47A248?style=for-the-badge&logo=mongodb&logoColor=white)](https://www.mongodb.com/atlas)
[![Flask](https://img.shields.io/badge/Flask-API-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

> **An end-to-end data pipeline and ML-powered web dashboard** for predicting delivery delays and assessing risk in e-commerce logistics. Built with Flask, MongoDB, and Scikit-Learn.

### 🔗 [Live Demo → https://smart-delivery-fleet-project.onrender.com/](https://smart-delivery-fleet-project.onrender.com/)

---

## 📌 Project Objective

- Ingest and store **5,000+ real e-commerce logistics records** in MongoDB using best-practice schema design.
- Perform **advanced analytical queries** with MongoDB aggregation pipelines.
- Train and evaluate a **Random Forest classifier** that predicts whether an order will be delayed.
- Serve a **live interactive dashboard** with real-time statistics, order tracking, and delay risk prediction.

---

## 🖥️ Dashboard Features

| Feature | Description |
|---|---|
| **Fleet Overview** | Total orders, on-time rate, high-risk count, avg transit hours |
| **Order Tracker** | Searchable & filterable order table with status, risk level, distance, delays |
| **Delivery Analytics** | Charts for status distribution, volume by day, avg distance/delay by status |
| **Shipping Analysis** | On-time delivery rates by shipping method (Economy, Standard, Express, Same Day) |
| **Driver Performance** | Driver-level delay rates and total order counts |
| **ML Risk Predictor** | Interactive form to predict delay probability for custom delivery scenarios |
| **Model Metrics** | Live accuracy, precision, recall, and F1-score of the trained model |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python, Flask, Gunicorn |
| **Database** | MongoDB Atlas (cloud) with `$jsonSchema` validation |
| **ML Model** | Scikit-Learn — Random Forest Classifier (200 trees) |
| **Frontend** | HTML5, CSS3, JavaScript (single-page dashboard) |
| **Hosting** | Render (Web Service — Free Tier) |
| **Data** | 5,000 records from real e-commerce logistics dataset |

---

## 📊 Dataset Description

The dataset (`ecommerce_logistics_delivery.csv`) contains **15,000 records** with the following key columns:

| Column | Description |
|---|---|
| `order_id` | Unique order identifier |
| `driver_id` | References a document in the `drivers` collection |
| `distance_km` | Route distance in kilometres |
| `traffic_delay_minutes` | Traffic-induced delay |
| `weather_delay_minutes` | Weather-induced delay |
| `package_weight_kg` | Weight of the package |
| `shipping_method` | Enum: `Standard`, `Express`, `Economy`, `Same Day` |
| `delivery_status` | Enum: `On Time`, `Delayed`, `Severely Delayed`, `Cancelled` |
| `is_delayed` | Target label (`0` = on-time, `1` = delayed) |
| `checkpoint_logs` | Embedded array of checkpoint sub-documents |
| `customer_satisfaction` | Customer satisfaction score (1–5) |
| `logistics_cost` | Total logistics cost per order |

---

## 🗄️ MongoDB Schema Design & Validation

- **Collections**: `drivers` (reference) and `orders` (main dataset)
- **JSON Schema Validation** on `orders` ensures:
  - `package_weight_kg` is a non-negative double
  - `delivery_status` matches the allowed enum values
  - `is_delayed` is an integer in `{0, 1}`
  - `checkpoint_logs` is an array of sub-documents with `timestamp`, `location`, and `status`
- **Referencing**: `orders.driver_id` → `drivers.driver_id`
- **Embedding**: `checkpoint_logs` are stored inside each order document
- **Indexes**: Unique on `order_id`, plus indexes on `driver_id`, `customer_region`, `delivery_status`

---

## 🔧 PyMongo CRUD Operations (`database.py`)

1. **Connection** — `MongoClient` with `MONGO_URI` env variable (supports local & Atlas)
2. **Collection Setup** — Creates `drivers` and `orders` with JSON Schema validator
3. **CSV Ingestion** — Upserts driver documents; inserts orders with embedded checkpoint logs
4. **Indexes** — Unique on `order_id`, indexed on `driver_id`, `customer_region`, `delivery_status`

---

## 📈 Aggregation Pipelines (`queries.py`)

1. **Metrics per delivery status** — `$group` to compute avg distance, total cost, and order count
2. **Driver enrichment** — `$lookup` + `$unwind` to join `orders` with `drivers`
3. **Faceted analysis** — `$facet` for top drivers by delay rate & regional insights

---

## 🤖 Machine Learning (`ml_model.py`)

- **Features**: 15 numeric features including distance, traffic/weather delays, courier rating, experience, etc.
- **Model**: Random Forest Classifier (200 trees, stratified 80/20 split)
- **Metrics**: Accuracy, Precision, Recall, F1-Score (displayed live on the dashboard)
- **Prediction API**: `POST /api/predict` — returns delay probability & feature importances

---

## 🚀 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Dashboard UI |
| `GET` | `/api/health` | System health check (MongoDB, model status) |
| `GET` | `/api/stats` | Fleet-wide statistics |
| `GET` | `/api/orders` | Order list with filtering & pagination |
| `GET` | `/api/aggregations` | Aggregation pipeline results |
| `POST` | `/api/predict` | ML delay prediction for custom inputs |

---

## ⚡ Installation & Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/kunalawasthi029/Smart_Delivery_Fleet_Project.git
cd Smart_Delivery_Fleet_Project

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set MongoDB URI (optional — defaults to localhost)
export MONGO_URI="mongodb://localhost:27017"   # or your MongoDB Atlas URI

# 5. Ingest data into MongoDB
python database.py

# 6. Run the application
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

---

## 📁 Project Structure

```
Smart_Delivery_Fleet_Project/
├── app.py                          # Flask API + dashboard server
├── dashboard.html                  # Interactive frontend dashboard
├── database.py                     # MongoDB connection & CSV ingestion
├── queries.py                      # Aggregation pipeline examples
├── ml_model.py                     # ML model training & evaluation
├── generate_data.py                # Synthetic data generator
├── ecommerce_logistics_delivery.csv # Real e-commerce logistics dataset (15K rows)
├── delivery_data.csv               # Generated synthetic dataset
├── requirements.txt                # Python dependencies
├── build.sh                        # Render build script
├── Project_Report.md               # Detailed project report
├── .gitignore                      # Git ignore rules
└── README.md                       # This document
```

---

## 🌐 Deployment

This project is deployed on **Render** with **MongoDB Atlas**:

- **Web Service**: [Render — Free Tier](https://render.com)
- **Database**: [MongoDB Atlas — M0 Free Cluster](https://www.mongodb.com/atlas)
- **Auto-deploy**: Pushes to `main` branch trigger automatic redeployment

---

## 👤 Author

**Kunal Awasthi**
- GitHub: [@kunalawasthi029](https://github.com/kunalawasthi029)

---

## 📄 License

This project is for educational purposes.
