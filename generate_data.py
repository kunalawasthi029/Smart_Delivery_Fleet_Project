'''generate_data.py
Generate synthetic delivery order data and save to CSV.
'''

import csv
import random
import string
from datetime import datetime, timedelta

def random_order_id() -> str:
    """Generate a random order ID like 'ORD-XXXX'"""
    return "ORD-" + "".join(random.choices(string.digits, k=6))

def random_driver_id() -> str:
    """Generate a random driver ID like 'DRV-XX'"""
    return "DRV-" + "".join(random.choices(string.ascii_uppercase, k=2))

def random_weather() -> str:
    return random.choice(["Sunny", "Rainy", "Snow", "Fog", "Windy"])

def random_status() -> str:
    return random.choice(["Delivered", "In-Transit", "Delayed"])

def generate_row() -> dict:
    distance = round(random.uniform(5, 200), 2)  # km
    traffic = round(random.uniform(0.5, 2.5), 2)   # traffic index (higher = more traffic)
    weight = round(random.uniform(0.1, 50), 2)    # kg
    status = random_status()
    is_delayed = 1 if status == "Delayed" else 0
    return {
        "order_id": random_order_id(),
        "driver_id": random_driver_id(),
        "distance_km": distance,
        "traffic_index": traffic,
        "weather_condition": random_weather(),
        "package_weight_kg": weight,
        "delivery_status": status,
        "is_delayed": is_delayed,
    }

def generate_data(num_rows: int = 500, output_file: str = "delivery_data.csv"):
    """Create a CSV file with synthetic delivery order data.

    Args:
        num_rows: Number of synthetic orders to generate.
        output_file: File name (saved in the same folder as this script).
    """
    fieldnames = [
        "order_id",
        "driver_id",
        "distance_km",
        "traffic_index",
        "weather_condition",
        "package_weight_kg",
        "delivery_status",
        "is_delayed",
    ]
    with open(output_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for _ in range(num_rows):
            writer.writerow(generate_row())
    print(f"Generated {num_rows} rows -> {output_file}")

if __name__ == "__main__":
    generate_data()
