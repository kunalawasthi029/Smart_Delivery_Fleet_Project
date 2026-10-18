#!/usr/bin/env bash
# build.sh — Render build script
set -o errexit

pip install -r requirements.txt

# Seed the database on first deploy (if orders collection is empty)
python -c "
import os
from pymongo import MongoClient

uri = os.environ.get('MONGO_URI', 'mongodb://localhost:27017')
client = MongoClient(uri, serverSelectionTimeoutMS=10000)
db = client['delivery_fleet']

if db['orders'].count_documents({}) == 0:
    print('Database is empty — seeding data...')
    import subprocess, sys
    subprocess.run([sys.executable, 'database.py'], check=True)
else:
    print(f'Database already has {db[\"orders\"].count_documents({})} orders — skipping seed.')
"
