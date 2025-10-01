import os
import pandas as pd
from pymongo import MongoClient

# MongoDB Connection
client = MongoClient("mongodb://localhost:27017")  # Change if needed
db = client["cos"]  # Your database name

# Folder containing CSV files
folder_path = "data"

# Loop through all CSV files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        collection_name = filename.replace(".csv", "")  # Collection name = filename
        file_path = os.path.join(folder_path, filename)

        # Read CSV file
        df = pd.read_csv(file_path)
        
        # Convert DataFrame to dictionary format
        data = df.to_dict(orient="records")

        # Insert into MongoDB
        db[collection_name].insert_many(data)

        print(f"✅ Imported {filename} into collection: {collection_name}")

print("🎉 All CSV files imported successfully!")
