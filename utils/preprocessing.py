import os
import sys
import pandas as pd

# Set working directory
print("📁 Current Directory:", os.getcwd())

# Optional: Check files in data directory
data_dir = r"D:\InsightIQ\data"
print("📦 Files in data folder:", os.listdir(data_dir))

# Add parent dir to sys.path for importing custom modules (like utils)
sys.path.append(r"D:\InsightIQ")

# Load raw dataset
file_path = os.path.join(data_dir, "telco_customer_churn.csv")
df = pd.read_csv(file_path)
print("✅ Loaded Dataset Shape:", df.shape)

# Show initial data stats
print("🔍 Initial dtypes:\n", df.dtypes)
print("🔍 Missing Values:\n", df.isnull().sum())

# Clean 'TotalCharges'
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna(subset=['TotalCharges'])

# Drop unused columns
if 'customerID' in df.columns:
    df.drop('customerID', axis=1, inplace=True)

# Encode target label
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Save intermediate cleaned CSV
cleaned_path = os.path.join(data_dir.replace("InsightIQ", "InsightIQ"), "cleaned_telco.csv")
df.to_csv(cleaned_path, index=False)
print(f"📁 Cleaned data saved to: {cleaned_path}")

# Import custom encoder
from utils.preprocessing import encode_categoricals

# Encode all categorical features
df, label_encoders = encode_categoricals(df)
print("🔢 Encoded data preview:")
print(df.head())
