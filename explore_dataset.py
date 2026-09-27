import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('IMDB Dataset.csv')

print("="*60)
print("IMDB DATASET EXPLORATION")
print("="*60)

print(f"\nDataset Shape: {df.shape}")
print(f"Total Reviews: {len(df)}")

print("\n" + "="*60)
print("COLUMN INFORMATION")
print("="*60)
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nData Types:\n{df.dtypes}")

print("\n" + "="*60)
print("FIRST 3 ROWS")
print("="*60)
print(df.head(3))

print("\n" + "="*60)
print("SENTIMENT DISTRIBUTION")
print("="*60)
print(df['sentiment'].value_counts())

print("\n" + "="*60)
print("MISSING VALUES")
print("="*60)
print(df.isnull().sum())

print("\n" + "="*60)
print("TEXT STATISTICS")
print("="*60)
print(f"Average review length: {df['review'].str.len().mean():.2f} characters")
print(f"Min review length: {df['review'].str.len().min()} characters")
print(f"Max review length: {df['review'].str.len().max()} characters")
