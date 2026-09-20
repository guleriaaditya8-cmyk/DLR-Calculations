import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Setup aesthetics (Unified)
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# 2. File path
path = r"D:\ECE M2\Master Thesis\VS Code Weather Data analysis\paris_weather_2025_2026.csv"

# 3. Load dataset 
try:
    df = pd.read_csv(path)
    print("File loaded successfully.\n")
except FileNotFoundError:
    print("Error: File not found. Check your path logic.")

# 4. Preview & Structure
print("--- First 5 Rows ---")
print(df.head()) 

print(f"\nShape (rows, columns): {df.shape}")

print("\n--- Data types and non-null counts ---")
df.info() # Directly prints to console

# 5. Descriptive stats
print("\n--- Descriptive statistics ---")
print(df.describe())

# 6. Data Quality Check
print("\n--- Missing Values Count ---")
print(df.isnull().sum())

print("\n--- Duplicate Rows ---")
print(f"Total duplicates: {df.duplicated().sum()}")


#Here I load the dataset from Solar_Energy_Production.csv into a DataFrame called df.
#I then display the first 5 rows with head() to get a first look at the structure (columns, types of values, etc.).
# we upload the CSV (left sidebar → Files → upload) or mount Drive and adapt file_path.

#  Check missing values Here I load the dataset from Solar_Energy_Production.csv into a DataFrame called df.
# I then display the first 5 rows with head() to get a first look at the structure (columns, types of values, etc.).

