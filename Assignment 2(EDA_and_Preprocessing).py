#EDA and Preprocessing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

# Load Data
file_path = "employee_data.csv" 
df = pd.read_csv(file_path)

# Data Exploration
# Display basic info
print("Shape of dataset:", df.shape)
print("\nColumn Names:", df.columns.tolist())

# Renaming columns for consistency
df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]

# Unique values and count
print("\nUnique values per column:")
for col in df.columns:
    unique_vals = df[col].unique()
    print(f"{col}: {unique_vals} (Count: {len(unique_vals)})")

# Statistical summary
print(df.describe(include='all'))

#Data Cleaning
# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Replace age = 0 with NaN
df['age'] = df['age'].replace(0, pd.NA)

# Treat missing values
for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:
        df[col].fillna(df[col].median(), inplace=True)
    else:
        df[col].fillna(df[col].mode()[0], inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Detect outliers using IQR
def find_outliers_IQR(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    outliers = data[(data[column] < Q1 - 1.5 * IQR) | (data[column] > Q3 + 1.5 * IQR)]
    return outliers

# Print outliers for numerical columns
print("\nOutliers detected:")
for col in df.select_dtypes(include=['int64', 'float64']).columns:
    outliers = find_outliers_IQR(df, col)
    print(f"{col}: {len(outliers)} outliers")


#Data Analysis
# Filter data: age > 40 and salary < 5000
filtered_df = df[(df['age'] > 40) & (df['salary'] < 5000)]
print("Filtered data:")
print(filtered_df)

# Plot Age vs Salary
plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x='age', y='salary')
plt.title("Age vs Salary")
plt.xlabel("Age")
plt.ylabel("Salary")
plt.grid(True)
plt.show()

# Count of people from each place
plt.figure(figsize=(8,5))
sns.countplot(y='place', data=df, order=df['place'].value_counts().index)
plt.title("Count of people from each place")
plt.show()

#Data Encoding
# Check for categorical columns
cat_cols = df.select_dtypes(include='object').columns
print("Categorical columns:", cat_cols)

# Apply Label Encoding / One-Hot Encoding
df_encoded = df.copy()
for col in cat_cols:
    if df_encoded[col].nunique() <= 2:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
    else:
        df_encoded = pd.get_dummies(df_encoded, columns=[col], drop_first=True)
print("\nEncoded Data Sample:")
print(df_encoded.head())

#Feature Scaling
# Select numerical columns
num_cols = df_encoded.select_dtypes(include=['int64', 'float64']).columns

# Standard Scaler
scaler_std = StandardScaler()
df_std_scaled = df_encoded.copy()
df_std_scaled[num_cols] = scaler_std.fit_transform(df_std_scaled[num_cols])

# MinMax Scaler
scaler_mm = MinMaxScaler()
df_mm_scaled = df_encoded.copy()
df_mm_scaled[num_cols] = scaler_mm.fit_transform(df_mm_scaled[num_cols])

print("Standard Scaled Sample:")
print(df_std_scaled.head())

print("MinMax Scaled Sample:")
print(df_mm_scaled.head())


