# Bangalore House Price Analysis

# Q1: Basic EDA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv("house_price.csv") 
print(df.head())
# Basic information
print(df.info())
print(df.describe())

# Checking nulls
print("\nMissing values:\n", df.isnull().sum())

# Add price per sqft column
df['price_per_sqft'] = df['price']*100000 / df['total_sqft']
df[['price', 'total_sqft', 'price_per_sqft']].head()

# Q2: Outlier Detection and Removal Methods
# a) Mean and Standard Deviation Method
df_mean_std = df.copy()
mean = df_mean_std['price_per_sqft'].mean()
std = df_mean_std['price_per_sqft'].std()

# Trim outliers
df_mean_std = df_mean_std[(df_mean_std['price_per_sqft'] > (mean - 3*std)) & 
                          (df_mean_std['price_per_sqft'] < (mean + 3*std))]

print("\nQ2a: Mean & STD")
print("Original shape:", df.shape)
print("After removing outliers:", df_mean_std.shape)
print(df_mean_std['price_per_sqft'].describe())

#b) Percentile method
df_percentile = df.copy()
low = df_percentile['price_per_sqft'].quantile(0.05)
high = df_percentile['price_per_sqft'].quantile(0.95)

# Trimming
df_percentile = df_percentile[(df_percentile['price_per_sqft'] >= low) & 
                              (df_percentile['price_per_sqft'] <= high)]
print("\nQ2b: Percentile Method")
print("Original shape:", df.shape)
print("After removing outliers:", df_percentile.shape)
print(df_percentile['price_per_sqft'].describe())

# c)IQR Method
df_iqr = df.copy()
Q1 = df_iqr['price_per_sqft'].quantile(0.25)
Q3 = df_iqr['price_per_sqft'].quantile(0.75)
IQR = Q3 - Q1

# Trimming
df_iqr = df_iqr[(df_iqr['price_per_sqft'] >= Q1 - 1.5*IQR) &
                (df_iqr['price_per_sqft'] <= Q3 + 1.5*IQR)]
print("\nQ2c: IQR Method")
print("Original shape:", df.shape)
print("After removing outliers:", df_iqr.shape)
print(df_iqr['price_per_sqft'].describe())

# d)Z-Score Method
df_zscore = df.copy()
z_scores = np.abs(stats.zscore(df_zscore['price_per_sqft']))
df_zscore = df_zscore[z_scores < 3]

print("\nQ2d: Z-Score Method")
print("Original shape:", df.shape)
print("After removing outliers:", df_zscore.shape)
print(df_zscore['price_per_sqft'].describe())

# Q3: Box Plot Comparison
plt.figure(figsize=(18, 6))

plt.subplot(1, 5, 1)
sns.boxplot(df['price_per_sqft'])
plt.title("Original")

plt.subplot(1, 5, 2)
sns.boxplot(df_mean_std['price_per_sqft'])
plt.title("Mean & STD")

plt.subplot(1, 5, 3)
sns.boxplot(df_percentile['price_per_sqft'])
plt.title("Percentile")

plt.subplot(1, 5, 4)
sns.boxplot(df_iqr['price_per_sqft'])
plt.title("IQR")

plt.subplot(1, 5, 5)
sns.boxplot(df_zscore['price_per_sqft'])
plt.title("Z-Score")

plt.tight_layout()
plt.show()

# Q4: Histogram + Normality + Transformations

# Before Transformation
plt.figure(figsize=(6,4))
sns.histplot(df['price_per_sqft'], kde=True)
plt.title('Original Price per Sqft Distribution')
plt.show()

print("Skewness (Before):", skew(df['price_per_sqft']))
print("Kurtosis (Before):", kurtosis(df['price_per_sqft']))

# After Log Transformation
df['log_price_per_sqft'] = np.log1p(df['price_per_sqft'])

plt.figure(figsize=(6,4))
sns.histplot(df['log_price_per_sqft'], kde=True)
plt.title('Log-Transformed Distribution')
plt.show()

print("Skewness (After):", skew(df['log_price_per_sqft']))
print("Kurtosis (After):", kurtosis(df['log_price_per_sqft']))

# Q5: Correlation Heatmap

plt.figure(figsize=(10,6))
correlation = df.corr(numeric_only=True)
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Q6: Scatter Plot to Check Correlation

sns.pairplot(df[['price', 'total_sqft', 'price_per_sqft']], diag_kind='kde')
plt.suptitle("Scatterplot Matrix", y=1.02)
plt.show()
