# Data Mining Task 1: Data Preprocessing and Exploration


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os

# Create directory structure for outputs
os.makedirs('visualizations', exist_ok=True)

# Step 1: Load the Iris Dataset
print("=" * 50)
print("STEP 1: LOADING IRIS DATASET")
print("=" * 50)

# Load iris dataset from scikit-learn
iris = load_iris()

# Convert to pandas DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target

# Map target numbers to species names for better readability
species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
df['species_name'] = df['species'].map(species_map)

print("Dataset loaded successfully!")
print(f"Dataset shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())

# Step 2: Preprocessing
print("\n" + "=" * 50)
print("STEP 2: PREPROCESSING")
print("=" * 50)

# Check for missing values
print("\nChecking for missing values:")
missing_values = df.isnull().sum()
print(missing_values)

if missing_values.sum() == 0:
    print("No missing values found in the dataset.")
else:
    print(f"Total missing values: {missing_values.sum()}")
    # Handle missing values if any (though Iris has none)
    df = df.dropna()

# Separate features and target
feature_columns = ['sepal length (cm)', 'sepal width (cm)', 
                   'petal length (cm)', 'petal width (cm)']
X = df[feature_columns]
y = df['species']

# Normalize features using Min-Max Scaling
print("\nApplying Min-Max Scaling to features...")
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Create scaled dataframe
df_scaled = pd.DataFrame(X_scaled, columns=feature_columns)
df_scaled['species'] = y.values
df_scaled['species_name'] = df['species_name'].values

print("Scaling complete!")
print("\nScaled data sample (first 5 rows):")
print(df_scaled.head())

# Encode class labels
print("\nEncoding class labels...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f"Original classes: {label_encoder.classes_}")
print(f"Encoded values: {np.unique(y_encoded)}")

# Step 3: Exploration
print("\n" + "=" * 50)
print("STEP 3: EXPLORATORY DATA ANALYSIS")
print("=" * 50)

# Summary statistics
print("\nSummary Statistics:")
print(df[feature_columns].describe())

# Visualization 1: Pairplot
print("\nGenerating pairplot...")
plt.figure(figsize=(12, 10))
pairplot = sns.pairplot(df, hue='species_name', palette='Set1', 
                        vars=feature_columns, diag_kind='kde')
pairplot.savefig('visualizations/pairplot.png', dpi=300, bbox_inches='tight')
plt.close()
print("Pairplot saved to visualizations/pairplot.png")

# Visualization 2: Correlation Heatmap
print("\nGenerating correlation heatmap...")
plt.figure(figsize=(8, 6))
correlation_matrix = df[feature_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
            center=0, square=True, linewidths=1)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('visualizations/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("Correlation heatmap saved to visualizations/correlation_heatmap.png")

print("\nCorrelation Matrix:")
print(correlation_matrix)

# Visualization 3: Boxplots for outlier detection
print("\nGenerating boxplots for outlier detection...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Boxplots for Outlier Detection', fontsize=16)

for idx, feature in enumerate(feature_columns):
    row = idx // 2
    col = idx % 2
    axes[row, col].boxplot(df[feature])
    axes[row, col].set_title(feature)
    axes[row, col].set_ylabel('Value (cm)')
    axes[row, col].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/boxplots.png', dpi=300, bbox_inches='tight')
plt.close()
print("Boxplots saved to visualizations/boxplots.png")

# Outlier analysis
print("\nOutlier Analysis:")
for feature in feature_columns:
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
    print(f"{feature}: {len(outliers)} potential outliers detected")

# Step 4: Train/Test Split Function
print("\n" + "=" * 50)
print("STEP 4: TRAIN/TEST SPLIT")
print("=" * 50)

def split_data(df, test_size=0.2, random_state=42):
    """
    Split dataset into training and testing sets
    
    Parameters:
    - df: pandas DataFrame with features and target
    - test_size: proportion of test set (default 0.2 for 80/20 split)
    - random_state: seed for reproducibility
    
    Returns:
    - X_train, X_test, y_train, y_test
    """
    # Define feature columns
    feature_cols = ['sepal length (cm)', 'sepal width (cm)', 
                    'petal length (cm)', 'petal width (cm)']
    
    # Separate features and target
    X = df[feature_cols]
    y = df['species']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

# Test the split function
X_train, X_test, y_train, y_test = split_data(df_scaled)

print(f"Training set size: {len(X_train)} samples ({len(X_train)/len(df)*100:.1f}%)")
print(f"Test set size: {len(X_test)} samples ({len(X_test)/len(df)*100:.1f}%)")
print(f"\nTraining set class distribution:")
print(y_train.value_counts().sort_index())
print(f"\nTest set class distribution:")
print(y_test.value_counts().sort_index())

# Summary
print("\n" + "=" * 50)
print("PREPROCESSING COMPLETE")
print("=" * 50)
print("\nGenerated files:")
print("- visualizations/pairplot.png")
print("- visualizations/correlation_heatmap.png")
print("- visualizations/boxplots.png")
print("\nDataset is now ready for modeling!")

if __name__ == "__main__":
    print("\nScript executed successfully!")
    print("All preprocessing steps completed.")
    print("Data is split and ready for clustering and classification tasks.")