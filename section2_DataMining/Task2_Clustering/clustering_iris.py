"""
Task 2: Clustering on Iris Dataset

This script performs K-Means clustering on the preprocessed Iris dataset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import os

# Create directories for outputs
os.makedirs('visualizations', exist_ok=True)
os.makedirs('reports', exist_ok=True)

# Load and preprocess Iris dataset
def load_and_preprocess_iris():
    """
    Load Iris dataset and apply preprocessing
    Returns features (scaled) and true labels
    """
    # Load Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Normalize features using Min-Max scaling
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

# Load preprocessed data
X_scaled, y_true = load_and_preprocess_iris()

print("Dataset loaded and preprocessed")
print(f"Features shape: {X_scaled.shape}")
print(f"Number of samples: {X_scaled.shape[0]}")
print(f"Number of features: {X_scaled.shape[1]}")
print()

# Step 1: K-Means with k=3
print("=" * 50)
print("Step 1: K-Means Clustering with k=3")
print("=" * 50)

# Fit K-Means with k=3
kmeans_3 = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters_3 = kmeans_3.fit_predict(X_scaled)

# Calculate Adjusted Rand Index
ari_score = adjusted_rand_score(y_true, clusters_3)
print(f"Adjusted Rand Index (ARI): {ari_score:.4f}")
print(f"Inertia: {kmeans_3.inertia_:.4f}")
print()

# Step 2: Experiment with different k values
print("=" * 50)
print("Step 2: Experimenting with k=2 and k=4")
print("=" * 50)

# Try k=2
kmeans_2 = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters_2 = kmeans_2.fit_predict(X_scaled)
ari_2 = adjusted_rand_score(y_true, clusters_2)
print(f"k=2: ARI = {ari_2:.4f}, Inertia = {kmeans_2.inertia_:.4f}")

# Try k=4
kmeans_4 = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters_4 = kmeans_4.fit_predict(X_scaled)
ari_4 = adjusted_rand_score(y_true, clusters_4)
print(f"k=4: ARI = {ari_4:.4f}, Inertia = {kmeans_4.inertia_:.4f}")
print()

# Elbow curve analysis
print("Generating elbow curve...")
k_values = range(1, 7)
inertias = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Plot elbow curve
plt.figure(figsize=(10, 6))
plt.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (k)', fontsize=12)
plt.ylabel('Inertia', fontsize=12)
plt.title('Elbow Curve for Optimal k', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xticks(k_values)
plt.tight_layout()
plt.savefig('visualizations/elbow_curve.png', dpi=300, bbox_inches='tight')
print("Elbow curve saved to visualizations/elbow_curve.png")
plt.close()

# Step 3: Visualize clusters
print()
print("=" * 50)
print("Step 3: Cluster Visualization")
print("=" * 50)

# Create scatter plot with petal length and petal width
# Feature indices: 0=sepal length, 1=sepal width, 2=petal length, 3=petal width
plt.figure(figsize=(10, 7))

# Plot clusters
scatter = plt.scatter(X_scaled[:, 2], X_scaled[:, 3], 
                     c=clusters_3, cmap='viridis', 
                     s=100, alpha=0.6, edgecolors='black', linewidth=0.5)

# Add cluster centers
centers = kmeans_3.cluster_centers_
plt.scatter(centers[:, 2], centers[:, 3], 
           c='red', marker='X', s=300, 
           edgecolors='black', linewidth=2, 
           label='Centroids')

plt.xlabel('Petal Length (scaled)', fontsize=12)
plt.ylabel('Petal Width (scaled)', fontsize=12)
plt.title('K-Means Clustering (k=3) - Iris Dataset', fontsize=14, fontweight='bold')
plt.colorbar(scatter, label='Cluster')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/cluster_scatter.png', dpi=300, bbox_inches='tight')
print("Cluster scatter plot saved to visualizations/cluster_scatter.png")
plt.close()

# Summary statistics
print()
print("=" * 50)
print("Clustering Summary")
print("=" * 50)
print(f"Optimal k (from elbow): 3")
print(f"Best ARI score: {ari_score:.4f} (k=3)")
print(f"Cluster distribution: {np.bincount(clusters_3)}")
print()
print("All visualizations saved successfully!")
print("Analysis report should be written in reports/clustering_analysis.md")

if __name__ == "__main__":
    print("\nScript execution completed successfully!")