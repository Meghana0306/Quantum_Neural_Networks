import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# --- Create synthetic dataset ---
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'id': range(1, n_samples + 1),
    'age': np.random.randint(18, 71, n_samples),
    'income': np.random.randint(30000, 150001, n_samples),
    'spending_score': np.random.randint(1, 101, n_samples),
    'purchase_frequency': np.random.randint(1, 51, n_samples)
})

print("Synthetic Dataset Preview:")
print(data.head())

# --- Feature Selection ---
features = ['age', 'income', 'spending_score', 'purchase_frequency']
X = data[features].values

# --- Manual Standard Scaling ---
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_scaled = (X - X_mean) / X_std

# --- Simulated "quantum" encoding ---
def encode_features(X):
    return np.hstack([np.sin(X), np.cos(X)])

encoded_features = encode_features(X_scaled)

# --- Custom KMeans implementation ---
def kmeans(X, n_clusters=3, n_iters=100):
    np.random.seed(42)
    centroids = X[np.random.choice(len(X), n_clusters, replace=False)]

    for _ in range(n_iters):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(n_clusters)])

        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return labels, centroids

# --- Run Clustering ---
clusters, centroids = kmeans(encoded_features, n_clusters=3)
data['Cluster'] = clusters

# --- 2D Scatter Plot: Income vs Spending Score ---
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='income', y='spending_score', hue='Cluster', palette='viridis', size='age', sizes=(20, 200))
plt.title('Customer Segmentation (Income vs Spending Score)')
plt.xlabel('Income')
plt.ylabel('Spending Score')
plt.savefig('customer_segments_manual.png')
plt.show()

# --- Additional 2D Scatter Plot: Age vs Purchase Frequency ---
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='age', y='purchase_frequency', hue='Cluster', palette='viridis', size='spending_score', sizes=(20, 200))
plt.title('Clusters by Age and Purchase Frequency')
plt.xlabel('Age')
plt.ylabel('Purchase Frequency')
plt.grid(True)
plt.savefig('cluster_age_vs_purchase.png')
plt.show()

# --- 3D Scatter Plot: Age, Income, Spending Score ---
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(data['age'], data['income'], data['spending_score'],
                c=data['Cluster'], cmap='viridis', s=data['purchase_frequency'], alpha=0.7)

ax.set_xlabel('Age')
ax.set_ylabel('Income')
ax.set_zlabel('Spending Score')
plt.title("3D Customer Segmentation")
plt.savefig("3d_customer_clusters.png")
plt.show()

# --- Bar Graph: Average Spending Score by Cluster ---
plt.figure(figsize=(8, 5))
cluster_avg = data.groupby('Cluster')['spending_score'].mean().reset_index()
sns.barplot(data=cluster_avg, x='Cluster', y='spending_score', palette='viridis')
plt.title('Average Spending Score per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Average Spending Score')
plt.savefig("bar_spending_score_per_cluster.png")
plt.show()

# --- Histogram: Spending Score Distribution by Cluster ---
plt.figure(figsize=(5, 5))
sns.histplot(data=data, x='spending_score', hue='Cluster', multiple='stack', palette='viridis', bins=30)
plt.title('Spending Score Distribution by Cluster')
plt.xlabel('Spending Score')
plt.ylabel('Count')
plt.savefig("hist_spending_score_by_cluster.png")
plt.show()

# --- Cluster Statistics ---
print("\nCluster Statistics:")
print(data.groupby('Cluster')[features].mean())

# --- Save Dataset ---
data.to_csv('segmented_customers_manual.csv', index=False)
print("\nDataset with clusters saved as 'segmented_customers_manual.csv'")