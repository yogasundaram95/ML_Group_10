
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from scipy.stats import f_oneway
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage

# Load dataset
df = pd.read_csv("ML_OBJ3_final_data.csv")
df.dropna(inplace=True)

# Identify columns
company_cols = ['INTC', 'AMD', 'NVDA', 'TSM', 'TXN', 'ASML', 'QCOM', 'AVGO', 'AMAT']
external_cols = ['InterestRate', 'CPI', 'Sentiment']
all_features = company_cols + external_cols

# Standardize features
scaler = StandardScaler()
X_all = scaler.fit_transform(df[all_features])
X_no_external = scaler.fit_transform(df[company_cols])

# KMeans with all features
kmeans_all = KMeans(n_clusters=2, random_state=42)
labels_all = kmeans_all.fit_predict(X_all)
silhouette_all = silhouette_score(X_all, labels_all)
dbi_all = davies_bouldin_score(X_all, labels_all)

# KMeans without external features
kmeans_base = KMeans(n_clusters=2, random_state=42)
labels_base = kmeans_base.fit_predict(X_no_external)
ari_score = adjusted_rand_score(labels_all, labels_base)

# ANOVA on external features
df['Cluster'] = labels_all
anova_results = {
    col: f_oneway(*(df[df['Cluster'] == c][col] for c in sorted(df['Cluster'].unique())))
    for col in external_cols
}

# Cluster characterization
cluster_means = df.groupby('Cluster')[all_features].mean()
cluster_means.to_csv("cluster_means_summary.csv", index=True)

# Temporal Stability (pre vs. post 2022)
df['Date'] = pd.to_datetime(df['Date'])
pre_2022 = df[df['Date'] < "2022-01-01"]
post_2022 = df[df['Date'] >= "2022-01-01"]
X_pre = scaler.fit_transform(pre_2022[all_features])
X_post = scaler.fit_transform(post_2022[all_features])

labels_pre = KMeans(n_clusters=2, random_state=42).fit_predict(X_pre)
labels_post = KMeans(n_clusters=2, random_state=42).fit_predict(X_post)
ari_temporal = adjusted_rand_score(labels_pre[:min(len(labels_pre), len(labels_post))],
                                   labels_post[:min(len(labels_pre), len(labels_post))])

# PCA Visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_all)
pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
pca_df['Cluster'] = labels_all
plt.figure(figsize=(8, 5))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Cluster', palette='tab10')
plt.title("PCA Cluster Visualization")
plt.savefig("pca_clusters.png")
plt.close()

# Dendrogram
linkage_matrix = linkage(X_all, method='ward')
plt.figure(figsize=(10, 5))
dendrogram(linkage_matrix, truncate_mode='level', p=5)
plt.title("Hierarchical Clustering Dendrogram")
plt.savefig("dendrogram.png")
plt.close()

# Cluster feature bar chart
cluster_melted = cluster_means.reset_index().melt(id_vars='Cluster', var_name='Feature', value_name='Value')
plt.figure(figsize=(10, 6))
sns.barplot(data=cluster_melted, x='Feature', y='Value', hue='Cluster')
plt.xticks(rotation=45)
plt.title("Average Feature Values per Cluster")
plt.tight_layout()
plt.savefig("cluster_feature_bars.png")
plt.close()

# Print Summary
print("Silhouette Score:", silhouette_all)
print("Davies-Bouldin Index:", dbi_all)
print("Adjusted Rand Index (w/ vs. w/o external):", ari_score)
print("Temporal Stability ARI:", ari_temporal)
for col in external_cols:
    print(f"{col} ANOVA F={anova_results[col].statistic:.2f}, p={anova_results[col].pvalue:.2e}")
