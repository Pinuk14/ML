from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.preprocessing import StandardScaler


df = pd.read_csv(Path(__file__).with_name("customers.csv"))
X = df.drop(columns=["Channel", "Region"])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
Z = linkage(X_scaled, method="ward", metric="euclidean")

plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Customer Index")
plt.ylabel("Euclidean Distance")
plt.tight_layout()
plt.show()

df["Cluster"] = fcluster(Z, t=3, criterion="maxclust")
print("\nCustomer Clusters:")
print(df[["Fresh", "Milk", "Grocery", "Cluster"]])

sns.scatterplot(data=df, x="Fresh", y="Milk", hue="Cluster", palette="Set1", s=100)
plt.title("Wholesale Customer Clusters")
plt.tight_layout()
plt.show()

new_values = [float(input(f"Enter {column} spending: ")) for column in X.columns]
new_customer = pd.DataFrame(scaler.transform([new_values]), columns=X.columns).iloc[0]
cluster_centroids = (
	pd.DataFrame(X_scaled, columns=X.columns)
	.assign(Cluster=df["Cluster"].to_numpy())
	.groupby("Cluster")
	.mean()
)
distances = cluster_centroids.sub(new_customer).pow(2).sum(axis=1)
print(
	"Nearest cluster prediction (centroid approximation):",
	distances.idxmin(),
)