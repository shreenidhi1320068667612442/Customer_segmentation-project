import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load dataset
data = pd.read_csv(r"C:\Users\acer\Downloads\projects\store_customers.csv")

# Select columns
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]
X=X.dropna()

# Apply KMeans
kmeans = KMeans(n_clusters=3, random_state=0)

X['Cluster'] = kmeans.fit_predict(X)
X.to_csv("clustered_customers.csv",index="False")
# Print first rows
print(X.head())

# Visualize clusters
plt.scatter(
    X['Annual Income (k$)'],
    X['Spending Score (1-100)'],
    c=X['Cluster']
)

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation")

plt.show()
