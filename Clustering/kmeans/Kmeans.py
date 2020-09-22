#import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("../mall_cust.csv")
x = dataset.iloc[:,[3,4]].values

from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, random_state=0).fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title("Elbow Mathod")
plt.show()

kmeans = KMeans(n_clusters=5, init='k-means++', n_init=10, max_iter=300, random_state=0)
y_kmeans = kmeans.fit_predict(x)

plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],s=50,c='red',label='Standard')
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],s=50,c='green',label='Careless')
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],s=50,c='blue',label='Target')
plt.scatter(x[y_kmeans==3,0],x[y_kmeans==3,1],s=50,c='magenta',label='Sensible')
plt.scatter(x[y_kmeans==4,0],x[y_kmeans==4,1],s=50,c='cyan',label='Carefull')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c='black',label='Centroids')
plt.title("Cluster of CLients")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.legend()
plt.show()






# from sklearn.datasets.samples_generator import make_blobs
# x,y = make_blobs(n_samples=300,centers=4,cluster_std=0.6,random_state=0)
