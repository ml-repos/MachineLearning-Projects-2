import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("mall_cust.csv")
x = dataset.iloc[:,[3,4]].values

import scipy.cluster.hierarchy as ch
dendogram = ch.dendrogram(ch.linkage(x,method='ward'))
plt.title('Dendogram')
plt.xlabel('Customer')
plt.ylabel('Euclidean Distance')
plt.show()

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, linkage='ward')
y_hc =hc.fit_predict(x)

plt.scatter(x[y_hc==0,0],x[y_hc==0,1],s=50,c='red',label='Carefull')
plt.scatter(x[y_hc==1,0],x[y_hc==1,1],s=50,c='green',label='Standard')
plt.scatter(x[y_hc==2,0],x[y_hc==2,1],s=50,c='blue',label='Target')
plt.scatter(x[y_hc==3,0],x[y_hc==3,1],s=50,c='magenta',label='Careless')
plt.scatter(x[y_hc==4,0],x[y_hc==4,1],s=50,c='cyan',label='Sensible')
plt.title("Cluster of CLients")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.legend()
plt.show()
