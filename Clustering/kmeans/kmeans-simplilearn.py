from sklearn.datasets import load_sample_image
import numpy as np
import matplotlib.pyplot as plt
china = load_sample_image('flower.jpg')
ax = plt.axes(xticks=[],yticks=[])
ax.imshow(china)

data = china/255.0
data = data.reshape(427*640,3)

def plot_pixels(data,title,colors=None,N=10000):
    if colors is None:
        colors=data
    rng =np.random.RandomState(0)
    i = rng.permutation(data.shape[0])[:N]
    colors = colors[i]
    R,G,B = data[i].T

plot_pixels(data, title='Input 16 Mill colors')

from sklearn.cluster import MiniBatchKMeans
kmean = MiniBatchKMeans(16)
kmean.fit(data)
ncol = kmean.cluster_centers_[kmean.predict(data)]
plot_pixels(data,colors=ncol,title='16 colors')

china_recol = ncol.reshape(china.shape)

fig,ax = plt.subplots(1,2, figsize=(16,6),subplot_kw=dict(xticks=[],yticks=[]))
fig.subplots_adjust(wspace=0.05)
ax[0].imshow(china)
ax[1].imshow(china_recol)
