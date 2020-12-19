import numpy as np
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


X = np.random.normal(size=(1000, 5))
pca = PCA(n_components=4)  # n_components is the number of dimensions to project to
X_four_d = pca.fit_transform(X)

# Now plot in 4D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = X_four_d[:,0]
y = X_four_d[:,1]
z = X_four_d[:,2]
c = X_four_d[:,3]

img = ax.scatter(x, y, z, c=c, cmap=plt.hot())
fig.colorbar(img)
plt.show()
