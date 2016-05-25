import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSCanonical, PLSRegression, CCA
from sklearn.decomposition import PCA
from sklearn.datasets import load_linnerud

linnerud = load_linnerud()

X = linnerud.data
Y = linnerud.target

print("X dimensions:", X.shape)
print("Y dimensions:", Y.shape)

pca = PCA(n_components=2)
X_pca = pca.fit(X).transform(X)

# as Y has 3 columns, we can interpret it as RGB colors
Y_colors = Y / Y.max(axis=0)

plt.figure()
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=Y_colors, s=100)
plt.legend()
plt.title('PCA of Linnerud dataset')

plt.show()
