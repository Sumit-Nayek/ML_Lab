import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.decomposition import KernelPCA

# 1. Generate concentric circles (2 features) + 3 noise features = 5 features
X, y = make_circles(n_samples=400, factor=0.3, noise=0.05, random_state=42)
extra_noise = np.random.normal(0, 0.05, (400, 3))
X_5d = np.hstack((X, extra_noise)) 

print(f"Dataset shape: {X_5d.shape}") # (400, 5)
# a) RBF Kernel - 2D and 3D
kpca_rbf_2d = KernelPCA(n_components=2, kernel="rbf", gamma=15)
X_rbf_2d = kpca_rbf_2d.fit_transform(X_5d)

kpca_rbf_3d = KernelPCA(n_components=3, kernel="rbf", gamma=15)
X_rbf_3d = kpca_rbf_3d.fit_transform(X_5d)

# Visualization
fig = plt.figure(figsize=(12, 5))

# 2D Plot
ax1 = fig.add_subplot(121)
ax1.scatter(X_rbf_2d[:, 0], X_rbf_2d[:, 1], c=y, cmap='viridis')
ax1.set_title("RBF Kernel PCA (2D)")

# 3D Plot
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(X_rbf_3d[:, 0], X_rbf_3d[:, 1], X_rbf_3d[:, 2], c=y, cmap='viridis')
ax2.set_title("RBF Kernel PCA (3D)")
plt.show()