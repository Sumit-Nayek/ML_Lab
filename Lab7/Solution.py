import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.decomposition import KernelPCA

# 1. Generate concentric circles (2 features) + 3 noise features = 5 features
X, y = make_circles(n_samples=400, factor=0.3, noise=0.05, random_state=42)
extra_noise = np.random.normal(0, 0.05, (400, 3))
X_5d = np.hstack((X, extra_noise)) 

print(f"Dataset shape: {X_5d.shape}") # (400, 5)