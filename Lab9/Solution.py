import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA

# 1. Export/Load the Olivetti faces dataset
# fetch_olivetti_faces() returns a Bunch object containing images and data
faces = fetch_olivetti_faces(shuffle=True, random_state=42)
images = faces.images  # Shape: (400, 64, 64)
targets = faces.target

# a) Flatten images into vectors (array form)
# The dataset already provides flattened vectors in 'faces.data' (400, 4096)
# Alternatively, manual reshaping: images.reshape(images.shape[0], -1)
X = faces.data 
n_samples, n_features = X.shape
print(f"Dataset loaded: {n_samples} images, each flattened to {n_features} features.")

# Define PCA components to test
n_comp_list = [25, 50, 100]
plt.figure(figsize=(12, 8))

# Select a random image index to visualize reconstruction
sample_idx = 10
original_face = X[sample_idx].reshape(64, 64)

# Plot Original
plt.subplot(1, 4, 1)
plt.imshow(original_face, cmap='gray')
plt.title("Original Face")
plt.axis('off')

# b) Apply PCA and c) Reconstruct images for each case
for i, n_comp in enumerate(n_comp_list):
    # Apply PCA with specified components
    pca = PCA(n_components=n_comp, whiten=True, random_state=42)
    X_pca = pca.fit_transform(X)
    
    # Reconstruct the image from the reduced components
    X_reconstructed = pca.inverse_transform(X_pca)
    reconstructed_face = X_reconstructed[sample_idx].reshape(64, 64)
    
    # Plot Reconstructed
    plt.subplot(1, 4, i + 2)
    plt.imshow(reconstructed_face, cmap='gray')
    plt.title(f"PCA ({n_comp} comp)")
    plt.axis('off')

plt.tight_layout()
plt.show()

# d) Conclusions
print("\nConclusions:")
print("1. Dimensionality Reduction: PCA significantly reduces the data size from 4096 features to as low as 25.")
print("2. Quality vs. Components: As the number of components increases (25 -> 50 -> 100), the reconstructed face becomes sharper and captures more unique facial details.")
print("3. Information Loss: At 25 components, the face is recognizable but blurry, indicating loss of high-frequency spatial details. At 100 components, the reconstruction is very close to the original.")
