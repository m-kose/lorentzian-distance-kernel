import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.decomposition import KernelPCA, PCA
from sklearn.svm import SVC
from sklearn.datasets import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def lorentz_metric_tensor(n, U, Y):
    G = np.zeros((n, n))
    G[:-1, :-1] = U * np.eye(n - 1)  # Top left block
    G[-1, -1] = -Y  # Bottom right element
    return G

def lorentzian_rbf_kernel(X, Y=None, gamma=1.0, G=None):
    if Y is None:
        Y = X
    if G is None:
        raise ValueError("The Lorentz metric tensor G must be provided")
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)

    X_reshaped = X[:, np.newaxis, :]
    Y_reshaped = Y[np.newaxis, :, :]
    differences = X_reshaped - Y_reshaped

    distances = np.sqrt(np.abs(np.einsum('ijk,kl,ijl->ij', differences, G, differences)))
    return np.exp(-gamma * distances)

def lorentz_transform(X, alpha):
    kernel = np.array([[np.cosh(alpha), np.sinh(alpha)], [np.sinh(alpha), np.cosh(alpha)]])
    result = np.dot(X, kernel)
    return result

dataset = load_wine()

X, y = dataset.data, dataset.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

alpha = np.pi / 2
combined_value = 0.946
X_train_transformed = lorentz_transform(X_train_pca, alpha)
X_test_transformed = lorentz_transform(X_test_pca, alpha)

n_features = X_train_transformed.shape[1]
G = lorentz_metric_tensor(n_features, 1/combined_value, combined_value)

kernel_train_transformed = lorentzian_rbf_kernel(X_train_transformed, gamma=combined_value, G=G)
kernel_test_transformed = lorentzian_rbf_kernel(X_test_transformed, X_train_transformed, gamma=combined_value, G=G)

svc_transformed = SVC(kernel="precomputed")
svc_transformed.fit(kernel_train_transformed, y_train)
svm_score_transformed = svc_transformed.score(kernel_test_transformed, y_test)
print(svm_score_transformed, "ACCURACY")