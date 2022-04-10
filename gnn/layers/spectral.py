import torch
import torch.nn as nn


class SpectralConv(nn.Module):
    """Spectral Convolutional Layer as proposed by Bruna et al. (2014).

    Reference: https://arxiv.org/abs/1312.6203

    """

    def __init__(self, eigenvectors: torch.Tensor, num_feature_maps: int):
        """

        Args:
            eigenvectors: precomputed matrix of eigenvectors [n_nodes, k_eigenvalues]
            num_feature_maps: number of feature maps
        """
        super().__init__()
        self.eigenvectors = eigenvectors
        self.eigenvectors.requires_grad = False
        self.k = num_feature_maps

        num_data_points, num_eigenvalues = self.eigenvectors.shape
        filters = nn.init.normal_(torch.empty((num_eigenvalues,)))
        self.spectral_filters = nn.Parameter(filters)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs spectral convolution with learnable filter.

        Spectral convolution performs convolution with a filter in a spectral domain:
            1. Fourier transform:
                feature projection into spectral domain determined by orthonormal basis
                (eigenvectors of graph laplacian)
            2. Convolution:
                convolution with spectral filter (matrix multiplication of diag replaces hadamard)
            3. Inverse Fourier Transform:
                results projection into spatial domain

        """
        logits = self.eigenvectors @ torch.diag(self.spectral_filters) @ self.eigenvectors.T @ x
        return self.activation(logits)
