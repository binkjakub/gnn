import numpy as np
import scipy.sparse as ssp
import scipy.sparse.csgraph as csg
import torch
from torch_geometric.datasets import CitationFull

from gnn import DATA_PATH
from gnn.layers.spectral import SpectralConv

K_EIGENVALUES = 10

dataset = CitationFull(root=DATA_PATH, name='citeseer')
data = dataset.data

num_nodes = len(data.x)
row, col = data.edge_index.numpy()
entries = np.ones(data.edge_index.shape[-1])

adjacency_matrix = ssp.coo_matrix((entries, (row, col)), (num_nodes, num_nodes))
laplacian = csg.laplacian(adjacency_matrix, normed=True)
_, eigenvectors = ssp.linalg.eigsh(laplacian, k=K_EIGENVALUES)
eigenvectors = torch.as_tensor(eigenvectors, dtype=torch.float32)

print(torch.linalg.norm())

layer = SpectralConv(eigenvectors, 1)
h = layer.forward(data.x)
print(h.shape)
