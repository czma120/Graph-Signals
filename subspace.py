# %%
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
from tqdm import tqdm

# %%
df = pd.read_pickle("zinc_test_part_rec_full.pkl")

# %%
def compute_laplacian(edge_index, num_nodes):
    row, col = edge_index.numpy()
    row_sym = np.concatenate([row, col])
    col_sym = np.concatenate([col, row])
    data = np.ones(len(row_sym))
    A = coo_matrix((data, (row_sym, col_sym)), shape=(num_nodes, num_nodes)).tocsc()
    A.setdiag(0)
    A.eliminate_zeros()
    deg = np.array(A.sum(axis=1)).flatten()
    D = coo_matrix((deg, (np.arange(num_nodes), np.arange(num_nodes))), shape=(num_nodes, num_nodes)).tocsc()
    return D - A

def projection(x_hat, U):
  
    norm_sq = np.dot(x_hat, x_hat)

    return float((x_hat.T @ (U @ (U.T @ x_hat))) / norm_sq)


k =15

#%%
#only every 100th
for i, row in tqdm(df.iterrows(), total=len(df)):
    if i % 100 != 0:
        continue 

    edge_index = torch.tensor(row['edge_index'])
    num_nodes = row['num_nodes']
    signals = row['signals']

    if num_nodes <= k:
        continue  

    try:
        L = compute_laplacian(edge_index, num_nodes)
        eigvals, eigvecs = eigsh(L, k=k, which='SM')
        U = eigvecs

        #compute projection ratios
        scores = [projection(np.array(sig), U) for sig in signals]


        plt.figure(figsize=(8, 4))
        plt.plot(range(1, len(scores) + 1), scores, marker='o')
        plt.ylim(0, 1.05)
        plt.xlabel("Signal Index")
        plt.ylabel("subspace similarity")
        plt.title(f"Graph {i}: METIS Signal Projection (n = {num_nodes})")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Skipping graph {i} due to error: {e}")
        continue

