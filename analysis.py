# %%
import torch
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
import pickle
import networkx as nx
#from zinc import test_zinc
import matplotlib.pyplot as plt
import pandas as pd
import math
import kahypar
import ast


# %%
df = pd.read_pickle("grid_rec_sort.pkl")


# %%
def read_edge_index(hgr_path):
    with open(hgr_path, 'r') as f:
        lines = f.readlines()[1:]  
        edge_index = [[int(x) - 1 for x in line.strip().split()] for line in lines]
        edge_index = torch.tensor(edge_index).T  #
    return edge_index

def read_num_nodes(hgr_path):
    with open(hgr_path, 'r') as f:
        first_line = f.readline().strip()
        _, num_nodes = map(int, first_line.split())
    return num_nodes


def compute_laplacian(edge_index, num_nodes):

    row, col = edge_index.numpy()
    # symteerize edge pairs 

    row_sym = np.concatenate([row, col])
    col_sym = np.concatenate([col, row])
    data = np.ones(len(row_sym))
    #adjacency matrix 
    A = coo_matrix((data, (row_sym, col_sym)), shape=(num_nodes, num_nodes)).tocsc()
    A.setdiag(0)
    A.eliminate_zeros()
    
    #degree matrix
    deg = np.array(A.sum(axis=1)).flatten()
    D = coo_matrix((deg, (np.arange(num_nodes), np.arange(num_nodes))), shape=(num_nodes, num_nodes)).tocsc()

    #unnormalized laplacian
    L = D - A
    return L


def rayleigh_quotient(signal, L):
    phi = np.array(signal)
    numerator = phi.T @ (L @ phi)
    denominator = phi.T @ phi
    return numerator / denominator

def build_partition_tree(hgr_path, k):

    context = kahypar.Context()
    context.loadINIconfiguration(config_path)
    context.setK(k)
    context.setEpsilon(ep)
    hypergraph = kahypar.createHypergraphFromFile(hgr_path, k)


    kahypar.partition(hypergraph, context)
    num_nodes = read_num_nodes(hgr_path)


    block_assignments = [hypergraph.blockID(i) for i in range(num_nodes)]


    return block_assignments
# %%

#one graph
i = 0
row = df.iloc[i]
edge_index = torch.tensor(row["edge_index"])
num_nodes = row["num_nodes"]
signals = row["signals"]

L = compute_laplacian(edge_index, num_nodes)
rq_values = [rayleigh_quotient(signal, L) for signal in signals]

plt.plot(range(1, len(rq_values) + 1), rq_values, marker='o')
plt.xlabel("Signal Number")
plt.ylabel("Rayleigh Quotient")
plt.title(f"Rayleigh Quotients (Graph ID {row['graph_id']})")
plt.grid(True)
plt.show()



# %%


# %% HISTOGRAMS of signal values
# %% Histogram of Rayleigh Quotients (frequencies) for df_dyadic

all_rq_values = []

for _, row in df_dyadic.iterrows():
    edge_index = torch.tensor(row["edge_index"])
    num_nodes = row["num_nodes"]
    signals = [np.array(s, dtype=float) for s in row["signals"]]

    L = compute_laplacian(edge_index, num_nodes)

    for s in signals:
        rq = rayleigh_quotient(s, L)
        all_rq_values.append(rq)

# Plot histogram of RQ values
plt.figure(figsize=(8, 5))
plt.hist(all_rq_values, bins=40, color='skyblue', edgecolor='black')
plt.xlabel("Rayleigh Quotient (Signal Frequency)")
plt.ylabel("Count")
plt.title("Distribution of Signal Frequencies (Recursive Signals)")
plt.grid(True)
plt.tight_layout()
plt.show()


# %%
indices = list(range(0, len(df), 3))  
n_plots = len(indices)
cols = 3
rows = math.ceil(n_plots / cols)

fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
axes = axes.flatten()

for ax_idx, i in enumerate(indices):
    row = df.iloc[i]
    edge_index = torch.tensor(row["edge_index"])
    num_nodes = row["num_nodes"]
    signals = row["signals"]

    L = compute_laplacian(edge_index, num_nodes)

    rq_values = [rayleigh_quotient(signal, L) for signal in signals]

    num_signals = len(signals)
    k_eigs = min(num_signals, num_nodes - 1) 

    try:
        eigvals_k, _ = eigsh(L, k=k_eigs, which='SM')
        eigvals_k = np.sort(eigvals_k)
    except Exception as e:
        print(f"Failed to compute eigvals for graph {row['graph_id']} with error: {e}")
        eigvals_k = [0] * k_eigs  

    if ax_idx == 0:
        print(f"\n--- Graph {row['graph_id']} ---")
        print("Num signals:", num_signals)
        print("Num eigenvalues computed:", k_eigs)
        print("Eigenvalues:", eigvals_k)


    x = range(1, k_eigs + 1)
    axes[ax_idx].plot(x, rq_values[:k_eigs], marker='o', label='METIS RQ')
    axes[ax_idx].plot(x, eigvals_k, marker='o', label='Eigenvalue')
    axes[ax_idx].set_title(f"Graph {row['graph_id']}")
    axes[ax_idx].set_xlabel("Index")
    axes[ax_idx].set_ylabel("Frequency")
    axes[ax_idx].legend()
    axes[ax_idx].grid(True)

for j in range(ax_idx + 1, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.show()

# %%

df_dyadic = pd.read_pickle("zinc_test_rec.pkl")  
df_haar = pd.read_pickle("zinc_test_haar.pkl")   
df_dyadic = df_dyadic.set_index("graph_id")
df_haar = df_haar.set_index("graph_id")


graph_ids = list(df_dyadic.index[::100]) 
n_plots = len(graph_ids)

#subpltos
cols = 3
rows = math.ceil(n_plots / cols)
fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
axes = axes.flatten()

#go thru the seleccted graphs to
for ax_idx, gid in enumerate(graph_ids):
    row_dy = df_dyadic.loc[gid]
    row_ha = df_haar.loc[gid]

    #edge index and num of nodes
    edge_index = torch.tensor(row_dy["edge_index"])
    num_nodes = row_dy["num_nodes"]

    #get signals
    signals_dyadic = [np.array(s, dtype=float) for s in row_dy["signals"]]
    signals_haar = [np.array(s, dtype=float) for s in row_ha["signals"]]

    #compute lapalcian of grpahs
    L = compute_laplacian(edge_index, num_nodes)

    #rq of the signals
    rq_dyadic = [rayleigh_quotient(s, L) for s in signals_dyadic]
    rq_haar = [rayleigh_quotient(s, L) for s in signals_haar]
    
    #k smallest eigvvals
    k_eigs = min(len(rq_dyadic + rq_haar), num_nodes - 1)
    eigvals_k, _ = eigsh(L, k=k_eigs, which='SM')
    eigvals_k = np.sort(eigvals_k)


    x = range(1, k_eigs + 1)

    ax = axes[ax_idx]
    ax.plot(x, rq_dyadic[:k_eigs], marker='o', label='Recursive RQ')
    ax.plot(x, rq_haar[:k_eigs], marker='o', label='Haar RQ')
    ax.plot(x, eigvals_k, marker='o', label='Eigenvalue')
    ax.set_title(f"Graph {gid}")
    ax.set_xlabel("Index")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True)

for j in range(ax_idx + 1, len(axes)):
    axes[j].axis("off")

plt.show()


