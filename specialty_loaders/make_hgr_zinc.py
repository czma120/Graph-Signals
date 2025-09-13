# %%
#from zinc import train_zinc, test_zinc, val_zinc
from hgr_converter import write_hgr
import os
import ast
import pandas as pd
import torch
from types import SimpleNamespace

# %%
#making output folders
# os.makedirs("hgr/train", exist_ok=True)
# os.makedirs("hgr/test", exist_ok=True)
# os.makedirs("hgr/val", exist_ok=True)

os.makedirs("cluster_new_hgr/train", exist_ok=True)
os.makedirs("cluster_new_hgr/test", exist_ok=True)
os.makedirs("cluster_new_hgr/val", exist_ok=True)

# %%
out_base = "cluster_new_hgr"

def convert(csv_path, split):
    out_dir = os.path.join(out_base, split)
    df = pd.read_csv(csv_path)
    for i, row in df.iterrows():
        edge_index = torch.tensor(ast.literal_eval(row["edge_index"]), dtype=torch.long)
        if edge_index.shape[0] != 2:  # ensure shape (2,E)
            edge_index = edge_index.T
        data = SimpleNamespace(edge_index=edge_index, num_nodes=int(row["num_nodes"]))
        write_hgr(data, os.path.join(out_dir, f"graph_{i}.hgr"))

convert("cluster_graphs_train.csv", "train")
convert("cluster_graphs_val.csv",   "val")
convert("cluster_graphs_test.csv",  "test")
# %%
#converting training graphs
for i, data in enumerate(cluster_graphs_train.csv):
    write_hgr(data, f"cluster_new_hgr/train/graph_{i}.hgr")

# %%
#converting validation graphs
for i, data in enumerate(cluster_graphs_val.csv):
    write_hgr(data, f"cluster_new_hgr/val/graph_{i}.hgr")

# %%
#converting test graphs
for i, data in enumerate(cluster_graphs_test.csv):
    write_hgr(data, f"cluster_new_hgr/test/graph_{i}.hgr")

# %%
