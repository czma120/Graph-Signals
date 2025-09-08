# %%
import pickle
import os
import torch
from torch_geometric.data import Data
from hgr_converter import write_hgr
import zipfile
import pandas as pd
import ast




# %%

output_dir = "grid_hgr"
csv_path = "grid_graphs.csv" 
os.makedirs(output_dir, exist_ok=True)
df = pd.read_csv(csv_path)


for i, row in df.iterrows():
    edge_index_list = ast.literal_eval(row['edge_index'])
    edge_index = torch.tensor(edge_index_list, dtype=torch.long)

    class DummyData:
        pass

    data = DummyData()
    data.edge_index = edge_index
    data.num_nodes = row['num_nodes']

    out_path = os.path.join(output_dir, f"graph_{i}.hgr")
    write_hgr(data, out_path)

print(f"Converted {len(df)} graphs to {output_dir}")
# %%


#input csv
csv_files = {
    "train": "cluster_graphs_train.csv",
    "test":  "cluster_graphs_test.csv",
    "val":   "cluster_graphs_val.csv",
}

#output dir
out_base = "cluster_new_hgr"
os.makedirs(out_base, exist_ok=True)

for split, csv_path in csv_files.items():
    out_dir = os.path.join(out_base, split)
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    for i, row in df.iterrows():
        edge_index_list = ast.literal_eval(row['edge_index'])
        edge_index = torch.tensor(edge_index_list, dtype=torch.long)

        if edge_index.ndim == 2 and edge_index.shape[0] != 2:
            edge_index = edge_index.T

        class DummyData:
            pass

        data = DummyData()
        data.edge_index = edge_index
        data.num_nodes = int(row['num_nodes'])

        out_path = os.path.join(out_dir, f"graph_{i}.hgr")
        write_hgr(data, out_path)

    print(f"Converted {len(df)} graphs to {out_dir}")

print("All splits converted to cluster_new_hgr/")
# %%
