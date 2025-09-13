# %%
import torch
import torch
import torch_geometric
import os
from hgr_converter import write_hgr

os.environ["OGB_DOWNLOAD_NOW"] = "true"


torch.serialization.add_safe_globals([
    torch_geometric.data.data.DataEdgeAttr,
    torch_geometric.data.data.DataTensorAttr,
    torch_geometric.data.storage.GlobalStorage
])
from ogb.graphproppred import PygGraphPropPredDataset

# %%
dataset = PygGraphPropPredDataset(name='ogbg-ppa', root='data')


# %%
os.makedirs("ppa_hgr", exist_ok=True)

for i, data in enumerate(dataset):
    write_hgr(data, f"ppa_hgr/graph_{i}.hgr")