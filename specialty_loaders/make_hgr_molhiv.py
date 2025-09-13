# %%
import torch
import torch
import torch_geometric
import os


torch.serialization.add_safe_globals([
    torch_geometric.data.data.DataEdgeAttr,
    torch_geometric.data.data.DataTensorAttr,
    torch_geometric.data.storage.GlobalStorage
])
from ogb.graphproppred import PygGraphPropPredDataset

# %%


dataset = PygGraphPropPredDataset(name='ogbg-molhiv', root='data')


# %%
from hgr_converter import write_hgr

os.makedirs("molhiv_hgr", exist_ok=True)

# convert each graph to .hgr format
for i, data in enumerate(dataset):
    write_hgr(data, f"molhiv_hgr/graph_{i}.hgr")


