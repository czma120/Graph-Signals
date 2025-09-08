# %%
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from hgr_converter import write_hgr

#%%
#loading in subset of train, test, and validation
train_zinc = ZINC(root = '.data/ZINC', subset= True , split = 'train')
test_zinc = ZINC(root = '.data/ZINC', subset= True , split = 'test')
val_zinc = ZINC(root = '.data/ZINC', subset= True , split = 'val')
# %%

print(len(train_zinc))
print(len(test_zinc))
print(len(val_zinc))
# %%

graph = train_zinc[0]
print(graph.keys())
print(graph.num_nodes)
print(graph.num_edges)
#%%
print(graph.edge_index)
print(graph.edge_attrs)
# %%
