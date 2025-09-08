# %%
#  WRITES GRAPH DATA INTO HGR FILES(hMETIS format)

"""
FIRST LINE: #of hyperedges  #of vertices  fmt

fmt = {
    1 if edges are weighted
    10 if vertices are weighted
    11 if both edges and vertices are weighted
    omitted if unweighted
}

EVERY I-TH LINE AFTER: contains vertices in the i-1th hyperedge
"""
def write_hgr(data, filepath):
    num_vertices = data.num_nodes

    edge_index = data.edge_index
    edges = {tuple(sorted((edge_index[0, i].item() + 1, edge_index[1, i].item() + 1)))
             for i in range(edge_index.size(1))}

    with open(filepath, 'w') as f:
        f.write(f"{len(edges)} {num_vertices}\n") 
        for u, v in edges:
            f.write(f"{u} {v}\n")



