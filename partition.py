# %%
import os
import math
import kahypar
import torch
import pandas as pd
from tqdm import tqdm
import pickle
import random
from collections import deque
import time



# %%
config_path = "config/km1_rKaHyPar_sea20.ini"
ep = 0.03 #allowed imbalance in blocks

# %%
# to read the first line of hgr and get the number of nodes
def read_num_nodes(hgr_path):
    with open(hgr_path, 'r') as f:
        first_line = f.readline().strip()
        _, num_nodes = map(int, first_line.split())
    return num_nodes

#get hypergraph edge list from hgr file and make into pytroch edge index form
def read_edge_index(hgr_path):
    with open(hgr_path, 'r') as f:
        lines = f.readlines()[1:]  # skip first line
        edge_index = [[int(x) - 1 for x in line.strip().split()] for line in lines]
        edge_index = torch.tensor(edge_index).T.tolist()
    return edge_index


#make parittion to signal
def signal_from_partition(partition_vector):
    block_ids = sorted(set(partition_vector))
    k = len(block_ids)

    chosen_blocks = set(random.sample(block_ids, k=k // 2))
    signal = [+1 if block in chosen_blocks else -1 for block in partition_vector]
    return signal

def extract_signals(partition_vector, num_levels=None, mode="recursive"):

    num_nodes = len(partition_vector)
    block_ids = sorted(set(partition_vector)) 
    k = len(block_ids)


    signals = []

    if mode == "recursive":
        assert num_levels is not None, "num_levels required for recursive mode"
        for level in range(num_levels):
            signal = []
            for block in partition_vector:
                if (block >> level) & 1 == 0:
                    signal.append(+1)
                else:
                    signal.append(-1)
            signals.append(signal)


    elif mode == "kway":
        for _ in range(k):  #always generate k random signals
            chosen_blocks = set(random.sample(block_ids, k=k // 2))  #randomly choose half of the blocks
            signal = []
            for block in partition_vector:
                if block in chosen_blocks:
                    signal.append(+1)
                else:
                    signal.append(-1)
            signals.append(signal)


    return signals




#that ccombinaorial smoothing idea
def compute_weights(num_nodes, edge_index, partition_vector):
    # making graph again from hgr
    G = [[] for _ in range (num_nodes)]
    for u, v in zip(*edge_index):
        G[u].append(v)
        G[v].append(u)
    

    w = [0] * num_nodes
    visited = [False] * num_nodes
    q = deque()


    for v in range(num_nodes):
        for v_prime in G[v]:
            if partition_vector[v] != partition_vector[v_prime]:
                w[v] = 1
                visited[v] = True
                q.append(v)
                break 
    while q:
        v = q.popleft()
        for v_prime in G[v]:
            if not visited[v_prime]:
                w[v_prime] = w[v] + 1
                visited[v_prime] = True
                q.append(v_prime)

    return w



def partition_graph(hgr_path, mode='recursive', max_k=15):
    edge_index = read_edge_index(hgr_path)
    num_nodes = read_num_nodes(hgr_path)

    if mode == "recursive":
        num_levels = math.floor(math.log2(num_nodes))
        final_k = 2 ** num_levels
    else:
        final_k = min(max_k, num_nodes)
        num_levels = None

    if mode == "new_kway":
        signals = []
        for k in range(2, max_k + 1):
            hypergraph = kahypar.createHypergraphFromFile(hgr_path, k)

            context = kahypar.Context()
            context.loadINIconfiguration(config_path)
            context.setK(k)
            context.setEpsilon(ep)

            kahypar.partition(hypergraph, context)
            partition_vector = [hypergraph.blockID(i) for i in range(num_nodes)]

            signal = signal_from_partition(partition_vector)
            signals.append(signal)
        return signals 

    context = kahypar.Context()
    context.loadINIconfiguration(config_path)
    context.setK(final_k)
    context.setEpsilon(ep)

    hypergraph = kahypar.createHypergraphFromFile(hgr_path, final_k)
    kahypar.partition(hypergraph, context)

    partition_vector = [hypergraph.blockID(i) for i in range(num_nodes)]

    signals = extract_signals(partition_vector, num_levels=num_levels, mode=mode)

    if mode == "combdist":
        weights = compute_weights(num_nodes, edge_index, partition_vector)
        signals = [[val * weights[i] for i, val in enumerate(sig)] for sig in signals]

    return signals




#formats all paritiotns resutls
def partition_to_df(input_dir, use_split_name=True, mode='recursive'):
    
    all_records = []

    for fname in tqdm(sorted(os.listdir(input_dir))):
        if not fname.endswith(".hgr"):
            continue

        if use_split_name:
           
            graph_id = int(fname.split("_")[1].split(".")[0])
        else:
            graph_id = len(all_records)

        hgr_path = os.path.join(input_dir, fname)

        num_nodes = read_num_nodes(hgr_path)
        edge_index = read_edge_index(hgr_path)

        signals = partition_graph(hgr_path, mode=mode)

        # signals = partition_graph(hgr_path)

        all_records.append({
            'graph_id': graph_id,
            'edge_index': edge_index,
            'num_nodes': num_nodes,
            'signals': signals  # shape = [L x n]
        })

    return pd.DataFrame(all_records)

          
# %%

# start_time = time.time()

# df = partition_to_df(".", use_split_name=False, mode="recursive")

# end_time = time.time()
# print(f"Partitioning took {end_time - start_time:.2f} seconds")


#print(df.head())





# %%

# MOLHIV
#df_molhiv_rec = partition_to_df("molhiv_hgr", use_split_name = False, mode="recursive")
#df_molhiv_kway = partition_to_df("molhiv_hgr", use_split_name = False, mode="kway")


#ZINC
# df_zinc_train_rec = partition_to_df("zinc_hgr/train", use_split_name=True, mode='recursive')
# df_zinc_val_rec = partition_to_df("zinc_hgr/val", use_split_name=True,  mode='recursive')
# df_zinc_test_rec = partition_to_df("zinc_hgr/test", use_split_name=True,  mode='recursive')


#df_zinc_train_kway = partition_to_df("zinc_hgr/train", use_split_name=True, mode='kway')
#df_zinc_val_kway = partition_to_df("zinc_hgr/val", use_split_name=True,  mode='kway')
#df_zinc_test_kway = partition_to_df("zinc_hgr/test", use_split_name=True,  mode='kway')


#df_zinc_val_kway = partition_to_df("zinc_hgr/val", use_split_name=True,  mode='combdist')

# %%

#ZINC
# df_zinc_test_rec.to_pickle("zinc_test_dy.pkl")
# df_zinc_train_rec.to_pickle("zinc_train_dy.pkl")
# df_zinc_val_rec.to_pickle("zinc_val_dy.pkl")

# df_zinc_test_kway.to_pickle("zinc_test_part_kway.pkl", protocol=4)
# df_zinc_train_kway.to_pickle("zinc_train_part_kway.pkl", protocol=4)
# df_zinc_val_kway.to_pickle("zinc_val_part_kway.pkl", protocol=4)



#MOLHIV
#df_molhiv_rec.to_pickle("molhiv_part_rec.pkl")
#df_molhiv_kway.to_pickle("molhiv_part_kway.pkl", protocol=4)

# %%

#df_zvalpk= pd.read_pickle("pickle_files/zinc_val_part_kway.pkl")

#print(df_zvalpk.head())

#df_zvalpk.to_csv("zinc_val_kway.csv", index=False, encoding='us-ascii') 
# %%

#test_df_load = pd.read_csv("zinc_test_part_kway.csv")
# %%