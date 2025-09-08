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
import numpy as np
import itertools



# %%
config_path = "config/km1_rKaHyPar_sea20.ini" 
ep = 0.03 

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



#run kahypar, get block ids
def build_partition_tree(hgr_path):
    num_nodes = read_num_nodes(hgr_path)
    d = math.ceil(math.log2(num_nodes))
    k = 2 ** d

    context = kahypar.Context()
    context.loadINIconfiguration(config_path)
    context.setK(k)
    context.setEpsilon(ep)
    hypergraph = kahypar.createHypergraphFromFile(hgr_path, k)


    kahypar.partition(hypergraph, context)
    num_nodes = read_num_nodes(hgr_path)


    block_assignments = [hypergraph.blockID(i) for i in range(num_nodes)]


    return block_assignments


#recursive signals
def extract_signals(block_assignments):
    n = len(block_assignments)
    d = max(len(format(b, 'b')) for b in block_assignments)

    #get binary strings for each blockid of lenght d
    bin_ids = [format(b % 2**d, f"0{d}b") for b in block_assignments]

    #gets dyadic signals
    bit_signals = []
    for level in range(d):
        signal = np.array([+1 if bin_id[level] == '0' else -1 for bin_id in bin_ids])
        bit_signals.append(signal)

    signals= []
    #signals = [np.ones(n, dtype=int).tolist()]#this is phi1

    for i in range(2**d):

        bits = format(i, f"0{d}b")#convert
        signal = np.ones(n) #phi1

        for j, b in enumerate(bits[::-1]):     
            if b == '1':
                signal *= bit_signals[j]

        signals.append(signal.astype(int).tolist())
    
    return signals


#haar signals
def extract_haar(block_assignments):
    n = len(block_assignments)
    #longest binary rep of all block IDs
    d = max(len(format(b, 'b')) for b in block_assignments)
    #get binary strings for each blockid oh lenght d
    bin_ids = [format(b % 2**d, f"0{d}b") for b in block_assignments]

    signals = []
    signals.append(np.ones(n, dtype=int).tolist()) #phi1

    #iterate over binary tree
    for level in range(d):
        pre_to_nodes = {}
        for idx, bstr in enumerate(bin_ids):
            pre = bstr[:level]  #current internal node, shared cut history
            child_bit = bstr[level]  #which side of the split
            #initialize
            if pre not in pre_to_nodes:
                pre_to_nodes[pre] = { '0': [], '1': [] }

            pre_to_nodes[pre][child_bit].append(idx)#add current node idx

        #make haar signal
        for pre, children in pre_to_nodes.items():
            left = children['0']
            right = children['1']
    
            #+1 if left side of split, -1 if right, 0 otherwise(in other partition group)
            signal = np.zeros(n, dtype = int)
            signal[left] = +1
            signal[right] = -1
            signals.append(signal.tolist())

    return signals

#function to call 
def partition_to_df(input_dir, use_split_name=True, mode= "rec"):
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
  

        block_assignments = build_partition_tree(hgr_path)

  
        if mode == "rec":
            signals = extract_signals(block_assignments)
        elif mode == "haar":
            signals = extract_haar(block_assignments)


        all_records.append({
            'graph_id': graph_id,
            'edge_index': edge_index,
            'num_nodes': num_nodes,
            'signals': signals  
        })

    return pd.DataFrame(all_records)
# %%

# start_time = time.time()

# df = partition_to_df(".", use_split_name=False, mode="haar")
# end_time = time.time()

# print(f"Partitioning took {end_time - start_time:.2f} seconds")
# print(df.head())

# print("Signals matrix shape:", len(df.iloc[0]['signals']), "x", len(df.iloc[0]['signals'][0]))
# print("First signal vector:", df.iloc[0]['signals'][0])
# print("Second signal vector:", df.iloc[0]['signals'][1])

# %%

#haar
# df_grid = partition_to_df("grid_hgr", use_split_name = False, mode = "haar")
# df_er = partition_to_df("er_hgr", use_split_name=False, mode="haar")

# df_grid.to_csv("grid_haar.csv")
# df_er.to_csv("er_haar.csv")

# df_grid.to_pickle("grid_haar.pkl")
# df_er.to_pickle("er_haar.pkl")

# df_cluster_train = partition_to_df("cluster_new_hgr/train", use_split_name = False, mode = "haar")
# df_cluster_test = partition_to_df("cluster_new_hgr/test", use_split_name = False, mode = "haar")
# df_cluster_val = partition_to_df("cluster_new_hgr/val", use_split_name = False, mode = "haar")

# df_cluster_train.to_csv("cluster_train_haar.csv")
# df_cluster_test.to_csv("cluster_test_haar.csv")
# df_cluster_val.to_csv("cluster_val_haar.csv")

# df_cluster_train.to_pickle("cluster_train_haar.pkl")
# df_cluster_test.to_pickle("cluster_test_haar.pkl")
# df_cluster_val.to_pickle("cluster_val_haar.pkl")


#rec
df_cluster_train_rec = partition_to_df("cluster_new_hgr/train", use_split_name = False, mode = "rec")
df_cluster_test_rec = partition_to_df("cluster_new_hgr/test", use_split_name = False, mode = "rec")
df_cluster_val_rec = partition_to_df("cluster_new_hgr/val", use_split_name = False, mode = "rec")


df_cluster_train_rec.to_csv("cluster_train_rec1.csv")
df_cluster_test_rec.to_csv("cluster_test_rec1.csv")
df_cluster_val_rec.to_csv("cluster_val_rec1.csv")


# df_cluster_train_rec.to_pickle("cluster_train_rec.pkl")
# df_cluster_test_rec.to_pickle("cluster_test_rec.pkl")
# df_cluster_val_rec.to_pickle("cluster_val_rec.pkl")

# df_grid1 = partition_to_df("grid_hgr", use_split_name = False, mode = "rec")
# df_er1 = partition_to_df("er_hgr", use_split_name=False, mode="rec")

# df_grid1.to_csv("grid_rec.csv")
# df_er1.to_csv("er_rec.csv")

# df_grid1.to_pickle("grid_rec.pkl")
# df_er1.to_pickle("er_rec.pkl")


#df_cluster = partition_to_df("cluster_hgr", use_split_name=False)
#df_cluster.to_pickle("cluster_rec_full.pkl")

# df_zinc_train_rec = partition_to_df("zinc_hgr/train", use_split_name=True)
# df_zinc_test_rec = partition_to_df("zinc_hgr/test", use_split_name=True)
# df_zinc_val_rec = partition_to_df("zinc_hgr/val", use_split_name=True)

# df_zinc_test_rec.to_pickle("zinc_test_rec.pkl")
# df_zinc_train_rec.to_pickle("zinc_train_rec.pkl")
# df_zinc_val_rec.to_pickle("zinc_val_rec.pkl")

# df_zinc_test_rec.to_csv("zinc_test_rec.csv")
# df_zinc_train_rec.to_csv("zinc_train_rec.csv")
# df_zinc_val_rec.to_csv("zinc_val_rec.csv")


# %%

# df_er_pkl = pd.read_pickle("er_rec_full.pkl")
# df_grid_pkl  = pd.read_pickle("grid_rec_full.pkl")

# #df_cluster_pkl = pd.read_pickle("cluster_rec_full.pkl")

# # df_zvalpk1= pd.read_pickle("zinc_val_part_rec_full.pkl")
# # df_zvalpk2= pd.read_pickle("zinc_test_part_rec_full.pkl")
# # df_zvalpk3= pd.read_pickle("zinc_train_part_rec_full.pkl")


# # %%
# df_er_pkl.to_csv("er_rec_full.csv", index= False, encoding= 'us-ascii')
# df_grid_pkl.to_csv("grid_rec_full.csv", index= False, encoding= 'us-ascii')



#df_cluster_pkl.to_csv("cluster_rec_full.csv", index=False, encoding= 'us-ascii')

# df_zvalpk1.to_csv("zinc_val_rec_full.csv", index=False, encoding='us-ascii')
# df_zvalpk2.to_csv("zinc_test_rec_full.csv", index=False, encoding='us-ascii') 
# df_zvalpk3.to_csv("zinc_train_rec_full.csv", index=False, encoding='us-ascii') 

# %%
#df = pd.read_csv("cluster_rec_full.csv")



#print(df.head())
# %%
#print(df.iloc[[0]])