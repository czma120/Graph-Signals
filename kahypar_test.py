# %%
import os
import kahypar as kahypar
import math
# %%

num_nodes = 7
num_nets = 4

hyperedge_indices = [0,2,6,9,12]
hyperedges = [0,2,0,1,3,4,3,4,6,2,5,6]

node_weights = [1,2,3,4,5,6,7]
edge_weights = [11,22,33,44]

k=4

hypergraph = kahypar.Hypergraph(num_nodes, num_nets, hyperedge_indices, hyperedges, k, edge_weights, node_weights)
# %%

context = kahypar.Context()
context.loadINIconfiguration("config/km1_rKaHyPar_sea20.ini")
# %%
context.setK(k)
context.setEpsilon(0.03)

kahypar.partition(hypergraph, context)


# %%

#printing which nodes are in which blocjs
for i in range(num_nodes):
    print(f"Node {i} is in block {hypergraph.blockID(i)}")

#parition vector that has the blockID for each node (which bloock it belongs to)
partition_vector = [hypergraph.blockID(i) for i in range(num_nodes)]


#converting the parition vector to a signal
# signal = [1 if part == 0 else -1 for part in partition_vector]

# print("Signal:", signal)


def extract_bitwise_signals(partition_vector, num_levels):
    signals = []
    for level in range(num_levels):
        signal = []
        for region in partition_vector:
            if (region >> level) & 1 == 0:
                signal.append(+1)
            else:
                signal.append(-1)
        signals.append(signal)
    return signals

num_levels = math.floor(math.log2(k))  # = 2 when k = 4

signals = extract_bitwise_signals(partition_vector, num_levels= num_levels)
for i, signal in enumerate(signals):
    print(f"Harmonic {i}: {signal}")
# %%
