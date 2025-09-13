
import pickle
import os
import torch
from torch_geometric.data import Data
from hgr_converter import write_hgr
import zipfile

import os
import torch
import pandas as pd
import ast
from hgr_converter import write_hgr

def convert_csv_to_hgr(input_csv, output_dir, exist_ok=True):
    """
    Converts graphs from a CSV file to HGR files in the specified output directory.
    Each row must have 'edge_index' and 'num_nodes' columns.
    """
    os.makedirs(output_dir, exist_ok=exist_ok)
    df = pd.read_csv(input_csv)

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

        out_path = os.path.join(output_dir, f"graph_{i}.hgr")
        write_hgr(data, out_path)


    print(f"Converted {len(df)} graphs to {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert CSV graphs to HGR files.")
    parser.add_argument("input_csv", type=str, help="Path to input CSV file.")
    parser.add_argument("output_dir", type=str, help="Directory to save HGR files.")
    parser.add_argument("--exist_ok", action="store_true", help="Do not raise error if output_dir exists.")
    args = parser.parse_args()
    convert_csv_to_hgr(args.input_csv, args.output_dir, exist_ok=args.exist_ok)

