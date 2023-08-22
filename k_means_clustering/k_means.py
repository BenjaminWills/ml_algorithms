import argparse

import pandas as pd

from k_means_clustering import K_means
from eda_functions import encode_string_columns

parser = argparse.ArgumentParser(description="K means command line function")
parser.add_argument(
    "--data_path",
    type=str,
    default="",
    help="Path to data that we wish to find clusters within.",
)
parser.add_argument(
    "--relative_tolerance",
    type=float,
    default=0.05,
    help="This is the allowed relative tollerance between the prior cluster and the next cluster",
)
parser.add_argument(
    "--max_iterations",
    type=int,
    default=10,
    help="This is the maximum number of iterations that the program is allowed to run",
)
parser.add_argument(
    "--num_clusters",
    type=int,
    default=10,
    help="This is the number of clusters that we can split our data into",
)
parser.add_argument(
    "--encode_data",
    type=str,
    default="no",
    help="Will encode all non numeric columns of the data sequentially. either yes or no",
)

if __name__ == "__main__":
    # Save all vars
    args = parser.parse_args()
    relative_tolerance = args.relative_tolerance
    max_iterations = args.max_iterations
    data_path = args.data_path
    num_clusters = args.num_clusters
    encode_data = args.encode_data

    # Load in the data
    data = pd.read_csv(data_path)

    # if encode_data = 'yes' then encode the data.
    if encode_data.lower() == "yes":
        data = encode_string_columns(data)

    # Run the algorithm
    k_means = K_means(
        num_clusters=num_clusters,
        data=data,
        max_iterations=max_iterations,
        relative_tolerance=relative_tolerance,
    )
