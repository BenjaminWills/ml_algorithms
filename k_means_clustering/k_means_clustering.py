import sys

sys.path.insert(0, "../")

from utility.logger import make_logger

logger = make_logger(
    logging_path="./k_means_logs.log", save_logs=True, logger_name="k_means"
)

import pandas as pd
import numpy as np

from collections import defaultdict
from tqdm import tqdm
from typing import List, Dict

tqdm.pandas()

Centroid = np.array
Centroid_index = int


class K_means:
    def __init__(
        self,
        num_clusters: int,
        data: pd.DataFrame,
        max_iterations: int = 2_000,
    ) -> None:
        self.num_clusters = num_clusters
        self.max_iterations = max_iterations

        # Initialise data and add nearest cluster columns
        self.data = data
        self.columns = data.columns
        self.data["nearest_centroid"] = 0

        # Dimension of the vector that is a row, minus 1 due to the fact
        # that we added a nearest_centroid column
        self.dimension: int = data.shape[1] - 1

        # Initialise with a random list of centroids
        self.centroids: List[Centroid] = self.generate_centroids()

    def generate_random_vector(
        self, N: int, min_value: float, max_value: float
    ) -> np.ndarray:
        """
        Generate a random vector of N dimensions with values between min_value and max_value.

        Parameters:
            N (int): Number of dimensions in the vector.
            min_value (float): Minimum value for the random numbers.
            max_value (float): Maximum value for the random numbers.

        Returns:
            np.ndarray: Random vector of N dimensions.
        """
        if N <= 0:
            logger.error("N must be a positive integer.")
            raise ValueError("N must be a positive integer.")
        if min_value >= max_value:
            logger.error("min_value must be less than max_value.")
            raise ValueError("min_value must be less than max_value.")

        random_vector = np.random.uniform(min_value, max_value, size=N)
        return random_vector

    def generate_centroids(self) -> List[Centroid]:
        """
        Generate random centroids for clustering.

        Returns:
            List[Centroid]: List of randomly generated centroids.
        """
        logger.info(f"Generating {self.num_clusters} random centroids")
        centroids = []
        for _ in tqdm(range(self.num_clusters)):
            centroid = self.generate_random_vector(self.dimension, -10_000, 10_000)
            centroids.append(centroid)

        return centroids

    def calculate_closest_point(self, row: pd.Series) -> Centroid:
        """
        Calculate the index of the closest centroid to a given data point.

        Parameters:
            row (pd.Series): Data point as a pandas Series.

        Returns:
            Centroid: Index of the closest centroid.
        """

        data_point = np.array(row[self.columns].to_list())

        distances = []

        for centroid in self.centroids:
            distance = np.linalg.norm(centroid - data_point)
            distances.append(distance)

        distances = np.array(distances)
        closest_centroid_index = distances.argmin()
        return closest_centroid_index

    def find_nearest_centroids(self):
        """
        Assign each data point to the nearest centroid and update the data frame.
        """
        logger.info("Finding nearest data points to the centroid")
        self.data["nearest_centroid"] = self.data.progress_apply(
            self.calculate_closest_point, axis=1
        )
        return self.data

    def group_centroids(self) -> Dict[int, List[np.array]]:
        """
        Group data points around centroids and calculate new centroid positions.

        Returns:
            Dict[int, List[np.array]]: Dictionary mapping cluster IDs to new centroid positions.
        """
        # Get the lists of nodes closest to each centriod
        centroid_data = list(self.data.groupby("nearest_centroid"))

        # Save these to a dictionary
        logger.info("Mapping data points to their nearest centroid")
        master = defaultdict(lambda: [])
        for centroid_id, data in tqdm(centroid_data):
            for index, row in data.iterrows():
                new_row = np.array(row.to_list())
                master[centroid_id].append(new_row)
        return master

    def calculate_new_centroids(
        self, grouped_data: Dict[Centroid_index, List[np.array]]
    ) -> List[Centroid]:
        # Find the mean distance between all datapoints that surround
        # a centroid
        logger.info("Calculating mean distances around each centroid")
        master = dict(grouped_data)
        new_clusters = {}
        for cluster_id in tqdm(range(self.num_clusters)):
            cluster = master.get(cluster_id, [])
            if cluster:
                mean = sum(cluster) / len(cluster)
                new_clusters[cluster_id] = mean[: self.dimension]
            else:
                new_clusters[cluster_id] = self.generate_random_vector(
                    self.dimension, -10_000, 10_000
                )

        return list(new_clusters.values())

    def centroids_close(
        self, original_centroids: List[Centroid], tolerance: float
    ) -> bool:
        equal = True
        for index in range(len(original_centroids)):
            original_centroid = original_centroids[index]
            new_centroid = self.centroids[index]
            close = np.linalg.norm(original_centroid - new_centroid) < tolerance
            if not close:
                return False
        return True

    def main_loop(self):
        iteration_count = 0
        for iteration in tqdm(range(self.max_iterations)):
            original_centriods = self.centroids
            self.find_nearest_centroids()
            grouped_data = self.group_centroids()
            self.centroids = self.calculate_new_centroids(grouped_data)
            iteration_count += 1

            if self.centroids_close(original_centriods, tolerance=5):
                break
        logger.info(f"Algorithm terminated after {iteration_count} iterastions.")
        return self.group_centroids()


if __name__ == "__main__":
    data = pd.read_csv("encoded_customer.csv")
    k_means = K_means(6, data=data, max_iterations=5)
    # print(k_means.find_nearest_centroids().nearest_centroid.value_counts())
    k_means.main_loop()
