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
        relative_tolerance: float = 0.05,
    ) -> None:
        """
        Initialize a K-means clustering algorithm.

        Parameters
        ----------
        num_clusters : int
            Number of clusters to form.
        data : pd.DataFrame
            Input data for clustering.
        max_iterations : int, optional
            Maximum number of iterations. Default is 2000.
        relative_tolerance : float, optional
            Relative tolerance for centroid convergence. Default is 0.05.
        """
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

        # Run main loop on data
        self.centroids = self.main_loop(relative_tolerance)

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
        for _ in range(self.num_clusters):
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

        self.data["nearest_centroid"] = self.data.apply(
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

        master = defaultdict(lambda: [])
        for centroid_id, data in centroid_data:
            for index, row in data.iterrows():
                new_row = np.array(row.to_list())
                master[centroid_id].append(new_row)
        return master

    def calculate_new_centroids(
        self, grouped_data: Dict[Centroid_index, List[np.array]]
    ) -> List[Centroid]:
        """
        Calculate new centroids based on grouped data points.

        Parameters
        ----------
        grouped_data : Dict[Centroid_index, List[np.array]]
            Grouped data points around centroids.

        Returns
        -------
        List[Centroid]
            List of newly calculated centroids.
        """
        # Find the mean distance between all datapoints that surround
        # a centroid

        master = dict(grouped_data)
        new_clusters = {}
        for cluster_id in range(self.num_clusters):
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
        self, original_centroids: List[Centroid], relative_tolerance: float
    ) -> bool:
        """
        Check if centroids have converged within the specified relative tolerance.

        Parameters
        ----------
        original_centroids : List[Centroid]
            Original centroids.
        relative_tolerance : float
            Relative tolerance for centroid convergence.

        Returns
        -------
        bool
            True if centroids have converged, False otherwise.
        """
        for index in range(len(original_centroids)):
            original_centroid = original_centroids[index]
            new_centroid = self.centroids[index]

            absolute_difference = np.linalg.norm(new_centroid) - np.linalg.norm(
                original_centroid
            )
            relative_difference = absolute_difference / np.linalg.norm(
                original_centroid
            )
            close = abs(relative_difference) <= relative_tolerance

            if not close:
                return False
        return True

    def main_loop(self, relative_tolerance):
        """
        Main loop for K-means clustering algorithm.

        Parameters
        ----------
        relative_tolerance : float
            Relative tolerance for centroid convergence.
        """
        iteration_count = 0
        logger.info(
            f"""
Begin training algorithm with:
Number of clusters : {self.num_clusters}
Max iterations     : {self.max_iterations}"""
        )
        for iteration in tqdm(range(self.max_iterations)):
            original_centriods = self.centroids
            self.find_nearest_centroids()
            grouped_data = self.group_centroids()
            self.centroids = self.calculate_new_centroids(grouped_data)
            iteration_count += 1

            if self.centroids_close(
                original_centriods, relative_tolerance=relative_tolerance
            ):
                break
        logger.info(f"Algorithm terminated after {iteration_count} iterations.")
        return self.group_centroids()


if __name__ == "__main__":
    data = pd.read_csv("encoded_customer.csv")
    k_means = K_means(6, data=data, max_iterations=5)
