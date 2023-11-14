import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

PATH = os.path.abspath(os.path.dirname(__file__))

N_CLUSTERS = 8


def k_means_plus_plus(X, n_clusters=N_CLUSTERS):
    # Choose the first centroid randomly
    centroids = np.array([X.sample().values[0]])

    # Choose the other centroids
    for _ in range(1, n_clusters):
        distances = np.array(
            [np.min(np.linalg.norm(centroids - x, axis=1)) for x in X.values]
        )
        probabilities = distances / np.sum(distances)
        centroids = np.append(
            centroids, [X.sample(weights=probabilities).values[0]], axis=0
        )

    return centroids


def main():
    input = pd.read_csv(PATH + "/data/data.csv", header=None)

    # ex.1:
    df = input.iloc[1:, 1:].astype(float)  # remove the first row and column

    kmeans = KMeans(init="random", n_clusters=8).fit(df)

    SSE = kmeans.inertia_

    print("Initial SSE: ", SSE)

    # ex.2:
    n_init_array = np.logspace(1, 2, num=4, base=10, dtype=int)
    max_iter_array = np.logspace(1, 3, num=6, base=10, dtype=int)

    optimal_SSE = np.inf
    optimal_ni = 0
    optimal_mi = 0

    for init in ["k-means++", "random"]:
        for ni in n_init_array:
            for mi in max_iter_array:
                print(ni, mi)

                improved_kmeans = KMeans(
                    init=init,
                    n_clusters=N_CLUSTERS,
                    n_init=ni,
                    max_iter=mi,
                ).fit(df)

                new_SSE = improved_kmeans.inertia_

                if new_SSE < optimal_SSE:
                    optimal_SSE = new_SSE
                    optimal_ni = ni
                    optimal_mi = mi
                    optimal_init = init
                    optimal_kmeans = improved_kmeans

    print(
        pd.DataFrame(
            data=[optimal_SSE, optimal_ni, optimal_mi, optimal_init],
            index=["SSE", "n_init", "max_iter", "init"],
        )
    )

    plt.figure()
    plt.scatter(
        df.mean(axis=1),
        np.zeros(df.shape[0]),
        c=optimal_kmeans.labels_,
        cmap="rainbow",
    )
    plt.scatter(
        optimal_kmeans.cluster_centers_.mean(axis=1),
        np.zeros(optimal_kmeans.cluster_centers_.shape[0]),
        marker="x",
        c="black",
    )
    plt.title("Clusters with centroids")
    plt.show()

    # ex.3:
    stock_label = list(zip(input.iloc[1:, 0], optimal_kmeans.labels_))

    for i in range(N_CLUSTERS):
        print("\nCluster: ", i)

        for stock, label in stock_label:
            if label == i:
                print("\t", stock)

    # ex.4: custom implementation of the k-means++ algorithm
    custom_kmeans = KMeans(
        init=k_means_plus_plus(df),
        n_clusters=N_CLUSTERS,
        n_init=1,
        max_iter=optimal_mi,
    ).fit(df)

    print("Custom K-Means SSE: ", custom_kmeans.inertia_)

    plt.figure()
    plt.scatter(
        df.mean(axis=1),
        np.zeros(df.shape[0]),
        c=custom_kmeans.labels_,
        cmap="rainbow",
    )
    plt.scatter(
        optimal_kmeans.cluster_centers_.mean(axis=1),
        np.zeros(optimal_kmeans.cluster_centers_.shape[0]),
        marker="v",
        c="black",
    )
    plt.scatter(
        optimal_kmeans.cluster_centers_.mean(axis=1),
        np.zeros(custom_kmeans.cluster_centers_.shape[0]),
        marker="^",
        c="red",
    )
    plt.title("Clusters with centroids")
    plt.show()

    print(
        f"Difference between centroids (optimal - custom K-means): {np.linalg.norm(np.sort(optimal_kmeans.cluster_centers_) - np.sort(custom_kmeans.cluster_centers_))}"
    )


if __name__ == "__main__":
    main()
