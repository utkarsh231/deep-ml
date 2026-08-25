import numpy as np

def k_means_clustering(points, k, initial_centroids, max_iterations):

    def dist(arr1, arr2):
        total = 0

        for i in range(len(arr1)):
            total += (arr1[i] - arr2[i]) ** 2

        return total ** 0.5

    centroids = initial_centroids

    for _ in range(max_iterations):

        # 1. Create empty clusters
        clusters = [[] for _ in range(k)]

        # 2. Assign each point to nearest centroid
        for point in points:

            min_dist = float("inf")
            closest = 0

            for i in range(k):
                d = dist(point, centroids[i])

                if d < min_dist:
                    min_dist = d
                    closest = i

            clusters[closest].append(point)

        # 3. Calculate new centroids
        new_centroids = []

        for cluster in clusters:
            if len(cluster) > 0:
                new_centroid = tuple(np.mean(cluster, axis=0))
                new_centroids.append(new_centroid)

        centroids = new_centroids

    # 4. Round final answer
    final_centroids = [
        tuple(round(x, 4) for x in centroid)
        for centroid in centroids
    ]

    return final_centroids