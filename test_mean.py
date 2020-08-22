import numpy as np
from clustering.kmesar import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    points = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1], [-0.5, -1], [-0.5, 1], [-1, -0.5], [1, -0.5]])
    centroids = np.array([[-1.5, 1.5], [-2, 2], [-2, 1.5]])

    labels = assign_points_to_centroids(points, centroids)
    print(labels)

    fig = plt.figure()

    fig.add_subplot(2, 1, 1)
    plt.scatter(points[:, 0], points[:, 1])
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=30, marker='x')

    new_centroids, new_labels, _ = empty_clusters_resolution(points, centroids, labels, ecr_method='max')
    print(new_labels)

    fig.add_subplot(2, 1, 2)
    plt.scatter(points[:, 0], points[:, 1])
    plt.scatter(new_centroids[:, 0], new_centroids[:, 1], c='black', s=30, marker='x')

    plt.show()

