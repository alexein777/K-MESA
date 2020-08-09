import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt


def euclidean_distance(vec1, vec2):
    """
    :param vec1: n-dimensional vector of real values: numpy array of shape (n, )
    :param vec2: n-dimensional vector of real values: numpy array of shape (n, )
    :return: Eucledian distance between vectors vec1 and vec2: real number
    """

    return la.norm(vec1 - vec2, ord=2)


def assign_points_to_centroids(points, centroids):
    """
    :param points: Points from training set to be clustered: numpy array of shape (m, n)
    :param centroids: Points representing cluster centroids: numpy array of shape (k, n)
    :return: Array of centroid indices (i.e cluster labels) with respect to point indices:
    numpy array of shape (m, )
    """

    m = points.shape[0]
    labels = np.zeros(shape=(m, ))

    for i, point in enumerate(points):
        distances = la.norm(point - centroids, ord=2, axis=1)
        centroid_index = np.argmin(distances)
        labels[i] = centroid_index

    return labels


def extract_labeled_points(points, labels, k_label):
    """
    :param points: Points from training set to be clustered: numpy array of shape (m, n)
    :param labels: Array of centroid indices (i.e cluster labels) with respect to point indices:
    numpy array of shape (m, )
    :param k_label: Specified cluster label (centroid index) for which we want to extract points: integer
    :return: Points labeled as k_label: numpy array of shape (None, n)
    """

    indices = np.where(labels == k_label)

    return points[indices]


def update_centroids(points, centroids, labels):
    """
    :param points: Points from training set to be clustered: numpy array of shape (m, n)
    :param centroids: Points representing (current) cluster centroids: numpy array pf shape (k, n)
    :param labels: Array of (current) centroid indices (i.e cluster labels) with respect to point indices:
    numpy array of shape (m, )
    :return: Updated centroids: numpy array of shape (k, n)
    """

    k_cluster_labels = centroids.shape[0]
    new_centroids = np.zeros(centroids.shape)

    for k_label in range(k_cluster_labels):
        points_with_k_label = extract_labeled_points(points, labels, k_label)  # extracting points with k label
        new_centroid = np.mean(points_with_k_label, axis=0)  # mean of points in cluster k
        new_centroids[k_label] = new_centroid  # updating current centroid with a new value

    return new_centroids


def sum_of_squared_error(points, centroids, labels):
    """
    :param points:  Points from training set to be clustered: numpy array of shape (m, n)
    :param centroids: Points representing cluster centroids: numpy array pf shape (k, n)
    :return: Sum of squared error between cluster centroids and points assigned to those centroids: real number
    """

    k_cluster_labels = centroids.shape[0]
    sse = 0

    for k_label in range(k_cluster_labels):
        points_with_label_k = extract_labeled_points(points, labels, k_label)
        sse += np.sum(np.power(points_with_label_k - centroids[k_label], 2))

    return sse


def initialize_centroids_random(points, k_centroids, lower_bound=None, upper_bound=None):
    """
    :param points: Points from training set to be clustered: numpy array of shape (m, n)
    :param k_centroids: Number of centroids to be initialized: integer
    :param lower_bound: Vector of minimum values across points dimensions: numpy array of shape (1, n)
    :param upper_bound: Vector of maximum values across points dimensions: numpy array of shape (1, n)
    :return: Initial centroids: numpy array of shape (k, n)
    """

    n = points.shape[1]
    centroids = np.zeros((k_centroids, n))

    if lower_bound is None or upper_bound is None:
        for k in range(k_centroids):
            centroids[k] = np.random.random_sample((n,))
    elif lower_bound is not None and upper_bound is not None:
        for k in range(k_centroids):
            centroids[k] = lower_bound + np.random.random_sample((n,)) * (upper_bound - lower_bound)

    return centroids

