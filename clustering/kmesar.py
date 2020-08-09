import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import pandas as pd


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
    centroids = np.zeros(shape=(k_centroids, n))

    if lower_bound is None or upper_bound is None:
        for k in range(k_centroids):
            centroids[k] = np.random.random_sample((n,))
    elif lower_bound is not None and upper_bound is not None:
        for k in range(k_centroids):
            centroids[k] = lower_bound + np.random.random_sample((n,)) * (upper_bound - lower_bound)

    # If there is a centroid with none points assigned to it, reinitialize one of centroids
    n_labels = len(np.unique(assign_points_to_centroids(points, centroids)))

    while n_labels < k_centroids:
        rand_index = np.random.randint(0, k_centroids)

        if lower_bound is None or upper_bound is None:
            centroids[rand_index] = np.random.random_sample((n,))
        elif lower_bound is not None and upper_bound is not None:
            centroids[rand_index] = lower_bound + np.random.random_sample((n,)) * (upper_bound - lower_bound)

        n_labels = len(np.unique(assign_points_to_centroids(points, centroids)))

    return centroids


def check_centroids_update(old_centroids, new_centroids, tol, norm='euclidean'):
    """
    :param old_centroids: Cluster centroids from previous iteration: numpy array of shape (k, n)
    :param new_centroids: Cluster centroids from current iteration: numpy array of shape (k, n)
    :param tol: Tolerance parameter: real number
    :return: K-Means stopping criterion: boolean
    """

    if norm == 'euclidean':
        diff = la.norm(old_centroids - new_centroids, ord=2)
    elif norm == 'frob':
        diff = la.norm(old_centroids - new_centroids, ord='frob')
    else:
        raise ValueError(f'Unknown norm type: {norm}')

    stoppping_criterion_reached = (diff <= tol).all()

    return stoppping_criterion_reached


class KMeansStandard:
    def __init__(self, k_clusters=5, init='random', n_init=10, max_iter=300, tol=1e-4):
        self.k_clusters = k_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol

        self.labels_ = None
        self.centroids_ = None
        self.intertia_ = None
        self.n_iter_ = None
        self.history_ = None

    def fit(self, points):
        lower_bound = np.min(points, axis=0)
        upper_bound = np.max(points, axis=0)
        history = {
            'labels': [],
            'centroids': [],
            'inertia': [],
            'n_iter': []
        }

        if type(points) == pd.DataFrame:
            points = np.array(points)

        for n_it in range(self.n_init):
            initial_centroids = initialize_centroids_random(points, self.k_clusters, lower_bound, upper_bound)
            centroids = initial_centroids

            for it in range(self.max_iter):
                labels = assign_points_to_centroids(points, centroids)
                new_centroids = update_centroids(points, centroids, labels)
                stopping_criterion_reached = check_centroids_update(centroids, new_centroids, self.tol)
                centroids = new_centroids

                if stopping_criterion_reached:
                    break

            history['labels'].append(labels)
            history['centroids'].append(centroids)
            history['inertia'].append(sum_of_squared_error(points, centroids, labels))
            history['n_iter'].append(it)

        best_result_index = np.argmin(history['inertia'])
        self.history_ = history
        self.labels_ = history['labels'][best_result_index]
        self.centroids_ = history['centroids'][best_result_index]
        self.inertia_ = history['inertia'][best_result_index]
        self.n_iter_ = history['n_iter'][best_result_index]

