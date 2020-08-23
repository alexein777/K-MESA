import numpy as np
import numpy.linalg as la
import pandas as pd
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from clustering.utils import time_elapsed


def euclidean_distance(vec1, vec2):
    """
    :param vec1: n-dimensional vector of real values: numpy array of shape (n, )
    :param vec2: n-dimensional vector of real values: numpy array of shape (n, )
    :return: Eucledian distance between vectors vec1 and vec2: real number
    """

    return la.norm(vec1 - vec2, ord=2)


def sse_single(points_j, centroid_j):
    """
    :param points_j: Points with label_j: numpy array of shape (?, n)
    :param centroid_j: Centroid that represents cluster j: numpy aray of shape (n, )
    :return: Sum of squared error in a single cluster
    """

    return np.sum(la.norm(centroid_j - points_j, ord=2, axis=1))


def sum_of_squared_error(points, centroids, labels):
    """
    :param points:  Points from training set to be clustered: numpy array of shape (m, n)
    :param centroids: Points representing cluster centroids: numpy array of shape (k, n)
    :param labels: Current centroid indices (i.e cluster labels) with respect to point indices:
    numpy array of shape (m, )
    :return: Sum of squared error between cluster centroids and points assigned to those centroids: real number
    """

    k_labels = centroids.shape[0]
    sse = 0

    for label_j in range(k_labels):
        points_with_label_j = extract_labeled_points(points, labels, label_j)
        sse += sse_single(points_with_label_j, centroids[label_j])

    return sse


def get_max_distances(points, centroids):
    """
    For every point in dataset, calculate maximum distance to centroids.
    :param points: Points from dataset to be clustered: numpy array of shape (m, n)
    :param centroids: Points representing cluster centroids: numpy array of shape (k, n)
    :return: Array of maximum distances from every point to centroids: numpy array of shape (m, n)
    """

    m = points.shape[0]
    max_distances = np.zeros(shape=(m,))

    for i, point in enumerate(points):
        max_distances[i] = np.max(la.norm(point - centroids, ord=2, axis=1))

    return max_distances


def get_furthest_point(points, centroids):
    """
    For every point in dataset, calculate distance to every centroid and sum distances.
    Get point with maximum sum of distances.
    :param points: Points from dataset to be clustered: numpy array of shape (m, n)
    :param centroids: Points representing cluster centroids: numpy array pf shape (k, n)
    :return: Point furthest from current centroids: numpy array of shape (n, )
    """

    m = points.shape[0]
    distances = np.zeros(shape=(m,))

    for i, point in enumerate(points):
        point_distances = la.norm(point - centroids, ord=2, axis=1)
        distances[i] = np.sum(point_distances)

    max_index = np.argmax(distances)

    return points[max_index]


def get_sses(points, centroids, labels):
    """
    Calculate sse for every cluster seperately.
    :param points: Points from dataset to be clustered: numpy array of shape (m, n)
    :param centroids: Points representing cluster centroids: numpy array pf shape (k, n)
    :param labels: Array of centroid indices (i.e cluster labels) with respect to point indices:
    numpy array of shape (m, )
    :return: List of SSE's with indices corresponding to centroid indices (i.e cluster labels)
    """

    k = centroids.shape[0]
    sses = np.zeros((k,))

    for j, centroid_j in enumerate(centroids):
        points_with_label_j = extract_labeled_points(points, labels, j)
        sses[j] = sse_single(points_with_label_j, centroid_j)

    return sses


def assign_points_to_centroids(points, centroids):
    """
    :param points: Points from training set to be clustered: numpy array of shape (m, n)
    :param centroids: Points representing cluster centroids: numpy array of shape (k, n)
    :return: Array of centroid indices (i.e cluster labels) with respect to point indices:
    numpy array of shape (m, )
    """

    m = points.shape[0]
    labels = np.zeros(shape=(m,))

    for i, point in enumerate(points):
        distances = la.norm(point - centroids, ord=2, axis=1)
        centroid_index = np.argmin(distances)
        labels[i] = centroid_index

    return labels


def empty_clusters_resolution(points, centroids, labels, ecr_method='random', bounds=None):
    """
    :param points: Points from training set to be clustered: numpy array of shape (m, n)
    :param centroids: Points representing cluster centroids: numpy array of shape (k, n)
    :param labels: Array of centroid indices (i.e cluster labels) with respect to point indices:
    numpy array of shape (m, )
    :param ecr_method: Method that determines how the empty clusters are resolved: string
    Possible values: 'random', 'max', 'sse'
    :param bounds: Lower and upper bounds for points from dataset: tupple
    :return: Updated centroids and labels, where every centroid covers at least one point from dataset; also number
    of empty cluster resolutions (ECR)
    """

    unique_labels = np.unique(labels)
    k_labels = centroids.shape[0]

    # Quick check if all labels are present
    if unique_labels.shape[0] == k_labels:
        return centroids, labels, 0, np.array([])

    # Prepare neccessary variables
    if bounds is not None:
        lower_bound, upper_bound = bounds
    else:
        lower_bound = np.min(points, axis=0)
        upper_bound = np.max(points, axis=0)

    n = points.shape[1]
    n_ecr = 0

    new_centroids = copy.deepcopy(centroids)
    new_labels = copy.deepcopy(labels)
    all_labels = np.array(range(k_labels))
    all_labels_present = False
    missing_labels = np.setdiff1d(all_labels, unique_labels)
    ecr_indices = set([])

    while not all_labels_present:
        for missing_label in missing_labels:
            # Update new_centroids[missing_label] to cover at least one point from dataset
            if ecr_method == 'random':
                new_centroids[missing_label] = lower_bound + np.random.random_sample((n,)) * (upper_bound - lower_bound)
            elif ecr_method == 'min':
                # distances = la.norm(new_centroids[missing_label] - points, ord=2, axis=1)
                # min_index = np.argmin(distances)
                # new_centroids[missing_label] = points[min_index]
                raise NotImplemented('"min" ECR is not working properly. Use "random" instead.')
            elif ecr_method == 'max':
                # furthest_point = get_furthest_point(points, new_centroids)
                # new_centroids[missing_label] = furthest_point
                raise NotImplemented('"max" ECR is not working properly. Use "random" instead.')
            elif ecr_method == 'sse':
                # # Get SSE for every cluster seperately
                # sses = get_sses(points, new_centroids, new_labels)
                #
                # # Extract cluster with maximum SSE
                # max_label = np.argmax(sses)
                # points_with_max_label = extract_labeled_points(points, new_labels, max_label)
                #
                # # Get random point from that cluster and assign centroid to be that point
                # rand_index = np.random.randint(0, points_with_max_label.shape[0])
                # new_centroids[missing_label] = points_with_max_label[rand_index]
                raise NotImplemented('"sse" ECR is not working properly. Use "random" instead.')
            else:
                raise ValueError(f'Unknown ECR method: {ecr_method}')

            n_ecr += 1
            ecr_indices.add(missing_label)

        new_labels = assign_points_to_centroids(points, new_centroids)
        unique_labels = np.unique(new_labels)
        missing_labels = np.setdiff1d(all_labels, unique_labels)

        if missing_labels.shape[0] == 0:
            all_labels_present = True

    return new_centroids, new_labels, n_ecr, np.array(list(ecr_indices))


def extract_labeled_points(points, labels, label_j):
    """
    :param points: Points from training set to be clustered: numpy array of shape (m, n)
    :param labels: Array of centroid indices (i.e cluster labels) with respect to point indices:
    numpy array of shape (m, )
    :param label_j: Specified cluster label (centroid index) for which we want to extract points: integer
    :return: Points labeled as label_j: numpy array of shape (None, n)
    """

    indices = np.where(labels == label_j)

    return points[indices]


def update_centroids(points, centroids, labels):
    """
    :param points: Points from training set to be clustered: numpy array of shape (m, n)
    :param centroids: Points representing (current) cluster centroids: numpy array pf shape (k, n)
    :param labels: Current centroid indices (i.e cluster labels) with respect to point indices:
    numpy array of shape (m, )
    Possible values: 'nearest_point' - centroid's nearest point is assigned to it; 'random' - centroid is reinitialized
    :return: Updated centroids: numpy array of shape (k, n)
    """

    k_labels = centroids.shape[0]
    new_centroids = np.zeros(centroids.shape)

    for label_j in range(k_labels):
        points_with_label_j = extract_labeled_points(points, labels, label_j)
        # Important: call empty_clusters_resolution before calculating mean!
        new_centroid = np.mean(points_with_label_j, axis=0)
        new_centroids[label_j] = new_centroid

    return new_centroids


def initialize_centroids_random(points, k_centroids, bounds=None):
    """
    This function may produce empty clusters. It's neccessary to call empty_clusters_resolution after this function.
    :param points: Points from training set to be clustered: numpy array of shape (m, n)
    :param k_centroids: Number of centroids to be initialized: integer
    :param bounds: Tupple of lower and upper bounds of training set. If None, they are calculated.
    :return: Initial centroids: numpy array of shape (k, n)
    """

    if bounds is not None:
        lower_bound, upper_bound = bounds
    else:
        lower_bound = np.min(points, axis=0)
        upper_bound = np.max(points, axis=0)

    n = points.shape[1]
    centroids = np.zeros(shape=(k_centroids, n))

    for k in range(k_centroids):
        centroids[k] = lower_bound + np.random.random_sample((n,)) * (upper_bound - lower_bound)

    labels = assign_points_to_centroids(points, centroids)
    unique_labels = np.unique(labels)

    if unique_labels.shape[0] == k_centroids:
        return centroids

    return centroids


def initialize_centroids_advanced(points, k_centroids):
    """
    K-Means++ algorithm for centroid initialization. Centroids are initialized iteratively: first is random point
    from dataset, and every other is the furthest point from current centroids.
    :param points: Points from dataset to be clustered: numpy array of shape (m, n)
    :param k_centroids: Number of centroids to be initialized: integer
    :return: Initialized centroids: numpy array of shape (k_centroids, n)
    """

    if k_centroids <= 0:
        raise ValueError(f'Invalid number of centroids: {k_centroids} (expected k > 0)')

    m = points.shape[0]
    n = points.shape[1]
    centroids = np.array([]).reshape(0, n)

    # Initializing first centroid
    rand_index = np.random.randint(0, m)
    rand_point = points[rand_index]
    centroids = np.vstack((centroids, rand_point))

    # Initialize other centroids iteratively
    for i in range(1, k_centroids):
        furthest_point = get_furthest_point(points, centroids)
        centroids = np.vstack((centroids, furthest_point))

    return centroids


def check_centroids_update(old_centroids, new_centroids, tol, norm='euclidean'):
    """
    :param old_centroids: Cluster centroids from previous iteration: numpy array of shape (k, n)
    :param new_centroids: Cluster centroids from current iteration: numpy array of shape (k, n)
    :param tol: Tolerance parameter: real number
    :param norm: How distance is calculated between points: string
    Possible values: 'euclidean', 'frob'
    :return: K-Means stopping criterion: boolean
    """

    if norm == 'euclidean':
        diff = la.norm(old_centroids - new_centroids, ord=2)
    elif norm == 'frob':
        diff = la.norm(old_centroids - new_centroids, ord='frob')
    else:
        raise ValueError(f'Unknown norm type: {norm}')

    # If all centroid updates are less than tolerance, stopping criterion is reached
    stoppping_criterion_reached = (diff <= tol).all()

    return stoppping_criterion_reached


def annealing_probability(it, annealing_prob_function, alpha=1):
    """
    :param it: Current iteration of the algorithm: integer
    :param annealing_prob_function: Decreasing function between 0 and 1 representing annealing probabilty: string
    Possible values: 'exp', 'log', 'sq', 'sqrt', 'sigmoid', 'recip', 'flex', 'fixed'
    :param alpha: Tunning parameter for annealing probability function: real number
    :return: Probability of acceptance the neighbouring solution (e.g moving of centroid in the specified direction)
    """

    if alpha <= 0:
        raise ValueError(f'Incorrect value for function hyperparameter: {alpha} (expected value > 0)')

    if annealing_prob_function == 'exp':
        return np.exp(np.divide(-it + 1, alpha))  # +1 when it=1 => 0
    elif annealing_prob_function == 'log':
        return np.divide(np.log(1 + alpha), np.log(it + alpha))
    elif annealing_prob_function == 'sq':
        if type(it) == np.ndarray:
            ones = np.zeros((it.shape[0],)) + 1
            ret = np.min([np.divide(alpha + it, np.power(it, 2)), ones], axis=0)
        else:
            ret = np.min([np.divide(alpha + it, np.power(it, 2)), 1])

        return ret
    elif annealing_prob_function == 'sqrt':
        return np.divide(alpha, (np.sqrt(it - 1) + alpha))
    elif annealing_prob_function == 'sigmoid':
        return np.divide(1, 1 + np.divide(it - 1, alpha + np.exp(-it)))
    elif annealing_prob_function == 'recip':
        return np.divide(1 + alpha, it + alpha)
    elif annealing_prob_function == 'flex':
        return np.divide(1, it ** alpha)
    elif annealing_prob_function == 'fixed':
        if type(it) == np.ndarray:
            ret = np.zeros((it.shape[0],)) + alpha
        else:
            ret = alpha

        return ret
    else:
        raise ValueError(f'Unknown annealing probability function: {annealing_prob_function}')


def annealing_weight(it, annealing_weight_function, beta):
    """Alias for annealing_probability function"""

    return annealing_probability(it, annealing_weight_function, beta)


def get_random_points_from_clusters(points, labels):
    """
    Helper function for one of the annealing methods. Extracts a random point from each cluster.
    :param points: Points from training set to be clustered: numpy array of shape (m, n)
    :param labels: Centroid indices (i.e cluster labels) with respect to point indices: numpy array of shape (m, n)
    :return: Random points, each point as a representative from its cluster: numpy array of shape (k, n)
    """
    rand_points = []
    k_labels = np.unique(labels).shape[0]

    for label_j in range(k_labels):
        points_with_label_j = extract_labeled_points(points, labels, label_j)

        # Empty cluster anomaly: skip step
        if points_with_label_j.shape[0] == 0:
            continue

        rand_index = np.random.randint(0, points_with_label_j.shape[0])
        rand_point = points_with_label_j[rand_index]
        rand_points.append(rand_point)

    return np.array(rand_points)


def calculate_annealing_vector(points,
                               labels,
                               centroids,
                               label_j,
                               it,
                               bounds=None,
                               annealing_method='random',
                               annealing_weight_function='log',
                               beta=1.2
                               ):
    """
    :param points: Points from the training set to be clustered: numpy array of shape (m, n)
    Note: Some annealing methods will require entire training set to evaluate annealing vector
    :param labels: Current centroid indices (i.e cluster labels) with respect to point indices:
    numpy array of shape (m, )
    :param centroids: Cluster centroids: numpy array of shape (k, n)
    Note: Annealing vector is calculated only for one centroid, i.e centroids[label_j]. Entire array is neccessary
    from some annealing methods to calculate direction points.
    Note: Function allows passing a single centroid that has to be annealed: numpy array of shape (n, )
    :param label_j: Cluster label for cluster represented by centroid: integer
    Note: This parameter is neccessary for extracting the points assigned to this cluster
    :param it: Current iteration of the algorithm: integer
    :param bounds: Lower and upper bounds (min and max) of the training set (points). If None, they are calculated,
    else unpacked from a tuple.
    :param annealing_method: Specifies how the centroids are annealed (i.e moved from their current position): string
    Possible values: 'random', 'min', 'max', 'maxmin', 'cluster_own', 'cluster_other', 'cluster_mean', 'centroid_split',
    'centroid_gather'
    :param annealing_weight_function: Decreasing function between 0 and 1 that handles the intensity by which will
    annealing vector pull the centroid in the specified direction: string
    Possible values: 'exp', 'log', 'sq', 'sqrt', 'sigmoid', 'recip', 'fixed' - in this case function is ignored and only
    beta parameter is taken in account (if beta > 0, beta is clamped to 1)
    Example: if function returns w = 0.8, centroid will move towards directional point by 80% of the annealing vector
    :param beta: Tunning parameter for annealing vector calculation: real number
    :return: Annealing vector that handles the movement direction of a single centroid: numpy array of shape (1, n)
    """

    if beta <= 0:
        raise ValueError(f'Bad value for parameter beta: {beta} (expected beta > 0)')

    # Exact centroid to be annealed
    if centroids.ndim == 1:
        centroid = centroids
    else:
        centroid = centroids[label_j]

    # Carousel method chooses one of the possible methods randomly
    if annealing_method == 'carousel':
        methods = ['random', 'min', 'max', 'cluster_own', 'cluster_other', 'cluster_mean',
                   'centroid_split', 'centroid_gather']
        method = np.random.choice(methods)
    else:
        method = annealing_method

    if method == 'random':
        # Direction point is random point from n-dimensional space of the training set with given bounds
        if bounds is None:
            lower_bound = np.min(points, axis=0)
            upper_bound = np.max(points, axis=0)
        else:
            lower_bound, upper_bound = bounds

        direction_point = lower_bound + np.random.random(points[0].shape) * (upper_bound - lower_bound)
    elif method == 'min':
        # Direction point is point from cluster label_j with the lowest distance from current centroid
        points_with_label_j = extract_labeled_points(points, labels, label_j)

        # Anomaly: if there are no points in this cluster, ignore annealing step
        if points_with_label_j.shape[0] == 0:
            direction_point = centroid
        else:
            distances = la.norm(centroid - points_with_label_j, ord=2, axis=1)
            min_index = np.argmin(distances)
            direction_point = points_with_label_j[min_index]
    elif method == 'max':
        # Direction point is point from cluster label_j with the highest distance from current centroid
        points_with_label_j = extract_labeled_points(points, labels, label_j)

        # Anomaly: if there are no points in this cluster, ignore annealing step
        if points_with_label_j.shape[0] == 0:
            direction_point = centroid
        else:
            distances = la.norm(centroid - points_with_label_j, ord=2, axis=1)
            max_index = np.argmax(distances)
            direction_point = points_with_label_j[max_index]
    elif method == 'maxmin':
        # Direction point is point from cluster label_j with the lowest/highest distance from current centroid,
        # depending on parity of current iteration it
        points_with_label_j = extract_labeled_points(points, labels, label_j)

        # Anomaly: if there are no points in this cluster, ignore annealing step
        if points_with_label_j.shape[0] == 0:
            direction_point = centroid
        else:
            distances = la.norm(centroid - points_with_label_j, ord=2, axis=1)

            # On first iteration (it=1) max annealing is applied, and every other odd iteration
            if it % 2 != 0:
                index = np.argmax(distances)
            else:
                index = np.argmin(distances)

            direction_point = points_with_label_j[index]
    elif method == 'cluster_own':
        # Direction point is random point from centroid's own cluster
        points_with_label_j = extract_labeled_points(points, labels, label_j)

        # Anomaly: if there are no points in this cluster, ignore annealing step
        if points_with_label_j.shape[0] == 0:
            direction_point = centroid
        else:
            rand_index = np.random.randint(0, points_with_label_j.shape[0])
            direction_point = points_with_label_j[rand_index]
    elif method == 'cluster_other':
        # Direction point is random point from some other cluster different from centroid's corresponding points
        labels_without_j = labels[np.where(labels != label_j)]

        # Fixing anomaly case where there are no labels other than label_j
        if labels_without_j.shape[0] == 0:
            # Take point from any cluster
            rand_label = np.random.randint(0, np.unique(labels).shape[0])
        else:
            rand_label = np.squeeze(np.random.choice(labels_without_j, size=1))

        points_with_rand_label = extract_labeled_points(points, labels, rand_label)

        # Anomaly: if there are no points in this cluster, ignore annealing step
        if points_with_rand_label.shape[0] == 0:
            direction_point = centroid
        else:
            rand_index = np.random.randint(0, points_with_rand_label.shape[0])
            direction_point = points_with_rand_label[rand_index]
    elif method == 'cluster_mean':
        # Direction point is mean of random points taken from every cluster respectively
        rand_points = get_random_points_from_clusters(points, labels)
        direction_point = np.mean(rand_points, axis=0)
    elif method == 'centroid_split':
        # Direction point is in opposite direction from nearest centroid (centroids are 'splitting')
        indices = np.array(range(centroids.shape[0]))
        other_centroids = centroids[np.where(indices != label_j)]
        distances = la.norm(centroid - other_centroids, ord=2, axis=1)
        nearest_centroid_index = np.argmin(distances)
        nearest_centroid = centroids[nearest_centroid_index]
        direction_point = centroid - nearest_centroid
    elif method == 'centroid_gather':
        # Direction point is mean of current centroids
        direction_point = np.mean(centroids, axis=0)
    else:
        raise ValueError(f'Unknown annealing method: {method}')

    # Annealing vector is weighted with respect to annealing_weight_function
    if annealing_weight_function == 'fixed':
        weight = beta
    else:
        weight = annealing_weight(it, annealing_weight_function, beta)

    annealing_vector = weight * (direction_point - centroid)

    if method == 'min' or (method == 'maxmin' and it % 2 == 0):
        # In cthese cases centroid 'jumps' over directional point by the distance + w% of that distance
        annealing_vector += (direction_point - centroid)

    return annealing_vector, direction_point, weight


def anneal_centroids(points,
                     centroids,
                     labels,
                     it,
                     bounds=None,
                     annealing_prob_function='sqrt',
                     alpha=1,
                     annealing_method='max',
                     annealing_weight_function='log',
                     beta=1.2,
                     ):
    """
    :param points: Points from the training set to be clustered: numpy array of shape (m, n)
    Note: Some annealing methods will require entire training set to evaluate annealed centroids
    :param centroids: Cluster centroids to be 'annealed': numpy array of shape (k, n)
    :param labels: Current cluster labels: numpy array of shape (m, )
    :param it: Current iteration of the algorithm: integer
    :param bounds: Lower and upper bounds (min and max) of the training set (points). If None, they are calculated,
    else unpacked from a tuple.
    :param annealing_prob_function: Annealing probability decreasing function: string
     Possible values: 'exp', 'log', 'sq', 'sqrt', 'sigmoid', 'recip'
    :param alpha: Tunning parameter for annealing function: real number
    :param annealing_method: Specifies how the centroids are annealed (i.e moved from their current position): string
    Possible values: 'random', 'min', 'max', 'maxmin', 'cluster_own', 'cluster_other', 'cluster_mean'
    :param annealing_weight_function: Decreasing function between 0 and 1 that calculates the weight of centroids
    movement: string
     Possible values: 'exp', 'log', 'sq', 'sqrt', 'sigmoid', 'recip', 'flex', 'fixed' - in this case function is
     ignored and only beta parameter is taken in account
    :param beta: Tunning parameter for annealing vector calculation: real number
    :return: Annealed centroids (centroids with updated positions in n-dimensional space)
    """

    k = centroids.shape[0]
    annealed_centroids = copy.deepcopy(centroids)
    annealed_indices = []
    annealing_weights = []
    n_annealings = 0

    p = annealing_probability(it, annealing_prob_function=annealing_prob_function, alpha=alpha)

    for i in range(k):
        q = np.random.uniform(0, 1)

        # If p > q, simulated annealing is applied to centroids
        if p > q:
            # Annealing vector for centroids[i]
            annealing_vector, _, weight = \
                calculate_annealing_vector(points,
                                           labels,
                                           centroids,
                                           i,
                                           it,
                                           bounds=bounds,
                                           annealing_method=annealing_method,
                                           annealing_weight_function=annealing_weight_function,
                                           beta=beta
                                           )
            annealed_centroids[i] += annealing_vector
            annealed_indices.append(i)
            annealing_weights.append(weight)
            n_annealings += 1

    return annealed_centroids, n_annealings, np.array(annealed_indices), np.array(annealing_weights)


def get_centroid_pairs(mean_centroids, annealed_centroids):
    """
    :param mean_centroids: Centroids updated by a regular K-Means update: numpy array of shape (l, n)
    :param annealed_centroids: Centroids updated by simulated annealing step: numpy array of shape (l, n)
    :return: Array of centroid pairs, prepared for annealing tracking: numpy array of shape (l, 2, n)
    """

    l = mean_centroids.shape[0]
    n = mean_centroids.shape[1]

    centroid_pairs = np.zeros(shape=(l, 2, n))
    centroid_pairs[:, 0, :] = mean_centroids
    centroid_pairs[:, 1, :] = annealed_centroids

    return centroid_pairs


class KMESA:
    def __init__(self,
                 k_clusters=5,
                 init='random',
                 init_centroids=None,
                 n_init=10,
                 max_iter=300,
                 tol=1e-4,
                 ecr_method='random',
                 simulated_annealing_on=True,
                 annealing_method='max',
                 annealing_prob_function='sqrt',
                 alpha=3,
                 annealing_weight_function='log',
                 beta=4,
                 convergence_tracking=False,
                 annealing_tracking=False,
                 ecr_tracking=False,
                 tracking_scaler=None
                 ):
        """
        :param k_clusters: Number of clusters
        :param init: Centroids initialization method
        :param init_centroids: Specify exact initial centroids if neccessary
        :param n_init: Number of reinitalization iterations
        :param max_iter: Maximum number of iterations for convergence
        :param tol: Convergence tolerance
        :param ecr_method: Empty clusters resolution method
        :param simulated_annealing_on: Simulated annealing turned on or off
        :param annealing_method: Annealing method used for centroids annealing
        :param annealing_prob_function: Annealing probability for centroids
        :param alpha: Annealing probability hyperparameter
        :param annealing_weight_function: Annealing weight used for centroid annealing
        :param beta: Annealing weight hyperparameter
        :param convergence_tracking: Track algorithm convergence
        :param annealing_tracking: Track annealings in every iteration
        :param tracking_scaler: Scaler used for dataset scaling (used for inverse transformation of centroids)
        """

        self.k_clusters = k_clusters
        self.init = init
        self.init_centroids = init_centroids
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.ecr_method = ecr_method
        self.simulated_annealing_on = simulated_annealing_on
        self.annealing_prob_function = annealing_prob_function
        self.alpha = alpha
        self.annealing_method = annealing_method
        self.annealing_weight_function = annealing_weight_function
        self.beta = beta
        self.convergence_tracking = convergence_tracking
        self.annealing_tracking = annealing_tracking
        self.ecr_tracking = ecr_tracking
        self.tracking_scaler = tracking_scaler

        if self.simulated_annealing_on is False:
            self.annealing_tracking = False

        self.best_result_index_ = None
        self.labels_ = None
        self.centroids_ = None
        self.scaled_centroids_ = None
        self.inertia_ = None
        self.n_iter_ = None
        self.total_annealings_ = None
        self.total_ecr_ = None
        self.history_ = None
        self.tracking_history_ = None
        self.time_info_ = None
        self._legend_annealing_prob = None
        self._legend_annealing_weight = None

        self._set_prob_functions_metadata()

    def _set_prob_functions_metadata(self):
        if self.annealing_prob_function == 'exp':
            self._legend_annealing_prob = r'$p = e^{\frac{-it}{\alpha}}$'
        elif self.annealing_prob_function == 'log':
            self._legend_annealing_prob = r'$p = \frac{ln(1 + \alpha)}{ln(it + \alpha)}$'
        elif self.annealing_prob_function == 'sq':
            self._legend_annealing_prob = r'$p = min(\frac{\alpha + it}{it^2}, 1)$'
        elif self.annealing_prob_function == 'sqrt':
            self._legend_annealing_prob = r'$p = \frac{\alpha}{\sqrt{it - 1} + \alpha}$'
        elif self.annealing_prob_function == 'sigmoid':
            self._legend_annealing_prob = r'$p = \frac{1}{1 + \frac{it - 1}{\alpha + e^{-it}}}$'
        elif self.annealing_prob_function == 'recip':
            self._legend_annealing_prob = r'$p = \frac{1 + \alpha}{it + \alpha}$'
        elif self.annealing_prob_function == 'flex':
            self._legend_annealing_prob = r'$p = \frac{1}{it^{\alpha}}$'
        elif self.annealing_prob_function == 'fixed':
            self._legend_annealing_prob = r'$p = \alpha$'

        if self.annealing_weight_function == 'exp':
            self._legend_annealing_weight = r'$w = e^{\frac{-it}{\beta}}$'
        elif self.annealing_weight_function == 'log':
            self._legend_annealing_weight = r'$w = \frac{ln(1 + \beta)}{ln(it + \beta)}$'
        elif self.annealing_weight_function == 'sq':
            self._legend_annealing_weight = r'$w = min(\frac{\beta + it}{it^2}, 1)$'
        elif self.annealing_weight_function == 'sqrt':
            self._legend_annealing_weight = r'$w = \frac{\beta}{\sqrt{it - 1} + \beta}$'
        elif self.annealing_weight_function == 'sigmoid':
            self._legend_annealing_weight = r'$w = \frac{1}{1 + \frac{it - 1}{\beta + e^{-it}}}$'
        elif self.annealing_weight_function == 'recip':
            self._legend_annealing_weight = r'$w = \frac{1 + \beta}{it + \beta}$'
        elif self.annealing_weight_function == 'flex':
            self._legend_annealing_weight = r'$w = \frac{1}{it^{\beta}}$'
        elif self.annealing_weight_function == 'fixed':
            self._legend_annealing_weight = r'$w = \beta$'

        self._colors = ['red', 'green', 'blue', 'yellow', 'brown', 'purple', 'm', 'cyan', 'indigo', 'forestgreen',
                        'plum', 'teal', 'orange', 'pink', 'lime', 'gold', 'lightcoral', 'cornflowerblue',
                        'orchid', 'darkslateblue', 'slategray', 'peru', 'steelblue', 'crimson', 'gray', 'darkorange']

    def fit(self, points):
        """
        :param points: Points from dataset to be clustered: numpy array of shape (m, n)
        :return: None. Estimator attributes are updated, such as labels_ and centroids_
        """

        start_ns = time.time_ns()

        lower_bound = np.min(points, axis=0)
        upper_bound = np.max(points, axis=0)

        history = {
            'labels': [],
            'centroids': [],
            'inertia': [],
            'n_iter': [],
            'total_annealings': [],
            'total_ecr': []
        }

        if self.convergence_tracking:
            tracking_history = []

        if type(points) == pd.DataFrame:
            points = np.array(points)

        for n_it in range(self.n_init):
            if self.init_centroids is not None:
                initial_centroids = self.init_centroids
            elif self.init == 'random':
                initial_centroids = initialize_centroids_random(points, self.k_clusters, (lower_bound, upper_bound))
            elif self.init == 'k-means++':
                initial_centroids = initialize_centroids_advanced(points, self.k_clusters)

            # Iteration 0 (i.e initial state of the algorithm, with initial centroids and labels assigned to them)
            labels = assign_points_to_centroids(points, initial_centroids)
            centroids, labels, n_ecr, ecr_indices = empty_clusters_resolution(points, initial_centroids, labels,
                                                                              ecr_method=self.ecr_method,
                                                                              bounds=(lower_bound, upper_bound))

            # Track number of annealings and empty cluster resolutions
            total_annealings = 0
            total_ecr = n_ecr

            if self.convergence_tracking:
                tracking_history.append({})

                # Keeping track of 0th iteration (i.e initial algorithm state with initial centroids)
                if self.tracking_scaler is None:
                    tracking_history[n_it]['centroids'] = [centroids]
                else:
                    tracking_history[n_it]['centroids'] = [self.tracking_scaler.inverse_transform(centroids)]

                tracking_history[n_it]['labels'] = [labels]
                tracking_history[n_it]['n_iter'] = 0

                if self.annealing_tracking:
                    tracking_history[n_it]['n_annealings'] = [0]
                    tracking_history[n_it]['annealing_history'] = [None]
                    tracking_history[n_it]['annealing_weights'] = [None]

                if self.ecr_tracking:
                    tracking_history[n_it]['n_ecr'] = [n_ecr]

                    if n_ecr > 0:
                        # Extract annealed centroids and corresponding ECR-updated centroids
                        if self.tracking_scaler is None:
                            init_centroids_ind = initial_centroids[ecr_indices]
                            ecr_centroids_ind = centroids[ecr_indices]
                        else:
                            init_centroids_ind = \
                                self.tracking_scaler.inverse_transform(initial_centroids[ecr_indices])
                            ecr_centroids_ind = \
                                self.tracking_scaler.inverse_transform(centroids[ecr_indices])

                        centroid_pairs = get_centroid_pairs(init_centroids_ind, ecr_centroids_ind)

                        tracking_history[n_it]['ecr_history'] = [centroid_pairs]
                    else:
                        tracking_history[n_it]['ecr_history'] = [None]

            # Repeat before convergence
            for it in range(1, self.max_iter + 1):
                # Step 1: Update centroids
                new_centroids = update_centroids(points, centroids, labels)

                # Step 2: Anneal centroids (update their position) in order to avoid local optima
                if self.simulated_annealing_on:
                    mean_centroids = new_centroids
                    new_centroids, n_annealings, annealed_indices, annealing_weights = \
                        anneal_centroids(points,
                                         new_centroids,
                                         labels,
                                         it,
                                         bounds=(lower_bound, upper_bound),
                                         annealing_prob_function=self.annealing_prob_function,
                                         alpha=self.alpha,
                                         annealing_method=self.annealing_method,
                                         annealing_weight_function=self.annealing_weight_function,
                                         beta=self.beta,
                                         )
                    total_annealings += n_annealings

                    # Keep track of number of annealings occured in current iteration
                    if self.convergence_tracking and self.annealing_tracking:
                        tracking_history[n_it]['n_annealings'].append(n_annealings)

                        # If there were any annealings in current iteration
                        if n_annealings > 0:
                            # Extract mean_centroids and corresponding annealed_centroids
                            if self.tracking_scaler is None:
                                mean_centroids_ind = mean_centroids[annealed_indices]
                                annealed_centroids_ind = new_centroids[annealed_indices]
                            else:
                                mean_centroids_ind = \
                                    self.tracking_scaler.inverse_transform(mean_centroids[annealed_indices])
                                annealed_centroids_ind = \
                                    self.tracking_scaler.inverse_transform(new_centroids[annealed_indices])

                            centroid_pairs = get_centroid_pairs(mean_centroids_ind, annealed_centroids_ind)

                            tracking_history[n_it]['annealing_history'].append(centroid_pairs)
                            tracking_history[n_it]['annealing_weights'].append(annealing_weights)
                        else:
                            tracking_history[n_it]['annealing_history'].append(None)
                            tracking_history[n_it]['annealing_weights'].append(None)

                # Step 3: Assign labels to new centroids and remove empty clusters if any
                labels = assign_points_to_centroids(points, new_centroids)
                ann_centroids_temp = new_centroids
                new_centroids, labels, n_ecr, ecr_indices = empty_clusters_resolution(points, new_centroids, labels,
                                                                                      ecr_method=self.ecr_method,
                                                                                      bounds=(lower_bound, upper_bound))
                total_ecr += n_ecr

                if self.convergence_tracking and self.ecr_tracking:
                    tracking_history[n_it]['n_ecr'].append(n_ecr)

                    # If there were any ECR triggers in current iteration
                    if n_ecr > 0:
                        # Extract annealed centroids and corresponding ECR-updated centroids
                        if self.tracking_scaler is None:
                            ann_centroids_temp_ind = ann_centroids_temp[ecr_indices]
                            ecr_centroids_ind = new_centroids[ecr_indices]
                        else:
                            ann_centroids_temp_ind = \
                                self.tracking_scaler.inverse_transform(ann_centroids_temp[ecr_indices])
                            ecr_centroids_ind = \
                                self.tracking_scaler.inverse_transform(new_centroids[ecr_indices])

                        centroid_pairs = get_centroid_pairs(ann_centroids_temp_ind, ecr_centroids_ind)

                        tracking_history[n_it]['ecr_history'].append(centroid_pairs)
                    else:
                        tracking_history[n_it]['ecr_history'].append(None)

                # Keep track of new centroids and labels assigned to them
                if self.convergence_tracking:
                    if self.tracking_scaler is None:
                        tracking_history[n_it]['centroids'].append(new_centroids)
                    else:
                        tracking_history[n_it]['centroids'].append(
                            self.tracking_scaler.inverse_transform(new_centroids)
                        )

                    tracking_history[n_it]['labels'].append(labels)

                # Check if stopping criterion is reached
                stopping_criterion_reached = check_centroids_update(centroids, new_centroids, self.tol)
                centroids = new_centroids

                if stopping_criterion_reached or it == self.max_iter:
                    if self.convergence_tracking:
                        tracking_history[n_it]['n_iter'] = it

                    break

            # Update history data for reinitialization n_it
            history['labels'].append(labels)
            history['centroids'].append(centroids)
            history['inertia'].append(sum_of_squared_error(points, centroids, labels))
            history['n_iter'].append(it)
            history['total_ecr'].append(total_ecr)

            if self.simulated_annealing_on:
                history['total_annealings'].append(total_annealings)

        # From n_init runs, check which clustering has the lowest SSE. Save data from that run.
        best_result_index = np.argmin(history['inertia'])
        self.best_result_index_ = best_result_index

        self.history_ = history
        self.labels_ = history['labels'][best_result_index]
        self.centroids_ = history['centroids'][best_result_index]

        if self.tracking_scaler is not None:
            self.scaled_centroids_ = self.tracking_scaler.inverse_transform(self.centroids_)

        self.inertia_ = history['inertia'][best_result_index]
        self.n_iter_ = history['n_iter'][best_result_index]
        self.total_ecr_ = history['total_ecr'][best_result_index]

        if self.simulated_annealing_on:
            self.total_annealings_ = history['total_annealings'][best_result_index]
        else:
            self.total_annealings_ = 0

        if self.convergence_tracking:
            self.tracking_history_ = tracking_history

        end_ns = time.time_ns()
        self.time_info_ = time_elapsed(start_ns, end_ns)

    def set_scaled_centroids(self, scaled_centroids):
        self.scaled_centroids_ = scaled_centroids

    def _plot_specified_tracking_history(self, points, reinit_iter, show_iter_mod, show_cc_labels, out_file):
        ndim = points.shape[1]

        if ndim <= 1 or ndim > 3:
            print(f'Tracking convergence for data with ndim != [2, 3] unavailable.')
            return

        if reinit_iter == 'best':
            th = self.tracking_history_[self.best_result_index_]
        else:
            th = self.tracking_history_[reinit_iter]

        n_iter = th['n_iter']
        iters = np.arange(1, n_iter + 1, show_iter_mod)
        iters = np.append(iters, n_iter) if iters[-1] != n_iter else iters
        iters_len = iters.shape[0]

        # Important: 0tracking_history iteration is taken in account in tracking_history_, so +1
        n_rows = iters_len // 2 + 1
        n_cols = 2

        fig = plt.figure(figsize=(12, 5 * n_rows))
        subplot_ind = 1

        # Plot initial state
        init_centroids = th['centroids'][0]
        init_labels = th['labels'][0]

        if ndim == 2:
            ax = fig.add_subplot(n_rows, n_cols, 1)
        else:
            ax = fig.add_subplot(n_rows, n_cols, 1, projection='3d')

        for cluster_label in range(self.k_clusters):
            indices = np.where(init_labels == cluster_label)
            cluster_subsample = points[indices]

            label_str = f'Cluster {cluster_label}' if show_cc_labels else '_nolegend_'
            if ndim == 2:
                ax.scatter(cluster_subsample[:, 0], cluster_subsample[:, 1],
                       c=self._colors[cluster_label], s=8, label=label_str)
            else:
                ax.scatter(cluster_subsample[:, 0], cluster_subsample[:, 1], cluster_subsample[:, 2],
                       c=self._colors[cluster_label], s=8, label=label_str)

        label_str = 'Initial centroids' if show_cc_labels else '_nolegend_'
        if ndim == 2:
            ax.scatter(init_centroids[:, 0], init_centroids[:, 1],
                       c='black', s=120, marker='x', label=label_str)
        else:
            ax.scatter(init_centroids[:, 0], init_centroids[:, 1], init_centroids[:, 2],
                       c='black', s=120, marker='x', label=label_str)

        if self.simulated_annealing_on:
            title = f'KMESA initial state (iteration=0)'
        else:
            title = f'K-Means initial state (iteration=0)'

        ax.set_title(title)

        if show_cc_labels:
            ax.legend(prop={'size': 7})

        subplot_ind += 1

        # Plot every show_iter's iteration
        for it in iters:
            centroids = th['centroids'][it]
            labels = th['labels'][it]

            if ndim == 2:
                ax = fig.add_subplot(n_rows, n_cols, subplot_ind)
            else:
                ax = fig.add_subplot(n_rows, n_cols, subplot_ind, projection='3d')

            for cluster_label in range(self.k_clusters):
                indices = np.where(labels == cluster_label)
                cluster_subsample = points[indices]

                label_str = f'Cluster {cluster_label}' if show_cc_labels else '_nolegend_'
                if ndim == 2:
                    ax.scatter(cluster_subsample[:, 0], cluster_subsample[:, 1],
                               c=self._colors[cluster_label], s=8, label=label_str)
                else:
                    ax.scatter(cluster_subsample[:, 0], cluster_subsample[:, 1], cluster_subsample[:, 2],
                               c=self._colors[cluster_label], s=8, label=label_str)

            label_str = 'Centroids' if show_cc_labels else '_nolegend_'
            if ndim == 2:
                ax.scatter(centroids[:, 0], centroids[:, 1],
                           c='black', s=120, marker='x', label=label_str)
            else:
                ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
                           c='black', s=120, marker='x', label=label_str)

            if self.annealing_tracking:
                n_annealings = th['n_annealings'][it]
                centroid_pairs = th['annealing_history'][it]
                annealing_weights = th['annealing_weights'][it]

                if centroid_pairs is not None:  # and annealing_weights is not None
                    labeled_once = False
                    for centroid_pair, annealing_weight in zip(centroid_pairs, annealing_weights):
                        mean_centroid = centroid_pair[0]
                        annealed_centroid = centroid_pair[1]
                        weight_string = f'w = {annealing_weight : .3}'

                        if not labeled_once:
                            if ndim == 2:
                                ax.plot([mean_centroid[0], annealed_centroid[0]],
                                        [mean_centroid[1], annealed_centroid[1]],
                                        c='dimgray',
                                        linewidth=0.8,
                                        label=f'Annealing trigger, ' + weight_string)
                            else:
                                ax.plot([mean_centroid[0], annealed_centroid[0]],
                                        [mean_centroid[1], annealed_centroid[1]],
                                        [mean_centroid[2], annealed_centroid[2]],
                                        c='dimgray',
                                        linewidth=0.8,
                                        label=f'Annealing trigger, ' + weight_string)

                            labeled_once = True
                        else:
                            if ndim == 2:
                                ax.plot([mean_centroid[0], annealed_centroid[0]],
                                        [mean_centroid[1], annealed_centroid[1]],
                                        c='dimgray',
                                        linewidth=0.8)
                            else:
                                ax.plot([mean_centroid[0], annealed_centroid[0]],
                                        [mean_centroid[1], annealed_centroid[1]],
                                        [mean_centroid[2], annealed_centroid[2]],
                                        c='dimgray',
                                        linewidth=0.8)

            if self.ecr_tracking:
                centroid_pairs = th['ecr_history'][it]
                n_ecr = th['n_ecr'][it]

                if centroid_pairs is not None:
                    labeled_once = False
                    for centroid_pair in centroid_pairs:
                        annealed_centroid = centroid_pair[0]
                        ecr_centroid = centroid_pair[1]

                        if not labeled_once:
                            if ndim == 2:
                                ax.plot([annealed_centroid[0], ecr_centroid[0]],
                                        [annealed_centroid[1], ecr_centroid[1]],
                                        c='rosybrown',
                                        linewidth=0.8,
                                        label='ECR trigger')
                            else:
                                ax.plot([annealed_centroid[0], ecr_centroid[0]],
                                        [annealed_centroid[1], ecr_centroid[1]],
                                        [annealed_centroid[2], ecr_centroid[2]],
                                        c='rosybrown',
                                        linewidth=0.8,
                                        label='ECR trigger')

                            labeled_once = True
                        else:
                            if ndim == 2:
                                ax.plot([annealed_centroid[0], ecr_centroid[0]],
                                        [annealed_centroid[1], ecr_centroid[1]],
                                        c='rosybrown',
                                        linewidth=0.8)
                            else:
                                ax.plot([annealed_centroid[0], ecr_centroid[0]],
                                        [annealed_centroid[1], ecr_centroid[1]],
                                        [annealed_centroid[2], ecr_centroid[2]],
                                        c='rosybrown',
                                        linewidth=0.8)

            if self.annealing_tracking and self.ecr_tracking:
                title = f'KMESA: iteration={it}, n_annealings={n_annealings}, n_ecr={n_ecr}'
            elif self.annealing_tracking and not self.ecr_tracking:
                title = f'KMESA: iteration={it}, n_annealings={n_annealings}'
            elif not self.annealing_tracking and self.ecr_tracking:
                title = f'KMESA: iteration={it}, n_ecr={n_ecr}'
            else:
                title = f'K-Means: iteration={it}'

            ax.set_title(title)

            if show_cc_labels or (self.annealing_tracking and n_annealings > 0) or (self.ecr_tracking and n_ecr > 0):
                ax.legend(prop={'size': 7})

            subplot_ind += 1

        if out_file == '_initial_':
            ii32 = np.iinfo(np.int32)
            rand_int = np.random.randint(0, ii32.max)

            ind = self.best_result_index_ if reinit_iter == 'best' else reinit_iter
            fname = f'KMESA_tracking_best_reinit={ind}_v{rand_int}'
        else:
            fname = out_file

        fig.tight_layout()

        if out_file is not None:
            fig.savefig(fname)

        plt.show()

    def plot_tracking_history(self, points, reinit_iter='best', show_iter_mod=1,
                              show_cc_labels=True, out_file='_initial_'):
        if self.tracking_history_ is None:
            print('No tracking histories present. Run algorithm with convergence_tracking=True '
                  'before tracking convergence.')
            return

        if reinit_iter not in ['best', 'all'] + list(range(self.n_init)):
            raise ValueError(f'Invalid reinit_iter argument: {reinit_iter}')

        if reinit_iter == 'all':
            for n_it in range(self.n_init):
                # Tracking history for n_it reinitialization's iteration
                self._plot_specified_tracking_history(points, n_it, show_iter_mod, show_cc_labels, out_file)
        else:
            self._plot_specified_tracking_history(points, reinit_iter, show_iter_mod, show_cc_labels, out_file)

    def plot_iteration(self, points, reinit_iter='best', it=1, show_cc_labels=True, out_file=None):
        ndim = points.shape[1]

        if ndim <= 1 or ndim > 3:
            print(f'Tracking convergence for data with ndim != [2, 3] unavailable.')
            return

        if reinit_iter == 'best':
            th = self.tracking_history_[self.best_result_index_]
        else:
            th = self.tracking_history_[reinit_iter]

        centroids = th['centroids'][it]
        labels = th['labels'][it]

        fig = plt.figure(figsize=(6, 6))

        if ndim == 2:
            ax = fig.add_subplot(111)
        else:
            ax = fig.add_subplot(111, projection='3d')

        for cluster_label in range(self.k_clusters):
            indices = np.where(labels == cluster_label)
            cluster_subsample = points[indices]

            label_str = f'Cluster {cluster_label}' if show_cc_labels else '_nolegend_'
            if ndim == 2:
                ax.scatter(cluster_subsample[:, 0], cluster_subsample[:, 1],
                           c=self._colors[cluster_label], s=8, label=label_str)
            else:
                ax.scatter(cluster_subsample[:, 0], cluster_subsample[:, 1], cluster_subsample[:, 2],
                           c=self._colors[cluster_label], s=8, label=label_str)

        label_str = 'Centroids' if show_cc_labels else '_nolegend_'
        if ndim == 2:
            ax.scatter(centroids[:, 0], centroids[:, 1],
                       c='black', s=120, marker='x', label=label_str)
        else:
            ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
                       c='black', s=120, marker='x', label=label_str)


        if self.annealing_tracking:
            n_annealings = th['n_annealings'][it]
            centroid_pairs = th['annealing_history'][it]
            annealing_weights = th['annealing_weights'][it]

            if centroid_pairs is not None:  # and annealing_weights is not None
                labeled_once = False
                for centroid_pair, annealing_weight in zip(centroid_pairs, annealing_weights):
                    mean_centroid = centroid_pair[0]
                    annealed_centroid = centroid_pair[1]
                    weight_string = f'w = {annealing_weight : .3}'

                    if not labeled_once:
                        if ndim == 2:
                            ax.plot([mean_centroid[0], annealed_centroid[0]],
                                    [mean_centroid[1], annealed_centroid[1]],
                                    c='dimgray',
                                    linewidth=0.8,
                                    label=f'Annealing trigger, ' + weight_string)
                        else:
                            ax.plot([mean_centroid[0], annealed_centroid[0]],
                                    [mean_centroid[1], annealed_centroid[1]],
                                    [mean_centroid[2], annealed_centroid[2]],
                                    c='dimgray',
                                    linewidth=0.8,
                                    label=f'Annealing trigger, ' + weight_string)

                        labeled_once = True
                    else:
                        if ndim == 2:
                            ax.plot([mean_centroid[0], annealed_centroid[0]],
                                    [mean_centroid[1], annealed_centroid[1]],
                                    c='dimgray',
                                    linewidth=0.8)
                        else:
                            ax.plot([mean_centroid[0], annealed_centroid[0]],
                                    [mean_centroid[1], annealed_centroid[1]],
                                    [mean_centroid[2], annealed_centroid[2]],
                                    c='dimgray',
                                    linewidth=0.8)

        if self.ecr_tracking:
            centroid_pairs = th['ecr_history'][it]
            n_ecr = th['n_ecr'][it]

            if centroid_pairs is not None:
                labeled_once = False
                for centroid_pair in centroid_pairs:
                    annealed_centroid = centroid_pair[0]
                    ecr_centroid = centroid_pair[1]

                    if not labeled_once:
                        if ndim == 2:
                            ax.plot([annealed_centroid[0], ecr_centroid[0]],
                                    [annealed_centroid[1], ecr_centroid[1]],
                                    c='rosybrown',
                                    linewidth=0.8,
                                    label='ECR trigger')
                        else:
                            ax.plot([annealed_centroid[0], ecr_centroid[0]],
                                    [annealed_centroid[1], ecr_centroid[1]],
                                    [annealed_centroid[2], ecr_centroid[2]],
                                    c='rosybrown',
                                    linewidth=0.8,
                                    label='ECR trigger')

                        labeled_once = True
                    else:
                        if ndim == 2:
                            ax.plot([annealed_centroid[0], ecr_centroid[0]],
                                    [annealed_centroid[1], ecr_centroid[1]],
                                    c='rosybrown',
                                    linewidth=0.8)
                        else:
                            ax.plot([annealed_centroid[0], ecr_centroid[0]],
                                    [annealed_centroid[1], ecr_centroid[1]],
                                    [annealed_centroid[2], ecr_centroid[2]],
                                    c='rosybrown',
                                    linewidth=0.8)

        if self.annealing_tracking and self.ecr_tracking:
            title = f'KMESA: iteration={it}, n_annealings={n_annealings}, n_ecr={n_ecr}'
        elif self.annealing_tracking and not self.ecr_tracking:
            title = f'KMESA: iteration={it}, n_annealings={n_annealings}'
        elif not self.annealing_tracking and self.ecr_tracking:
            title = f'KMESA: iteration={it}, n_ecr={n_ecr}'
        else:
            title = f'K-Means: iteration={it}'

        ax.set_title(title)

        if show_cc_labels or (self.annealing_tracking and n_annealings > 0) or (self.ecr_tracking and n_ecr > 0):
            ax.legend(prop={'size': 7})

        if out_file == '_initial_':
            ii32 = np.iinfo(np.int32)
            rand_int = np.random.randint(0, ii32.max)

            if self.annealing_tracking:
                fname = f'KMESA_it={it}_a_method={self.annealing_prob_function}_v{rand_int}'
            else:
                fname = f'K-Means_it={it}_v{rand_int}'
        else:
            fname = out_file

        if out_file is not None:
            fig.savefig(fname)

        plt.show()

    def plot_iterations(self, points, reinit_iter='best', iterations=range(11), show_cc_labels=True, out_file=None):
        ndim = points.shape[1]

        if ndim <= 1 or ndim > 3:
            print(f'Tracking convergence for data with ndim != [2, 3] unavailable.')
            return

        if reinit_iter == 'best':
            th = self.tracking_history_[self.best_result_index_]
        else:
            th = self.tracking_history_[reinit_iter]

        iters_len = len(iterations)
        n_rows = iters_len // 2 if iters_len % 2 == 0 else iters_len // 2 + 1
        n_cols = 2

        fig = plt.figure(figsize=(12, 5 * n_rows))
        subplot_ind = 1

        # Plot every specified iteration
        for it in iterations:
            centroids = th['centroids'][it]
            labels = th['labels'][it]

            if ndim == 2:
                ax = fig.add_subplot(n_rows, n_cols, subplot_ind)
            else:
                ax = fig.add_subplot(n_rows, n_cols, subplot_ind, projection='3d')

            for cluster_label in range(self.k_clusters):
                indices = np.where(labels == cluster_label)
                cluster_subsample = points[indices]

                label_str = f'Cluster {cluster_label}' if show_cc_labels else '_nolegend_'
                if ndim == 2:
                    ax.scatter(cluster_subsample[:, 0], cluster_subsample[:, 1],
                               c=self._colors[cluster_label], s=8, label=label_str)
                else:
                    ax.scatter(cluster_subsample[:, 0], cluster_subsample[:, 1], cluster_subsample[:, 2],
                               c=self._colors[cluster_label], s=8, label=label_str)

            label_str = 'Centroids' if show_cc_labels else '_nolegend_'
            if ndim == 2:
                ax.scatter(centroids[:, 0], centroids[:, 1],
                           c='black', s=120, marker='x', label=label_str)
            else:
                ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
                           c='black', s=120, marker='x', label=label_str)

            if self.annealing_tracking:
                n_annealings = th['n_annealings'][it]
                centroid_pairs = th['annealing_history'][it]
                annealing_weights = th['annealing_weights'][it]

                if centroid_pairs is not None:  # and annealing_weights is not None
                    labeled_once = False
                    for centroid_pair, annealing_weight in zip(centroid_pairs, annealing_weights):
                        mean_centroid = centroid_pair[0]
                        annealed_centroid = centroid_pair[1]
                        weight_string = f'w = {annealing_weight : .3}'

                        if not labeled_once:
                            if ndim == 2:
                                ax.plot([mean_centroid[0], annealed_centroid[0]],
                                        [mean_centroid[1], annealed_centroid[1]],
                                        c='dimgray',
                                        linewidth=0.8,
                                        label=f'Annealing trigger, ' + weight_string)
                            else:
                                ax.plot([mean_centroid[0], annealed_centroid[0]],
                                        [mean_centroid[1], annealed_centroid[1]],
                                        [mean_centroid[2], annealed_centroid[2]],
                                        c='dimgray',
                                        linewidth=0.8,
                                        label=f'Annealing trigger, ' + weight_string)

                            labeled_once = True
                        else:
                            if ndim == 2:
                                ax.plot([mean_centroid[0], annealed_centroid[0]],
                                        [mean_centroid[1], annealed_centroid[1]],
                                        c='dimgray',
                                        linewidth=0.8)
                            else:
                                ax.plot([mean_centroid[0], annealed_centroid[0]],
                                        [mean_centroid[1], annealed_centroid[1]],
                                        [mean_centroid[2], annealed_centroid[2]],
                                        c='dimgray',
                                        linewidth=0.8)

            if self.ecr_tracking:
                centroid_pairs = th['ecr_history'][it]
                n_ecr = th['n_ecr'][it]

                if centroid_pairs is not None:
                    labeled_once = False
                    for centroid_pair in centroid_pairs:
                        annealed_centroid = centroid_pair[0]
                        ecr_centroid = centroid_pair[1]

                        if not labeled_once:
                            if ndim == 2:
                                ax.plot([annealed_centroid[0], ecr_centroid[0]],
                                        [annealed_centroid[1], ecr_centroid[1]],
                                        c='rosybrown',
                                        linewidth=0.8,
                                        label='ECR trigger')
                            else:
                                ax.plot([annealed_centroid[0], ecr_centroid[0]],
                                        [annealed_centroid[1], ecr_centroid[1]],
                                        [annealed_centroid[2], ecr_centroid[2]],
                                        c='rosybrown',
                                        linewidth=0.8,
                                        label='ECR trigger')

                            labeled_once = True
                        else:
                            if ndim == 2:
                                ax.plot([annealed_centroid[0], ecr_centroid[0]],
                                        [annealed_centroid[1], ecr_centroid[1]],
                                        c='rosybrown',
                                        linewidth=0.8)
                            else:
                                ax.plot([annealed_centroid[0], ecr_centroid[0]],
                                        [annealed_centroid[1], ecr_centroid[1]],
                                        [annealed_centroid[2], ecr_centroid[2]],
                                        c='rosybrown',
                                        linewidth=0.8)

            if self.annealing_tracking and self.ecr_tracking:
                title = f'KMESA: iteration={it}, n_annealings={n_annealings}, n_ecr={n_ecr}'
            elif self.annealing_tracking and not self.ecr_tracking:
                title = f'KMESA: iteration={it}, n_annealings={n_annealings}'
            elif not self.annealing_tracking and self.ecr_tracking:
                title = f'KMESA: iteration={it}, n_ecr={n_ecr}'
            else:
                title = f'K-Means: iteration={it}'

            ax.set_title(title)

            if show_cc_labels or (self.annealing_tracking and n_annealings > 0) or (self.ecr_tracking and n_ecr > 0):
                ax.legend(prop={'size': 7})

            subplot_ind += 1

        if out_file == '_initial_':
            ii32 = np.iinfo(np.int32)
            rand_int = np.random.randint(0, ii32.max)

            ind = self.best_result_index_ if reinit_iter == 'best' else reinit_iter
            fname = f'KMESA_tracking_best_reinit={ind}_v{rand_int}'
        else:
            fname = out_file

        fig.tight_layout()

        if out_file is not None:
            fig.savefig(fname)

        plt.show()

    def plot_annealing_prob_function(self, n_iter=30, color='teal'):
        x = np.arange(1, n_iter + 1, dtype=np.int16)
        y = annealing_probability(x, annealing_prob_function=self.annealing_prob_function, alpha=self.alpha)

        fig, ax = plt.subplots(figsize=(5, 5))

        ax.plot(x, y, c=color)
        ax.set_xlim(0, n_iter + 1)
        ax.set_ylim(0, 1.1)
        ax.set_xlabel('iteration')
        ax.set_ylabel('probability')

        alpha_title = r'$\alpha = $' + f'{self.alpha}'
        ax.set_title(f'Annealing probability function: ' + alpha_title)
        ax.legend([self._legend_annealing_prob], prop={'size': 20})

        plt.show()

    def plot_annealing_weight_function(self, n_iter=30, color='firebrick'):
        x = np.arange(1, n_iter + 1, dtype=np.int16)
        y = annealing_weight(x, annealing_weight_function=self.annealing_weight_function, beta=self.beta)

        fig, ax = plt.subplots(figsize=(5, 5))

        ax.plot(x, y, c=color)
        ax.set_xlim(0, n_iter + 1)
        ax.set_ylim(0, 1.1)
        ax.set_xlabel('iteration')
        ax.set_ylabel('probability')

        beta_title = r'$\beta = $' + f'{self.beta}'
        ax.set_title(f'Annealing weight function: ' + beta_title)
        ax.legend([self._legend_annealing_weight], prop={'size': 20})

        plt.show()

    def plot_annealing_functions(self, n_iter=30, color_prob='teal', color_weight='firebrick'):
        x = np.arange(1, n_iter + 1, dtype=np.int16)
        y_prob = annealing_probability(x, annealing_prob_function=self.annealing_prob_function, alpha=self.alpha)
        y_weight = annealing_weight(x, annealing_weight_function=self.annealing_weight_function, beta=self.beta)

        fig, ax = plt.subplots(figsize=(6, 6))

        ax.plot(x, y_prob, c=color_prob)
        ax.plot(x, y_weight, c=color_weight)
        ax.set_xlim(0, n_iter + 1)
        ax.set_ylim(0, 1.1)
        ax.set_xlabel('iteration')
        ax.set_ylabel('probability')

        alpha_title = r'$\alpha = $' + f'{self.alpha}'
        beta_title = r'$\beta = $' + f'{self.beta}'
        ax.set_title(f'Annealing probability and weight functions: ' + alpha_title + ', ' + beta_title)
        ax.legend([self._legend_annealing_prob, self._legend_annealing_weight], prop={'size': 18})

        plt.show()

    def algorithm_details(self):
        init_method_ignored = ' (ignored)\n' if self.init_centroids is not None else '\n'

        if self.tracking_scaler is None:
            scaler_type = 'None'
        else:
            scaler_type_str = str(type(self.tracking_scaler))
            dot_index = scaler_type_str.rindex('.')
            scaler_type = scaler_type_str[dot_index + 1: -2]

        if not self.simulated_annealing_on:
            info = '--------------- Algorithm details ---------------\n' + \
                   f'    * Type: Standard K-Means\n' + \
                   f'    * Number of clusters (k): {self.k_clusters}\n' + \
                   f'    * Centroid initialization method: {self.init}' + init_method_ignored + \
                   f'    * Initial centroids (specified): {self.init_centroids is not None}\n' + \
                   f'    * Number of initialization repetition: {self.n_init}\n' + \
                   f'    * Maximum iterations: {self.max_iter}\n' + \
                   f'    * Convergence tolerance: {self.tol}\n' + \
                   f'    * Empty clusters resolution method: {self.ecr_method}\n' + \
                   f'    * Convergence tracking: {self.convergence_tracking}\n' + \
                   f'    * ECR tracking: {self.ecr_tracking}\n' + \
                   f'    * Tracking scaler: {scaler_type}\n' + \
                   f'-------------------------------------------------'
        else:
            info = '--------------- Algorithm details ---------------\n' + \
                   f'    * Type: KMESA\n' + \
                   f'    * Number of clusters (k): {self.k_clusters}\n' + \
                   f'    * Centroid initialization method: {self.init}' + init_method_ignored + \
                   f'    * Initial centroids (specified): {self.init_centroids is not None}\n' + \
                   f'    * Number of initialization repetition: {self.n_init}\n' + \
                   f'    * Maximum iterations: {self.max_iter}\n' + \
                   f'    * Convergence tolerance: {self.tol}\n' + \
                   f'    * Empty clusters resolution method: {self.ecr_method}\n' + \
                   f'    * Annealing method: {self.annealing_method}\n' + \
                   f'    * Annealing probability function: {self.annealing_prob_function}\n' + \
                   f'    * Annealing probability alpha: {self.alpha}\n' + \
                   f'    * Annealing weight function: {self.annealing_weight_function}\n' + \
                   f'    * Annealing weight beta: {self.beta}\n' + \
                   f'    * Convergence tracking: {self.convergence_tracking}\n' + \
                   f'    * Annealing tracking: {self.annealing_tracking}\n' + \
                   f'    * ECR tracking: {self.ecr_tracking}\n' + \
                   f'    * Tracking scaler: {scaler_type}\n' + \
                   f'-------------------------------------------------'

        return info

    def print_details(self):
        print(self.algorithm_details())

    def clustering_info(self):
        if self.labels_ is None:
            print('Run algorithm before checking clustering information.')
            return

        if not self.simulated_annealing_on:
            info = '------------- K-Means clustering -------------\n' + \
                   f'    * Iterations before convergence: {self.n_iter_}\n' + \
                   f'    * Total empty cluster resolutions: {self.total_ecr_}\n' + \
                   f'    * Sum of squared error: {self.inertia_ : .3}\n' + \
                   f'    * Time elapsed: {self.time_info_}\n' + \
                   f' ---------------------------------------------'
        else:
            info = '------------- KMESA clustering -------------\n' + \
                   f'    * Iterations before convergence: {self.n_iter_}\n' + \
                   f'    * Total empty cluster resolutions: {self.total_ecr_}\n' + \
                   f'    * Total annealings: {self.total_annealings_}\n' + \
                   f'    * Sum of squared error: {self.inertia_ : .3}\n' + \
                   f'    * Time elapsed: {self.time_info_}\n' + \
                   f' ---------------------------------------------'

        return info

    def print_clustering_info(self):
        print(self.clustering_info())

    def clustering_plot_title(self):
        if self.labels_ is None:
            print('Run algorithm before checking clustering information.')
            return

        if self.simulated_annealing_on:
            title = f'KMESA: n_iter={self.n_iter_}, annealings={self.total_annealings_}, SSE={self.inertia_ : .3}'
        else:
            title = f'K-Means: n_iter={self.n_iter_}, SSE={self.inertia_ : .3}'

        return title

    def plot_clustered_data(self, points, s=10, colors=None, show_cc_labels=True, out_file='_initial_'):
        if self.labels_ is None:
            print('Run algorithm before plotting clustered dataset.')
            return

        ndim = points.shape[1]

        if ndim <= 1 or ndim > 3:
            print(f'Tracking convergence for data with ndim != [2, 3] unavailable.')
            return

        if colors is None:
            colors = self._colors

        fig = plt.figure(figsize=(6, 6))

        if ndim == 2:
            ax = fig.add_subplot(111)
        else:
            ax = fig.add_subplot(111, projection='3d')

        for label in range(self.k_clusters):
            cluster = points[np.where(self.labels_ == label)]

            label_str = f'Cluster {label}' if show_cc_labels else '_nolegend_'
            if ndim == 2:
                ax.scatter(cluster[:, 0], cluster[:, 1],
                           c=colors[label], s=s, label=label_str)
            else:
                ax.scatter(cluster[:, 0], cluster[:, 1], cluster[:, 2],
                           c=colors[label], s=s, label=label_str)

        centroids = self.centroids_ if self.tracking_scaler is None else self.scaled_centroids_

        label_str = 'Centroids' if show_cc_labels else '_nolegend_'
        if ndim == 2:
            ax.scatter(centroids[:, 0], centroids[:, 1],
                       c='black', s=200, marker='x', label=label_str)
        else:
            ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
                       c='black', s=200, marker='x', label=label_str)

        if show_cc_labels:
            ax.legend(loc='upper right')

        ax.set_title(self.clustering_plot_title())

        if out_file == '_initial_':
            ii32 = np.iinfo(np.int32)
            rand_int = np.random.randint(0, ii32.max)

            if self.annealing_tracking:
                fname = f'KMESA_a_method={self.annealing_prob_function}_v{rand_int}'
            else:
                fname = f'K-Means_v{rand_int}'
        else:
            fname = out_file

        if out_file is not None:
            fig.savefig(fname)

        plt.show()
