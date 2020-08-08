import numpy as np
import numpy.linalg as la


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
    :return labels: Array of centroid indices (i.e cluster labels) with respect to point indices:
    numpy array of shape (m, )
    """

    m = points.shape[0]
    labels = np.zeros(shape=(m, ))

    for i, point in enumerate(points):
        distances = la.norm(point - centroids, ord=2, axis=1)
        centroid_index = np.argmin(distances)
        labels[i] = centroid_index

    return labels


def update_centroids(points, centroids, labels):
    """
    :param points: Points from training set to be clustered: numpy array of shape (m, n)
    :param centroids: Points representing (current) centroids: numpy array pf shape (k, n)
    :param labels: Array of (current) centroid indices (i.e cluster labels) with respect to point indices:
    numpy array of shape (m, )
    :return:
    """

    new_centroids = np.zeros(centroids.shape)
    unique_labels = np.unique(labels)

    for unique_label in unique_labels:
        indices = labels.index(unique_label)  # extracting indices of unique cluster label to get point indices
        new_centroid = np.mean(points[indices])  # mean of points in cluster i
        new_centroids[unique_label] = new_centroid

    return new_centroids


# point = np.array([1, 2, 3])
# centroids = np.array([
#     [1, 2, 2],
#     [0, 0.5, 1],
#     [-1, 0, -1]
# ])
#
# print(point - centroids)
# print(la.norm(point - centroids, ord=2, axis=1))
# print('-------------')
#
# points = np.array([point])
# labels = assign_points_to_centroids(points, centroids)
#
# print(labels)

x = np.array([1, 0, 5, 6, 3, 2])
indices = np.array([1, 3, 4])
print(np.mean(x[indices]))