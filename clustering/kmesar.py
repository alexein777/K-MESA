import numpy as np
import numpy.linalg as la
import pandas as pd
import copy
import matplotlib.pyplot as plt

LN2 = np.log(2)


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
    labels = np.zeros(shape=(m,))

    for i, point in enumerate(points):
        distances = la.norm(point - centroids, ord=2, axis=1)
        centroid_index = np.argmin(distances)
        labels[i] = centroid_index

    return labels


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
    :return: Updated centroids: numpy array of shape (k, n)
    """

    k_cluster_labels = centroids.shape[0]
    new_centroids = np.zeros(centroids.shape)

    for label_j in range(k_cluster_labels):
        points_with_label_j = extract_labeled_points(points, labels, label_j)  # extracting points with k label
        new_centroid = np.mean(points_with_label_j, axis=0)  # mean of points in cluster k
        new_centroids[label_j] = new_centroid  # updating current centroid with a new value

    return new_centroids


def sum_of_squared_error(points, centroids, labels):
    """
    :param points:  Points from training set to be clustered: numpy array of shape (m, n)
    :param centroids: Points representing cluster centroids: numpy array pf shape (k, n)
    :param labels: Current centroid indices (i.e cluster labels) with respect to point indices:
    numpy array of shape (m, )
    :return: Sum of squared error between cluster centroids and points assigned to those centroids: real number
    """

    k_cluster_labels = centroids.shape[0]
    sse = 0

    for label_j in range(k_cluster_labels):
        points_with_label_j = extract_labeled_points(points, labels, label_j)
        sse += np.sum(np.power(points_with_label_j - centroids[label_j], 2))

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

    stoppping_criterion_reached = (diff <= tol).all()

    return stoppping_criterion_reached


def annealing_probability(it, annealing_function_prob, alpha=1):
    """
    :param it: Current iteration of the algorithm: integer
    :param annealing_function_prob: Decreasing function between 0 and 1 representing annealing probabilty: string
    Possible values: 'exp', 'log', 'sq', 'sqrt', 'sigmoid'
    :param alpha: Tunning parameter for annealing probability function: real number
    :return: Probability of acceptance the neighbouring solution (e.g moving of centroid in the specified direction)
    """

    if annealing_function_prob == 'exp':
        return np.exp(-it / alpha)
    elif annealing_function_prob == 'log':
        return LN2 / np.log(alpha + it)
    elif annealing_function_prob == 'sq':
        return np.min([(alpha + it) / (it ** 2), 1])
    elif annealing_function_prob == 'sqrt':
        return alpha / np.sqrt(it)
    elif annealing_function_prob == 'sigmoid':
        return 1 - (1 / (1 + np.exp(-it / alpha)))
    else:
        raise ValueError(f'Unknown annealing function probability: {annealing_function_prob}')


def calculate_annealing_vector(points,
                               labels,
                               centroid,
                               label_j,
                               it,
                               bounds=None,
                               annealing_method='random',
                               annealing_vector_function='log',
                               beta=1
                               ):
    """
    :param points: Points from the training set to be clustered: numpy array of shape (m, n)
    Note: Some annealing methods will require entire training set to evaluate annealing vector
    :param labels: Current centroid indices (i.e cluster labels) with respect to point indices:
    numpy array of shape (m, )
    :param centroid: Centroid for which annealing vector is calculated: numpy array of shape (n, )
    :param label_j: Cluster label for cluster represented by centroid: integer
    Note: This parameter is neccessary for extracting the points assigned to this cluster
    :param it: Current iteration of the algorithm: integer
    :param bounds: Lower and upper bounds (min and max) of the training set (points). If None, they are calculated,
    else unpacked from a tuple.
    :param annealing_method: Specifies how the centroids are annealed (i.e moved from their current position): string
    Possible values: 'random', 'min', 'max', 'maxmin'
    :param annealing_vector_function: Decreasing function between 0 and 1 that handles the intensity by which will
    annealing vector pull the centroid in the specified direction: string
    Possible values: 'exp', 'log', 'sq', 'sqrt', 'sigmoid', 'fixed' - in this case function is ignored and only
    beta parameter is taken in account (if beta > 0, beta is clamped to 1)
    Example: if function returns w = 0.8, centroid will move towards directional point by 80% of the annealing vector
    :param beta: Tunning parameter for annealing vector calculation: real number
    :return: Annealing vector that handles the movement direction of a single centroid: numpy array of shape (1, n)
    """

    if beta <= 0:
        raise ValueError(f'Bad value for parameter beta: {beta} (expected beta > 0)')

    if annealing_method == 'random':
        # Directional point is random point from n-dimensional space of the training set with given bounds
        if bounds is None:
            lower_bound = np.min(points, axis=0)
            upper_bound = np.max(points, axis=0)
        else:
            lower_bound, upper_bound = bounds

        direction_point = lower_bound + np.random.random(points[0].shape) * (upper_bound - lower_bound)
    elif annealing_method == 'min':
        # Directional point is point from cluster label_j with the lowest distance from current centroid
        points_with_label_j = extract_labeled_points(points, labels, label_j)
        distances = la.norm(centroid - points_with_label_j, ord=2, axis=1)
        min_index = np.argmin(distances)
        direction_point = points_with_label_j[min_index]
    elif annealing_method == 'max':
        # Direction point is point from cluster label_j with the highest distance from current centroid
        points_with_label_j = extract_labeled_points(points, labels, label_j)
        distances = la.norm(centroid - points_with_label_j, ord=2, axis=1)
        max_index = np.argmax(distances)
        direction_point = points_with_label_j[max_index]
    elif annealing_method == 'maxmin':
        # Directional point is point from cluster label_j with the lowest/highest distance from current centroid,
        # depending on parity of current iteration it
        points_with_label_j = extract_labeled_points(points, labels, label_j)
        distances = la.norm(centroid - points_with_label_j, ord=2, axis=1)

        # On first iteration (it=1) max annealing is applied, and every other odd iteration
        if it % 2 != 0:
            index = np.argmax(distances)
        else:
            index = np.argmin(distances)

        direction_point = points_with_label_j[index]

    # Annealing vector is weighted with respect to annealing_vector_function
    if annealing_vector_function == 'fixed':
        if beta > 1:
            beta = 1

        annealing_vector = beta * (direction_point - centroid)
    else:
        w = annealing_probability(it, annealing_vector_function, beta)
        annealing_vector = w * (direction_point - centroid)

        # In case of 'min' annealing, centroids 'jumps' over directional point by the distance + w% of that distance
        if annealing_method == 'min' or (annealing_method == 'maxmin' and it % 2 == 0):
            annealing_vector += (direction_point - centroid)

    return annealing_vector, direction_point


def anneal_centroids(points,
                     centroids,
                     labels,
                     it,
                     bounds=None,
                     annealing_function_prob='sqrt',
                     alpha=1,
                     annealing_method='max',
                     annealing_vector_function='log',
                     beta=1.2
                     ):
    """
    :param points: Points from the training set to be clustered: numpy array of shape (m, n)
    Note: Some annealing methods will require entire training set to evaluate annealed centroids
    :param centroids: Cluster centroids to be 'annealed': numpy array of shape (k, n)
    :param labels: Current cluster labels: numpy array of shape (m, )
    :param it: Current iteration of the algorithm: integer
    :param bounds: Lower and upper bounds (min and max) of the training set (points). If None, they are calculated,
    else unpacked from a tuple.
    :param annealing_function_prob: Annealing probability decreasing function: string (possible values: 'exp', 'log',
    'sq', 'sqrt', 'sigmoid')
    :param alpha: Tunning parameter for annealing function: real number
    :param annealing_method: Specifies how the centroids are annealed (i.e moved from their current position): string
    (possible values: ... )
    :param annealing_vector_function: Decreasing function between 0 and 1 that calculates the weight of centroids
    movement: string (possible values: 'same' - value is equal to p from annealing_function_prob, 'exp', 'log', 'sq', 'sqrt',
    'sigmoid', 'fixed' - in this case function is ignored and only beta parameter is taken in account)
    :param beta: Tunning parameter for annealing vector calculation: real number
    :return: Annealed centroids (centroids with updated positions in n-dimensional space)
    """

    k = centroids.shape[0]
    annealed_centroids = copy.deepcopy(centroids)
    p = annealing_probability(it, annealing_function_prob=annealing_function_prob, alpha=alpha)
    n_annealings = 0

    for i in range(k):
        q = np.random.uniform(0, 1)

        if p > q:
            annealing_vector, _ = calculate_annealing_vector(points,
                                                             labels,
                                                             centroids[i],
                                                             i,
                                                             it,
                                                             bounds=bounds,
                                                             annealing_method=annealing_method,
                                                             annealing_vector_function=annealing_vector_function,
                                                             beta=beta
                                                             )
            annealed_centroids[i] += annealing_vector
            n_annealings += 1

    return annealed_centroids, n_annealings


class KMESAR:
    def __init__(self,
                 k_clusters=5,
                 init='random',
                 init_centroids=None,
                 n_init=10,
                 max_iter=300,
                 tol=1e-4,
                 simulated_annealing_on=True,
                 annealing_function_prob='sqrt',
                 alpha=1,
                 annealing_method='max',
                 annealing_vector_function='log',
                 beta=1,
                 convergence_tracking=False
                 ):

        self.k_clusters = k_clusters
        self.init = init
        self.init_centroids = init_centroids
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.simulated_annealing_on = simulated_annealing_on
        self.annealing_function_prob = annealing_function_prob
        self.alpha = alpha
        self.annealing_method = annealing_method
        self.annealing_vector_function = annealing_vector_function
        self.beta = beta
        self.convergence_tracking = convergence_tracking

        self.labels_ = None
        self.centroids_ = None
        self.inertia_ = None
        self.n_iter_ = None
        self.total_annealings_ = None
        self.history_ = None
        self.tracking_history_ = None

    def fit(self, points):
        lower_bound = np.min(points, axis=0)
        upper_bound = np.max(points, axis=0)
        history = {
            'labels': [],
            'centroids': [],
            'inertia': [],
            'n_iter': [],
            'total_annealings': []
        }

        if self.convergence_tracking:
            tracking_history = []

        if type(points) == pd.DataFrame:
            points = np.array(points)

        for n_it in range(self.n_init):
            if self.init_centroids is not None:
                initial_centroids = self.init_centroids
            else:
                initial_centroids = initialize_centroids_random(points, self.k_clusters, lower_bound, upper_bound)

            centroids = initial_centroids
            total_annealings = 0

            if self.convergence_tracking:
                tracking_history.append({})
                tracking_history[n_it]['centroids'] = [initial_centroids]
                tracking_history[n_it]['labels'] = []
                tracking_history[n_it]['n_iter'] = 0
                tracking_history[n_it]['n_annealings'] = []

            for it in range(self.max_iter):
                labels = assign_points_to_centroids(points, centroids)

                if self.convergence_tracking:
                    tracking_history[n_it]['labels'].append(labels)

                new_centroids = update_centroids(points, centroids, labels)

                if self.simulated_annealing_on:
                    new_centroids, n_annealings = anneal_centroids(points,
                                                                   new_centroids,
                                                                   labels,
                                                                   it + 1,
                                                                   bounds=(lower_bound, upper_bound),
                                                                   annealing_function_prob=self.annealing_function_prob,
                                                                   alpha=self.alpha,
                                                                   annealing_method=self.annealing_method,
                                                                   annealing_vector_function=self.annealing_vector_function,
                                                                   beta=self.beta
                                                                   )
                    total_annealings += n_annealings

                    if self.convergence_tracking:
                        tracking_history[n_it]['centroids'].append(new_centroids)
                        tracking_history[n_it]['n_annealings'].append(n_annealings)

                stopping_criterion_reached = check_centroids_update(centroids, new_centroids, self.tol)
                centroids = new_centroids

                if stopping_criterion_reached or it + 1 == self.max_iter:
                    labels = assign_points_to_centroids(points, centroids)

                    if self.convergence_tracking:
                        tracking_history[n_it]['labels'].append(labels)
                        tracking_history[n_it]['n_iter'] = it

                    break

            history['labels'].append(labels)
            history['centroids'].append(centroids)
            history['inertia'].append(sum_of_squared_error(points, centroids, labels))
            history['n_iter'].append(it)
            history['total_annealings'].append(total_annealings)

        best_result_index = np.argmin(history['inertia'])

        self.history_ = history
        self.labels_ = history['labels'][best_result_index]
        self.centroids_ = history['centroids'][best_result_index]
        self.inertia_ = history['inertia'][best_result_index]
        self.n_iter_ = history['n_iter'][best_result_index]
        self.total_annealings_ = history['total_annealings'][best_result_index]

        if self.convergence_tracking:
            self.tracking_history_ = tracking_history

    def plot_tracking_history(self, points):
        if self.labels_ is None:
            print('No tracking histories present. Run algorithm before tracking convergence.')
            return

        colors = ['red', 'green', 'blue', 'yellow', 'brown', 'm', 'orange', 'cyan', 'plum', 'teal', 'orange', 'pink',
                  'lime', 'gold', 'lightcoral', 'cornflowerblue', 'darkslateblue', 'orchid', 'slategray', 'peru']

        # Proveriti iscrtavanje
        for n_it in range(self.n_init):
            n_iter = self.tracking_history_[n_it]['n_iter']

            n_rows = n_iter // 2 if n_iter % 2 == 0 else n_iter // 2 + 1
            n_cols = 2

            fig = plt.figure(figsize=(10, 2 * n_iter))
            subplot_ind = 1

            for i in range(n_iter):
                centroids = self.tracking_history_[n_it]['centroids'][i]
                labels = self.tracking_history_[n_it]['labels'][i]

                ax = fig.add_subplot(n_rows, n_cols, subplot_ind)

                for cluster_label in range(self.k_clusters):
                    indices = np.where(labels == cluster_label)
                    cluster_subsample = points[indices]

                    ax.scatter(cluster_subsample[:, 0], cluster_subsample[:, 1],
                               c=colors[cluster_label], s=10, label=f'Cluster {cluster_label}')

                ax.scatter(centroids[:, 0], centroids[:, 1], c='black', s=60, marker='x', label='Centroids')

                ax.legend(loc='upper right', prop={'size': 6})
                ax.set_title(f'KMESAR: iteration={i}')

                subplot_ind += 1

            fig.savefig(f'KMESAR_tracking_n_it={n_it}')
            plt.show()
