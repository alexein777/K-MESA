import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


def load_dataset_3_clusters_separate(c1_size=25, c2_size=25, c3_size=50):
    """
    :param c1_size: Number of points in cluster 1: integer
    :param c2_size: Number of points in cluster 2: integer
    :param c3_size: Number of points in cluster 3: integer
    :return: Dataset of points with 3 clusters
    """

    np.random.seed(7)

    c1_lower = np.array([[-1.5, -0.5]])
    c1_upper = np.array([[-0.5, 0.5]])
    c2_lower = np.array([[0.6, -0.5]])
    c2_upper = np.array([[1.5, 0.4]])
    c3_lower = np.array([[-0.6, 1]])
    c3_upper = np.array([[1.2, 3]])
    c3_addition_lower = np.array([[-0.1, 1.2]])
    c3_addition_upper = np.array([[0.5, 2.2]])

    p1 = c1_lower + (c1_upper - c1_lower) * np.random.random((c1_size, 2))
    p2 = c2_lower + (c2_upper - c2_lower) * np.random.random((c2_size, 2))
    p3 = c3_lower + (c3_upper - c3_lower) * np.random.random((c3_size, 2))
    p3_addition = c3_addition_lower + (c3_addition_upper - c3_addition_lower) * np.random.random((25, 2))
    noise = np.array([[-0.5, 0.6], [-0.2, 0.65], [0.4, 0.5]])

    correction_indices = np.where(p2[:, 0] < 1)
    p2[correction_indices, 0] += 0.3

    correction_indices = np.where(p3[:, 0] > 0.2)
    p3[correction_indices, 0] -= 0.6

    return np.concatenate([p1, p2, p3, p3_addition, noise])


def create_circle(center=(0, 0), radius=1, n_samples=50):
    return np.array([[center[0] + radius * np.cos(phi), center[1] + radius * np.sin(phi)]
                     for phi in np.linspace(0, 2 * np.pi, n_samples, endpoint=False)])


def create_filled_circle(center=(0, 0), radius=1, n_samples_outer=50):
    c = np.array([center[0], center[1]])
    filled_circle = np.array([c])
    radius_iter = radius
    radius_step = 10 * radius / n_samples_outer
    n_samples_iter = n_samples_outer

    while radius_iter > 0 and n_samples_iter > 0:
        filled_circle = np.concatenate([filled_circle, create_circle(center, radius_iter, n_samples_iter)])
        radius_iter -= radius_step
        n_samples_iter -= np.floor(30 * radius_step)

    return filled_circle


def load_different_density_clusters(n_outer_1=150, n_outer_2=60, n_outer_3=30, noise=False):
    c1 = create_filled_circle(radius=2.5, n_samples_outer=n_outer_1)
    c2 = create_filled_circle(center=(-5, -0.5), radius=1, n_samples_outer=n_outer_2)
    c3 = create_filled_circle(center=(6, 0.8), radius=1.2, n_samples_outer=n_outer_3)

    blob_1, _ = make_blobs(n_samples=100, cluster_std=0.6, center_box=(4, -6))
    blob_2, _ = make_blobs(n_samples=50, cluster_std=1.5, center_box=(-7, 5))

    if noise:
        noise_arr = np.array([[-6, 2], [-5.4, -2], [1.8, 5], [4., -1.2], [6, 2.5]])
        X = np.concatenate([c1, c2, c3, blob_1, blob_2, noise_arr])
    else:
        X = np.concatenate([c1, c2, c3, blob_1, blob_2])

    return X


def plot_prob_function(prob_function, alpha, n_iter=15, color='blue'):
    """
    :param prob_function: Type of function to be plotted: string
    Possible values: 'exp', 'log', 'sq', 'sqrt', 'sigmoid', 'recip', 'flex'
    :param alpha: Hyperparameter: float
    :param n_iter: Number of iterations on x-axis
    :param color: Plot color: string
    :return: None
    """

    x = np.arange(1, n_iter + 1, dtype=np.int16)

    if prob_function == 'exp':
        y = np.exp((-x + 1)/ alpha)
        legend = r'$f(it) = e^{\frac{-it}{\alpha}}$'
    elif prob_function == 'log':
        y = np.log(1 + alpha) / np.log(x + alpha)
        legend = r'$f(it) = \frac{ln(1 + \alpha)}{ln(it + \alpha)}$'
    elif prob_function == 'sq':
        ones = np.zeros(x.shape[0]) + 1
        y = np.min([(alpha + x) / (x ** 2), ones], axis=0)
        legend = r'$f(it) = min(\frac{\alpha + it}{it^2}, 1)$'
    elif prob_function == 'sqrt':
        y = alpha / np.sqrt(x)
        legend = r'$f(it) = \frac{\alpha}{\sqrt{it - 1} + \alpha}$'
    elif prob_function == 'sigmoid':
        y = 1 / (1 + (x - 1) / (alpha + np.exp(-x)))
        legend = r'$f(it) = \frac{1}{1 + \frac{it - 1}{\alpha + e^{-it}}}$'
    elif prob_function == 'recip':
        y = (1 + alpha) / (x + alpha)
        legend = r'$f(it) = \frac{1 + \alpha}{it + \alpha}$'
    elif prob_function == 'flex':
        y = 1 / (x ** alpha)
        legend = r'$f(it) = \frac{1}{it^{\alpha}}$'
    else:
        raise ValueError(f'Unknown probability function: {prob_function}')

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.plot(x, y, c=color)
    ax.set_xlim(0, n_iter + 1)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel('iteration')
    ax.set_ylabel('probability')

    alpha_title = r'$\alpha = $' + f'{alpha}'
    ax.set_title(f'Annealing probability function, ' + alpha_title)
    ax.legend([legend], prop={'size': 20})

    plt.show()


def plot_all_annealing_prob_functions(n_iter=20, alpha=1):
    """
    :param n_iter: Number of iterations on x-axis: integer
    :param alpha: Hyperparameter: float
    :return: None
    """

    n_func = 7
    fig = plt.figure(figsize=(8, 8))

    x = np.arange(1, n_iter + 1)
    y_exp = np.exp((-x + 1) / alpha)
    y_log = np.log(1 + alpha) / np.log(x + alpha)
    y_sq = np.min([(alpha + x) / (x ** 2), np.zeros(x.shape[0]) + 1], axis=0)
    y_sqrt = alpha / (np.sqrt(x - 1) + alpha)
    y_sigmoid = 1 / (1 + (x - 1) / (alpha + np.exp(-x)))
    y_recip = (1 + alpha) / (x + alpha)
    y_flex = 1 / (np.power(x, alpha))
    y = np.array([y_exp, y_log, y_sq, y_sqrt, y_sigmoid, y_recip, y_flex])

    labels = [
        r'$f(it) = e^{\frac{-it}{\alpha}}$',
        r'$f(it) = \frac{ln(1 + \alpha)}{ln(it + \alpha)}$',
        r'$f(it) = min(\frac{\alpha + it}{it^2}, 1)$',
        r'$f(it) = \frac{\alpha}{\sqrt{it - 1} + \alpha}$',
        r'$f(it) = \frac{1}{1 + \frac{it - 1}{\alpha + e^{-it}}}$',
        r'$f(it) = \frac{1 + \alpha}{it + \alpha}$',
        r'$f(it) = \frac{1}{it^{\alpha}}$'
    ]

    colors = ['blue', 'green', 'red', 'yellow', 'cyan', 'm', 'darkorange', 'teal', 'lightcoral', 'crimson', 'plum']
    for i in range(n_func):
        plt.plot(x, y[i], c=colors[i], label=labels[i])

    plt.xlim(0, n_iter + 1)
    plt.ylim(0, 1.1)
    plt.xlabel('iteration', fontsize=14)
    plt.ylabel('probability', fontsize=14)

    alpha_title = r'$\alpha = $' + f'{alpha}'
    plt.title('Decreasing probability functions, ' + alpha_title, fontsize=16)
    plt.legend(loc='upper right', prop={'size': 20})

    plt.show()

    fig.savefig(f'annealing_prob_functions_alpha={alpha}.png')


def time_elapsed(start_ns, end_ns):
    """
    :param start_ns: Timestamp before procedure start in nanoseconds
    :param end_ns: Timestamp after procedure end in nanoseconds
    :return: Time elapsed during procedure run: formated string
    """

    milisec = int(round((end_ns - start_ns) / 1000000))
    secs = 0
    mins = 0
    hours = 0

    while milisec >= 1000:
        milisec -= 1000
        secs += 1

    while secs >= 60:
        secs -= 60
        mins += 1

    while mins >= 60:
        mins -= 60
        hours += 1

    if hours > 0:
        time_string = f'{hours}h {mins}min {secs}s'
    elif mins > 0:
        time_string = f'{mins}min {secs}s'
    elif secs > 0:
        time_string = f'{secs}s {milisec}ms'
    else:
        time_string = f'{milisec}ms'

    return time_string


