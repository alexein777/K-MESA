import numpy as np
import matplotlib.pyplot as plt


def load_dataset_3_clusters_separate(c1_size=25, c2_size=25, c3_size=50):
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


def plot_prob_function(prob_function, alpha, n_iter=15, color='blue'):
    x = np.arange(1, n_iter + 1, dtype=np.int16)

    if prob_function == 'exp':
        y = np.exp((-x + 1)/ alpha)
        legend = r'$f(it) = e^{\frac{-it}{\alpha}}$'
    elif prob_function == 'log':
        y = np.log(2) / np.log(alpha + x)
        legend = r'$f(it) = \frac{ln2}{ln(it + \alpha)}$'
    elif prob_function == 'sq':
        ones = np.zeros(x.shape[0]) + 1
        y = np.min([(alpha + x) / (x ** 2), ones], axis=0)
        legend = r'$f(it) = min(\frac{\alpha + it}{it^2}, 1)$'
    elif prob_function == 'sqrt':
        y = alpha / np.sqrt(x)
        legend = r'$f(it) = \frac{\alpha}{\sqrt{it}}$'
    elif prob_function == 'sigmoid':
        y = 1 / (1 + (x - 1) / (alpha + np.exp(-x)))
        legend = r'$f(it) = \frac{1}{1 + \frac{it - 1}{\alpha + e^{-x}}}$'
    elif prob_function == 'recip':
        y = (1 + alpha) / (x + alpha)
        legend = r'$f(it) = \frac{1 + \alpha}{it + \alpha}$'
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


def plot_all_annealing_prob_functions(n_iter=15, alpha=1):
    n_func = 6
    fig = plt.figure(figsize=(8, 8))

    x = np.arange(1, n_iter + 1)
    y_exp = np.exp((-x + 1)/ alpha)
    y_log = np.log(2) / np.log(alpha + x)
    y_sq = np.min([(alpha + x) / (x ** 2), np.zeros(x.shape[0]) + 1], axis=0)
    y_sqrt = alpha / (np.sqrt(x - 1) + alpha)
    y_sigmoid = 1 / (1 + (x - 1) / (alpha + np.exp(-x)))
    y_recip = (1 + alpha) / (x + alpha)
    y = np.array([y_exp, y_log, y_sq, y_sqrt, y_sigmoid, y_recip])

    labels = [
        r'$f(it) = e^{\frac{-it}{\alpha}}$',
        r'$f(it) = \frac{ln2}{ln(it + \alpha)}$',
        r'$f(it) = min(\frac{\alpha + it}{it^2}, 1)$',
        r'$f(it) = \frac{\alpha}{\sqrt{it - 1} + \alpha}$',
        r'$f(it) = \frac{1}{1 + \frac{it - 1}{\alpha + e^{-it}}}$',
        r'$f(it) = \frac{1 + \alpha}{it + \alpha}$'
    ]

    colors = ['blue', 'green', 'red', 'yellow', 'cyan', 'm']
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

    fig.savefig(f'annealing_prob_functions_alpha={alpha}')
