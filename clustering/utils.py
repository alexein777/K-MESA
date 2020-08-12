import numpy as np

def load_dataset_3_clusters_separate(c1_size=25, c2_size=25, c3_size=50):
    # c1: x -> [-1.5, -0.5],  y -> [-0.5, 0.5]
    # c2: x -> [0.5, 1.5],    y -> [-0.5, 0.5]
    # c3: x -> [-0.6, 1.2],   y -> [1, 3]

    # return np.array([
    #     [-0.4, 3], [0.4, 3], [0.1, 2.6], [0.15, 2.61], [0.35, 2.5], [0.39, 2.54],
    #     [-0.2, 2.5], [-0.5, 2.45], []
    # ])

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
