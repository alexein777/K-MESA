import numpy as np
from clustering.kmesar import KMESAR
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# X, y = make_blobs(n_samples=100000, centers=15, center_box=(-100, 100), cluster_std=4)
X, y = make_blobs(n_samples=1000, centers=20, center_box=(-120, 120), cluster_std=4)
plt.scatter(X[:, 0], X[:, 1], s=4)

est = KMESAR(k_clusters=20,
             init='random',
             n_init=1,
             max_iter=200,
             tol=1e-3,
             ecr_method='random',
             annealing_method='carousel',
             annealing_prob_function='exp',
             alpha=50,
             annealing_weight_function='log',
             beta=12,
             convergence_tracking=True,
             annealing_tracking=True
             )
est.print_details()
est.plot_annealing_functions()

est.fit(X)
est.print_clustering_info()
est.plot_clustered_data(X)

est.plot_tracking_history(X, show_iter_mod=10, show_cc_labels=False, out_file='KMESAR_20_clusters_MIN_TEST')


