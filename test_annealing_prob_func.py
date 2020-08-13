from clustering.kmesar import KMESAR
import numpy as np
import matplotlib.pyplot as plt


est = KMESAR(annealing_prob_function='sigmoid', alpha=2)
est.n_iter_ = 15

est.plot_annealing_prob_function()
