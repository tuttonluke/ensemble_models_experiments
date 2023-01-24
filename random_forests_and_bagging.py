# %%
import numpy as np
np.set_printoptions(suppress=True)
np.random.seed(42)
import matplotlib.pyplot as plt
import sklearn.datasets
from utils import get_classification_data, show_data, colors

# %%
def create_bootstrapped_dataset(existing_X, existing_y, size):
    """Create a single bootstrapped dataset.
    """
    # randomly sample indices with replacement (allows for duplicates)
    indxs = np.random.choice(np.arange(len(existing_X)), size=size, replace=True)
    # return examples at these indices
    return existing_X[indxs], existing_y[indxs]

# %%
if __name__ == "__main__":
    m = 500
    X, Y = get_classification_data(m=m, variant="circles", noise=0.1, factor=0.7)
    show_data(X, Y)

    n_trees = 10
    dataset_size = int(m / 5)

    bootstrapped_datasets = create_bootstrapped_dataset(X, Y, dataset_size)