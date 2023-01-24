# %%
import numpy as np
np.set_printoptions(suppress=True)
np.random.seed(42)
import json
from random_forests_and_bagging import create_bootstrapped_dataset
import sklearn.datasets
from sklearn.ensemble import RandomForestClassifier
import sklearn.tree
from utils import get_classification_data, visualise_predictions
# %%
def project_into_subspace(X, feature_idxs):
    """
    Returns only the features of dataset X at the indices provided 
    feature_idxs should be a list of integers representing the indices of the features that should remain 
    """
    return X[:, feature_idxs] # slice out wanted features, excluding the others

# %%
class RandomForest:
    def __init__(self, n_trees=10, max_depth=4, max_samples=10):
        self.n_trees = n_trees # how many trees in the forest
        self.max_depth = max_depth # what is the max depth of each tree
        self.trees = [] # init an empty list of trees
        self.max_samples = max_samples # how many samples from the whole dataset should each tree be trained on

    def fit(self, X, Y):
        """Fits a bunch of decision trees to input X and output Y"""
        for tree_idx in range(self.n_trees): # for each bootstrapped dataset
            bootstrapped_X, bootstrapped_Y = create_bootstrapped_dataset(X, Y, size=self.max_samples) # get features and labels of new bootstrapped dataset
            n_features = np.random.choice(range(1, bootstrapped_X.shape[1])) # choose the number of features to be used by this tree to make predictions
            subspace_feature_indices = np.random.choice(range(bootstrapped_X.shape[1]), size=n_features) # randomly choose that many features to use as inputs
            projected_X = project_into_subspace(bootstrapped_X, subspace_feature_indices) # remove unused features from the dataset
            tree = sklearn.tree.DecisionTreeClassifier(max_depth=self.max_depth) # init a decision tree
            tree.fit(projected_X, bootstrapped_Y) # fit the tree on these examples
            tree.feature_indices = subspace_feature_indices # give the tree a new attribute: which features were used 
            self.trees.append(tree) # add this tree to the list of trees

    def predict(self, X):
        """Use the fitted decision trees to return predictions"""
        predictions = np.zeros((len(X), self.n_trees)) # empty array of predictions with shape n_examples x n_trees
        for tree_idx, tree in enumerate(self.trees): # for each tree in our forest
            x = project_into_subspace(X, tree.feature_indices) # throw away some features of each input example for this tree to predict based on those alone
            predictions[:, tree_idx] = tree.predict(x) # predict the integer label
        prediction = np.mean(predictions, axis=1) # average predictions from different models
        # prediction = np.round(prediction).astype(int) # comment this line to show probability confidences of predictions rather than integer predictions
        return prediction

    def __repr__(self):
        """Returns a string representation of the random forest"""
        forest = [] # init an empty list of trees
        for idx, tree in enumerate(self.trees): # for each tree in the forest
            forest.append({ # add a dictionary of info about the tree
                'depth': tree.max_depth, # how many binary splits?
                'features': tree.feature_indices.tolist() # which features is it using
            })
        return json.dumps(forest, indent=4) # convert dict to string with a nice indentation
# %%
if __name__ == "__main__":
    m = 500
    X, Y = get_classification_data(m=m, variant="circles", noise=0.1, factor=0.7)


    randomForest = RandomForest(n_trees=80, max_depth=2, max_samples=10) # fit many considerably weak (depth=2) learners
    randomForest.fit(X, Y) # fit model
    randomForest.predict(X) # make predictions
    visualise_predictions(randomForest.predict, X, Y) # visualise
    print('forest:', randomForest) # use our __repr__ method to visualise the tree
# %% SKLEARN IMPLEMENTATION 
randomForest = RandomForestClassifier(n_estimators=80, max_depth=2, max_samples=10) # init random forest
randomForest.fit(X, Y) # fit random forest of decision trees
visualise_predictions(randomForest.predict, X, Y) # visualise
randomForest.score(X, Y) # use the model's score method to compute its accuracy