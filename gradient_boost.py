# %%
from utils import get_regression_data, show_regression_data, visualise_predictions, show_data
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import json
# %%
class GradientBoostedDecisionTree:
    def __init__(self, n_trees=10, learning_rate=0.1):
        self.n_trees = n_trees
        self.learning_rate = learning_rate

    def calc_loss(self, predictions, labels):
        return 0.5 * np.sum((predictions - labels)**2)

    def calc_loss_gradient(self, predictions, labels):
        # print('labels:', labels.shape)
        # print('predictions:', predictions.shape)
        # print((labels - predictions).shape)
        return labels - predictions # in the case of MSE loss the gradient is equal to the residual

    def fit(self, X, Y):
        labels = Y
        self.trees = []
        losses = []
        for tree_idx in range(self.n_trees):
            # print(f'training tree {tree_idx}')
            tree = DecisionTreeRegressor(max_depth=2)
            tree.fit(X, labels)
            predictions = tree.predict(X)
            predictions = predictions.reshape(-1, 1)
            labels = self.calc_loss_gradient(predictions, labels) # calculate residual
            self.trees.append(tree)
            # print()

    def predict(self, X):
        predictions = np.zeros((len(X), 1))
        for tree_idx, tree in enumerate(self.trees):
            this_prediction = tree.predict(X).reshape(-1, 1)
            if tree_idx == 0:
                predictions += this_prediction
            else:
                predictions += self.learning_rate * tree.predict(X).reshape(-1, 1)
        
        return predictions

    def __repr__(self):
        return json.dumps([{'depth': t.max_depth} for t in self.trees])


# %%
X, Y = get_regression_data()
show_regression_data(X, Y)

gradientBoostedDecisionTree = GradientBoostedDecisionTree()
gradientBoostedDecisionTree.fit(X, Y)
predictions = gradientBoostedDecisionTree.predict(X)
visualise_predictions(X, predictions, Y)
gradientBoostedDecisionTree.calc_loss(predictions, Y)