# %%
import sklearn.tree
from utils import get_classification_data, calc_accuracy, visualise_predictions, show_data
import numpy as np
import matplotlib.pyplot as plt
import json
# %%
def encode_labels(labels):
    labels[labels == 0] = -1
    labels[labels == 1] = +1
    return labels

class AdaBoost:
    def __init__(self, n_layers=20):
        self.n_layers = n_layers
        self.models = [] # init empty list of models

    def sample(self, X, Y, weights):
        idxs = np.random.choice(range(len(X)), size=len(X), replace=True, p=weights)
        X = X[idxs]
        Y = Y[idxs]
        return X, Y

    def calc_model_error(self, predictions, labels, example_weights):
        """Compute the classifier error rate"""
        diff = predictions != labels
        diff = diff.astype(float)
        diff *= example_weights
        diff /= np.sum(example_weights)
        return np.sum(diff)

    def calc_model_weight(self, error, delta=0.01):
        z = (1 - error) / (error + delta) + delta
        return 0.5 * np.log(z)

    def update_weights(self, predictions, labels, model_weight):
        weights = np.exp(- model_weight * predictions * labels)
        weights /= np.sum(weights)
        return weights

    def fit(self, X, Y):
        example_weights = np.full(len(X), 1/len(X)) # assign initial importance of classifying each example as uniform and equal
        for layer_idx in range(self.n_layers):
            model = sklearn.tree.DecisionTreeClassifier(max_depth=1)
            bootstrapped_X, bootstrapped_Y = self.sample(X, Y, example_weights)
            model.fit(bootstrapped_X, bootstrapped_Y)
            predictions = model.predict(X) # make predictions for all examples
            model_error = self.calc_model_error(predictions, Y, example_weights)
            model_weight = self.calc_model_weight(model_error)
            model.weight = model_weight
            self.models.append(model)
            example_weights = self.update_weights(predictions, Y, model_weight)
            # print(f'trained model {layer_idx}')
            # print()

    def predict(self, X):
        prediction = np.zeros(len(X))
        for model in self.models:
            prediction += model.weight * model.predict(X)
        prediction = np.sign(prediction) # comment out this line to visualise the predictions in a more interpretable way
        return prediction

    def __repr__(self):
        return json.dumps([m.weight for m in self.models])
        return json.dumps([
            {
                'weight': model.weight
            }
            for model in self.models
        ], indent=4)

X, Y = get_classification_data(sd=1)
Y = encode_labels(Y)
adaBoost = AdaBoost()
adaBoost.fit(X, Y)
predictions = adaBoost.predict(X)
print(f'accuracy: {calc_accuracy(predictions, Y)}')
visualise_predictions(adaBoost.predict, X, Y)
show_data(X, Y)
print(adaBoost)
# %%
fig = plt.figure()
fig.add_subplot(211)
X, Y = get_classification_data(variant='circles')

for i in range(20):
    adaBoost = AdaBoost(n_layers=i)
    adaBoost.fit(X, Y)
    predictions = adaBoost.predict(X)
    print(f'model {i}')
    print(f'accuracy: {calc_accuracy(predictions, Y)}')
    print(f'weights: {[ round(m.weight, 2) for m in adaBoost.models]}')
    visualise_predictions(adaBoost.predict, X, Y)
    # show_data(X, Y)
    print()

# %% SKLEARN IMPLEMENTATION
import sklearn.ensemble

adaBoost = sklearn.ensemble.AdaBoostClassifier()
adaBoost.fit(X, Y)
predictions = adaBoost.predict(X)
calc_accuracy(predictions, Y)
visualise_predictions(adaBoost.predict, X, Y)