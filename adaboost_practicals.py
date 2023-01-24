# %%
from sklearn import metrics
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, cross_val_score 
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# %%
def visualise_confusion_matrix(y_true, y_pred):
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    display = metrics.ConfusionMatrixDisplay(confusion_matrix)
    display.plot()
    plt.show()
# %%
# load in data
breast_cancer_data = load_breast_cancer()
X = pd.DataFrame(breast_cancer_data.data, columns=breast_cancer_data.feature_names)
y = pd.Categorical.from_codes(breast_cancer_data.target, breast_cancer_data.target_names)

# encode labels as numbers
encoder = LabelEncoder()
binary_encoded_y = pd.Series(encoder.fit_transform(y))

# split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, binary_encoded_y, 
                                                        random_state=1, test_size=0.3)

# construct and fit model to the training set
classifier = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1),
    n_estimators=200
)
classifier.fit(X_train, y_train)

# use model to predict malignant tumors on test set
test_predictions = classifier.predict(X_test)

# evaluate predictions
visualise_confusion_matrix(y_test, test_predictions)
# %% hyperparameter tuning
def build_models():
    # dict of models
    adaboost_models = dict()

    # number of decision stumps
    descision_stump = [10, 50, 100, 500, 1000]

    for i in descision_stump:
        adaboost_models[str(i)] = AdaBoostClassifier(n_estimators=i)
    
    return adaboost_models

def evaluate_model(model, X, y):
    # define method of validation
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3)

    # validate model based on accuracy score
    accuracy = cross_val_score(model, X, y, scoring="accuracy", cv=cv, n_jobs=1)

    return accuracy
# %%
models = build_models()

results, names = list(), list()

for name, model in models.items():
    scores = evaluate_model(model, X_train, y_train)

    results.append(scores)
    names.append(name)
    print('---->Stump tree (%s)---Accuracy( %.5f)' % (name, np.mean(scores)))
# %% TUNE HYPERPARAMETERS WITH GRID SEARCH
# define classifier
model = AdaBoostClassifier()

grid = {}
grid['n_estimators'] = [10, 50, 100, 200, 500]
grid['learning_rate'] = [0.0001, 0.01, 0.1, 1.0, 1.1, 1.2]
grid['depth'] = [i + 1 for i in range(8)]

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3)

grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=1, cv=cv, scoring="accuracy")
