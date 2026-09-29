import numpy as np
from sklearn.linear_model import LogisticRegression


def test_model_predicts_binary_labels():
    X = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
    y = np.array([0, 1, 1, 0])
    model = LogisticRegression().fit(X, y)
    preds = model.predict(X)
    assert set(preds).issubset({0, 1})
    assert len(preds) == len(y)
