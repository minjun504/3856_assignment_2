from abc import ABC, abstractmethod
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score, roc_auc_score

class Models(ABC):
    def __init__(self, prune=None, random_state=None, param_dist=None):
        self.prune  = prune
        self.random_state = random_state
        self.param_dist = param_dist
        self.model = None
        self.clf = None

    @abstractmethod
    def classifier(self):
        raise NotImplementedError("Subclasses must define a classifier.")

    @abstractmethod
    def get_default_params(self):
        raise NotImplementedError("Subclasses must define param distributions.")

    def get_param_dist(self):
        return self.param_dist if self.param_dist is not None else self.get_default_params()

    def train(self, X_train, y_train):
        if self.clf is None:
            self.clf = self.classifier()
        search = RandomizedSearchCV(
            estimator = self.clf, 
            param_distributions=self.get_param_dist(),
            n_iter=50,
            cv=5,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model = search.fit(X_train, y_train)
        return self.model.best_estimator_

    def train_pred(self, X_train):
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call .train(X_train, y_train) first.")
        return self.model.predict(X_train)

    def test_pred(self, X_test):
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call .train(X_train, y_train) first.")
        return self.model.predict(X_test)

    def evaluate(self, X_train, X_test, y_train, y_test):
        y_train_pred = self.train_pred(X_train)
        y_test_pred = self.test_pred(X_test)
        avg = "micro"
        return f1_score(y_train, y_train_pred, average=avg), f1_score(y_test, y_test_pred, average=avg)