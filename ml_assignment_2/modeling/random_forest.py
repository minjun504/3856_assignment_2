from .models import Models
from sklearn.ensemble import RandomForestClassifier
from ml_assignment_2.config import random_forest_params

class RandomForest(Models):
    def __init__(self, prune=None, random_state=None, param_dist=None):
        super().__init__(prune=prune, random_state=random_state, param_dist=param_dist)
        self._classifier = RandomForestClassifier(random_state=self.random_state)

    def classifier(self):
        return RandomForestClassifier(random_state=self.random_state)
    
    def get_default_params(self):
        return random_forest_params