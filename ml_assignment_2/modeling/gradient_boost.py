from sklearn.ensemble import GradientBoostingClassifier
from ml_assignment_2.config import gradient_boost_params
from .models import Models

class GradientBoost(Models):
    def __init__(self, prune=None, random_state=None, param_dist=None):
        super().__init__(prune=prune, random_state=random_state, param_dist=param_dist)

    def classifier(self):
        return GradientBoostingClassifier(random_state=self.random_state)
    
    def get_default_params(self):
        return gradient_boost_params
    
