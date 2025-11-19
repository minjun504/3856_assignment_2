from sklearn.tree import DecisionTreeClassifier
from .models import Models
from ml_assignment_2.config import pre_prune_tree_params, post_prune_tree_params

class DecisionTree(Models):
    def __init__(self, prune=None, random_state=None, param_dist=None):
        super().__init__(prune=prune, random_state=random_state, param_dist=param_dist)

    def classifier(self):
        return DecisionTreeClassifier(random_state=self.random_state)
    
    def get_detaul_params(self):
        if self.prune == "pre":
            return pre_prune_tree_params
        elif self.prune == "post":
            return post_prune_tree_params
        else:
            return {}

    





