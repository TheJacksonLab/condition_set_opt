import numpy as np
from learners.learner import Classifier

class ALClassifierBasic(Classifier):
    '''Active learning classifier that suggests the top n points with the highest uncertainty that haven't been measured yet'''
    def __init__(self, space_shape:tuple, cutoff_certainty:float=.9, cpus:int|None=None, model_type='RF'):
        '''@params:
            space_shape: shape of the chemical space being learned
            cutoff_certainty: minimum average certainty of top points for the model to stop learning
            cpus: number of cpus the model can use for training and prediction
            model_type: type of model to use for the classifier 
                    (Gaussian Process (GP) and Random Forest (RF) are currently supported)
        '''
        super().__init__(space_shape, model_type, cpus)
        self.certainty_cutoff = (cutoff_certainty - .5)/.5

    def suggest_next_n_points(self, X:np.ndarray, n:int, measured_indices:set)->tuple:
        '''suggests the next n points to be measured based on the points with the highest uncertainty
            @params:
            X: np.ndarray of shape (n_reactions, n_features) of all featurized points in the chemical space
            n: number of points to suggest
            measured_indices: set of indices of points to exclude from suggestion
            @returns:
            uncertainty: np.ndarray of predicted uncertainty for each point
            predicted_surface: np.ndarray of shape (n_reactions,) of predicted values for each point
            next_points: list of indices of the next points to be measured
            point_uncertainties: list of uncertainties for each point in next_points'''
        uncertainty = self.predict(X)

        assert len(uncertainty) != 0, 'Failed to predict uncertainty'

        distance_from_unknown = abs(uncertainty.T[0] - .5)
        rand_nums = list(range(len(X)))
        np.random.shuffle(rand_nums)
        unmeasured_points = np.array([(i, distance_from_unknown[i], rand_nums[i]) for i in range(len(uncertainty)) if i not in measured_indices], dtype=[('idx', int), ('distance', float), ('rand_num', int)])
        uncertainty_order = np.partition(unmeasured_points, n, order=['distance', 'rand_num'])[:n]

        next_points = uncertainty_order['idx'].tolist()
        point_uncertainties = uncertainty_order['distance'].tolist()
        
        avg_point_certainty = np.average(point_uncertainties)

        self.done = avg_point_certainty > self.certainty_cutoff
        return uncertainty, self.predicted_surface, next_points, point_uncertainties
    