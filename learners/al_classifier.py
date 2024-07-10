import time
import numpy as np
from learners.learner import Classifier, YieldPred

class ALClassifierBasic(Classifier):
    '''Active learning classifier that suggests the top n points with the highest uncertainty that haven't been measured yet to be measured next'''
    def __init__(self, space_shape:tuple, cutoff_certainty=.9, cpus=None):
        super().__init__(space_shape, cpus)
        self.certainty_cutoff = (cutoff_certainty - .5)/.5

    def suggest_next_n_points(self, X:np.ndarray, n:int, measured_indices:set)->list:
        '''next_points is a list of indices of the next points to be measured'''
        last_time = time.time()
        uncertainty = self.predict(X)
        print(f"Time to predict: {time.time() - last_time}")
        last_time = time.time()

        assert len(uncertainty) != 0, 'Failed to predict uncertainty'

        distance_from_unknown = abs(uncertainty.T[0] - .5)
        unmeasured_points = np.array([(i, distance_from_unknown[i]) for i in range(len(uncertainty)) if i not in measured_indices], dtype=[('idx', int), ('distance', float)])
        uncertainty_order = np.partition(unmeasured_points, n, order='distance')[:n]

        next_points = uncertainty_order['idx'].tolist()
        point_uncertainties = uncertainty_order['distance'].tolist()
        
        avg_point_certainty = np.average(point_uncertainties)

        self.done = avg_point_certainty > self.certainty_cutoff
        return uncertainty, self.predicted_surface, next_points, point_uncertainties
    
class ALYieldPredBasic(YieldPred):
    '''Active learning classifier that suggests the top n points with the highest uncertainty that haven't been measured yet to be measured next'''
    def __init__(self, space_shape:tuple, cutoff_std=.05):
        super().__init__(space_shape)
        self.certainty_cutoff = cutoff_std

    def suggest_next_n_points(self, X:np.ndarray, n:int, measured_indices:set)->list:
        '''next_points is a list of indices of the next points to be measured'''
        yield_std = self.predict(X)

        if len(yield_std) == 0:
            return []
        return []

