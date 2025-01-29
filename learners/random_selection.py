import numpy as np
from learners.learner import Classifier

class ALClassifierRandomSelection(Classifier):
    '''Implementation of Active Learning acquisition function built on
        Classifier abstract class in learners.learner.
        ALClassifierRandomSelection randomly selects the next points to be measured'''
    def __init__(self, space_shape:int, cpus:int|None=None, model_type='RF'):
        '''
        @params:
            space_shape: shape of the chemical space which the points can be drawn from
            cpus: number of cpus the model can use for training and prediction
            model_type: type of model to use for the classifier 
                    (Gaussian Process (GP) and Random Forest (RF) are currently supported)
        '''
        super().__init__(space_shape, model_type, cpus)

    def suggest_next_n_points(self, X:np.ndarray, n:int, measured_indices:set)->tuple:
        '''randomly selects and returns the next n points to be measured from the points not yet measured
        @params:
            X: np.ndarray of shape (n_reactions, n_features) of all featurized points in the chemical space
            n: number of points to suggest
            measured_indices: set of indices of points to exclude from suggestion
        @returns:
            uncertainty: np.ndarray of predicted uncertainty for each point
            predicted_surface: np.ndarray of shape (n_reactions,) of predicted values for each point
            next_points: list of indices of the next points to be measured
            point_uncertainties: list of uncertainties for each point in next_points
        '''
        uncertainty = self.predict(X)

        assert len(uncertainty) != 0, 'Failed to predict uncertainty'

        num_options = np.prod(self.shape)

        options = np.array([i for i in range(num_options) if not (i in measured_indices)])

        next_points = np.random.choice(options, n, replace=False).tolist()
        
        point_uncertainties = (abs(np.array([uncertainty[i][0] for i in next_points]) - .5) / .5).tolist()


        return uncertainty, self.predicted_surface, next_points, point_uncertainties