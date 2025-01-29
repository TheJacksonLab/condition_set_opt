import numpy as np
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from space_mat import SpaceMatrix
import time

class Learner(ABC):
    '''abstract class for active learning agent, used in Controller'''
    def __init__(self, shape):
        '''
        @params: shape: shape of the chemical space being learned
        '''
        self.done = False # if early stopping is implemented in the learner, setting done to True indicates the active learning process is complete
        self.shape = shape

    @abstractmethod
    def fit(self, X:np.ndarray, y:np.ndarray)->None:
        '''fit the model to the measured data (X, y)
            @params: 
            X: np.ndarray of shape (n_measured, n_features) of featurized points
            y: np.ndarray of shape (n_measured,) of measured values'''
        pass

    @abstractmethod
    def predict(self, X:np.ndarray)->np.ndarray:
        '''predict the reaction outcomes for all points in X
            @params: 
            X: np.ndarray of shape (n_points, n_features) of featurized points to predict
            @returns:
            uncertainty: np.ndarray of predicted uncertainty for each point'''
        pass

    @abstractmethod
    def suggest_next_n_points(self, X:np.ndarray, n:int, measured_indices:np.ndarray)->tuple:
        '''suggest the next n points to be measured
            @params:
            X: np.ndarray of shape (n_points, n_features) of featurized points to predict
            n: number of points to suggest
            measured_indices: set of indices of points to exclude from suggestion
            @returns:
            uncertainty: np.ndarray of predicted uncertainty for each point
            predicted_surface: SpaceMatrix of predicted reaction outcomes
            next_points: list of indices of the next points to be measured
            point_uncertainties: list of uncertainties for each point in next_points
        '''
        pass

    @abstractmethod
    def reset(self)->None:
        '''reset the learner to its initial state'''
        pass

    @abstractmethod
    def initialize_model(self)->None:
        '''initialize the model for the learner'''
        pass


class Classifier(Learner):
    '''abstract class for binary classifier model, used in Controller
    subclasses must implement suggest_next_n_points to complete the active learning agent'''
    def __init__(self, space_shape:int, model_type='RF', cpus:int|None=None):
        '''
        @params:
            space_shape: shape of the chemical space being learned
            cpus: number of cpus the model can use for training and prediction
            model_type: type of model to use for the classifier 
                    (Gaussian Process (GP) and Random Forest (RF) are currently supported)
        '''
        super().__init__(space_shape)
        self.cpus = cpus
        self.model_type = model_type
        self.reset()
    
    def fit(self, X:np.ndarray, y:np.ndarray)->None:
        '''fit the model to the measured data (X, y)
            @params: 
            X: np.ndarray of shape (n_measured, n_features) of featurized points
            y: np.ndarray of shape (n_measured,) of measured values'''
        if self.model == None:
            self.initialize_model()
        self.model.fit(X, y)

    def initialize_model(self)->None:
        '''initialize the model for the learner'''
        if self.model_type == 'GP':
            kernel = 1.0 * RBF(1.0)
            self.model = GaussianProcessClassifier(kernel=kernel, copy_X_train=False, n_jobs=self.cpus, max_iter_predict=1000)
        elif self.model_type == 'RF':
            self.model = RandomForestClassifier(n_estimators=100, n_jobs=self.cpus)
        else:
            raise ValueError('Invalid model type')
            
    def predict(self, X:np.ndarray)->tuple:
        '''compute the uncertainty of the model for each point in X
            @params: X: np.ndarray of shape (n_points, n_features) of featurized points to predict
            @returns: uncertainty: np.ndarray of shape (n_points, 2) of predicted uncertainty for each point (probability of 0, probability of 1)'''
        assert len(X) > 0, 'No points to predict'
        uncertainty = []
        # fixes memory error for large datasets
        start = time.time()
        while len(X) > 25000:
            uncertainty.extend(self.model.predict_proba(X[:25000]))
            X = X[25000:]
        uncertainty.extend(self.model.predict_proba(X))
        print(f"Time to predict: {time.time() - start}")
        # Predicted surface is a matrix of the predicted probability of the positive class (above .5 is predicted to be true)
        # self.featurized must be in the correct order for the predicted surface to be correct
        self.predicted_surface = SpaceMatrix(np.array([uncertainty[i][1] for i in range(len(uncertainty))]).reshape(self.shape, order='C'))
        return np.array(uncertainty)

    def reset(self)->None:
        '''reset the learner to its initial state'''
        self.done = False
        self.predicted_surface = None
        self.initialize_model()
    