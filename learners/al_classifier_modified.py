import itertools
import numpy as np
from learners.learner import Classifier
from space_mat import THRESHOLDED_COUNT
from tools.featurize import convert_point_to_idx
# import time
from math import comb

class ALClassifier(Classifier): # Exploit** and Combined**
    '''Active learning classifier that suggests the top n points that haven't been measured yet with high uncertainty and high complementarity
    Exploit** and Combined** implementation'''
    def __init__(self, space_shape:tuple, all_conditions:list, max_set_size:int, alpha_init_fun=(lambda x: np.linspace(0, 1, x, endpoint=True)), cpus:int|None=None, model_type='RF'):
        '''
        @params:
        space_shape: shape of the chemical space being learned
        all_conditions: list of all possible conditions
        max_set_size: maximum number of conditions to consider in a set
        alpha_init_fun: lambda function with input of the number of points to suggest, generates the alphas for the combined acquisition function, alpha of 1 is pure exploration, alpha of 0 is pure exploitation
        cpus: number of cpus the model can use for training and prediction
        model_type: type of model to use for the classifier 
                (Gaussian Process (GP) and Random Forest (RF) are currently supported)
        '''
        super().__init__(space_shape, model_type, cpus)
        self.all_conditions = all_conditions
        self.max_set_size = max_set_size
        self.alpha_init_fun = alpha_init_fun

    def compute_exploit(self, uncertainty:np.ndarray)->np.ndarray:
        '''compute the exploit value for each point
            @params:
            uncertainty: np.ndarray predicted uncertainty for each point
            @returns:
            exploit: np.ndarray of shape (n_points,) of the exploit value for each point'''
        exploit = [0] * len(uncertainty)
        # t0 = time.time()

        # compute all coverages for all sets
        sets = self.predicted_surface.get_all_set_coverages(self.all_conditions, THRESHOLDED_COUNT(np.prod(self.shape[len(self.all_conditions[0]):]))(.5), self.max_set_size, num_cpus=self.cpus)
        
        # print(f"Time to get all set coverages: {time.time() - t0} seconds")
        # t0 = time.time()
        
        # compute exploit val for all points
        all_reactants = list(itertools.product(*[range(s) for s in self.shape[len(self.all_conditions[0]):]]))
        for i, s in enumerate(sets['set']):
            for reactant in all_reactants:
                if len(s) == 1:
                    exploit[convert_point_to_idx(self.shape, s[0] + reactant)] += sets[i]['coverage'] # maybe add a weight here
                else:
                    for cond in s:
                        exploit[convert_point_to_idx(self.shape, cond + reactant)] += sets[i]['coverage'] * ((np.sum([1- self.predicted_surface[(c + reactant)] for c in s if c != cond]))/(len(s) - 1))
        
        exploit = exploit/(1 + np.sum([comb(len(self.all_conditions) - 1, n) for n in range(1, self.max_set_size)]))
        
        # print(f"time to create exploit: {time.time() - t0}")
        # t0 = time.time()
        
        # multiply by probablity of success for each point
        exploit = exploit * uncertainty.T[1]
        
        return exploit

    def suggest_next_n_points(self, X:np.ndarray, n:int, measured_indices:set)->list:
        '''suggests the next n points to be measured based on the points with the highest uncertainty and highest complementarity
            @params:
            X: np.ndarray of shape (n_reactions, n_features) of all featurized points in the chemical space
            n: number of points to suggest
            measured_indices: set of indices of points to exclude from suggestion
            @returns:
            uncertainty: np.ndarray of predicted uncertainty for each point
            predicted_surface: SpaceMatrix of predicted reaction outcomes
            next_points: list of indices of the next points to be measured
            point_uncertainties: list of uncertainties for each point in next_points'''
        uncertainty = self.predict(X)
        
        explore_a = self.alpha_init_fun(n)
        exploit_a = 1 - explore_a

        # compute explore val for all points
        explore = 1 - (2*abs(uncertainty.T[0] - .5))
        
        # compute exploit val for all points
        exploit = self.compute_exploit(uncertainty)

        unmeasured_points = np.array([(i, explore[i], exploit[i]) for i in range(len(uncertainty)) if i not in measured_indices], dtype=[('idx', int), ('explore', float), ('exploit', float)])

        # multiply explore and exploit by alphas, find max idx along that axis (as long as two are not the same)
        opt_vals = np.array([unmeasured_points['explore']]).T@np.array([explore_a]) + np.array([unmeasured_points['exploit']]).T@np.array([exploit_a])

        top_k_idxs = np.argpartition(opt_vals, -n, axis=0)[-n:]

        top_k_idxs = np.diagonal(top_k_idxs[np.argsort([[opt_vals[idxs[i]][i] for i in range(len(explore_a))] for idxs in top_k_idxs], axis=0, kind='stable')], axis1=1, axis2=2)

        # get unique points for all alpha values
        idxs = top_k_idxs[-1]
        best_idxs = np.full(n, -1, dtype=int)
        for i in range(len(top_k_idxs)):
            idx = idxs[i]
            if idxs[i] in best_idxs:
                j = 2
                while top_k_idxs[-1*j][i] in best_idxs:
                    j += 1
                idx = top_k_idxs[-j][i]
            best_idxs[i] = idx

        next_idxs = unmeasured_points[best_idxs]['idx'].tolist()
        print(next_idxs)
        point_uncertainties = unmeasured_points[best_idxs]['explore'].tolist()

        return uncertainty, self.predicted_surface, next_idxs, point_uncertainties
    

