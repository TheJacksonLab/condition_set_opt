import numpy as np
import itertools
from joblib import Parallel, delayed

# scoring functions for condition set coverage of a chemical space
# more complex scoring functions can be added
# the last argument (x) is an np.ndarray of the maximum yield surface of the conditions
# accepts a surface of yields to compute coverage, where a is the minimum yield to count as a successful reaction
THRESHOLDED_COUNT = lambda s: lambda a: lambda x: (np.sum(x >= a))/s
# takes a surface of 0 and 1 to compute coverage
BINARY_COUNT = lambda s: lambda x: np.sum(x)/s

# accepts a surface of yields to score a condition, where a is the minimum yield to count as a successful reaction
# scored by the number of points above the threshold + the average yield of all points above the threshold
WEIGHTED_COUNT = lambda a: lambda x: np.sum(x >= a) + np.average(x[x >= a])
class SpaceMatrix:
    def __init__(self, matrix:np.ndarray) -> None:
        '''
        @params
        matrix: np.ndarray, the matrix of the chemical space can be either yields or binary values, the shape should be the conditions followed by the reactants such that if n is the number of dimensions in a condition, shape[:n] should all relate to condition options and shape[n:] should all relate to reactant options

        attributes:
        mat: np.ndarray, the matrix of the chemical space
        shape: tuple, the shape of the matrix
        '''
        self.mat = matrix
        self.shape = self.mat.shape

    '''Allow for indexing and slicing of the matrix'''
    def __getitem__(self, key:tuple) -> np.ndarray:
        return self.mat[key]
    
    def __setitem__(self, key:tuple, value:np.ndarray) -> None:
        self.mat[key] = value

    def __iter__(self):
        return iter(self.mat)

    def get_condition_surface(self, condition:tuple) -> np.ndarray:
        '''
        @params:
        condition: tuple of integers that represents a single condition in the chemical space
        
        returns:
        np.ndarray, the yield surface for the given condition over all reactant combinations
        '''
        s = self.mat[condition[0]]
        for c in condition[1:]:
            s = s[c]
        return s

    def get_condition_coverage(self, condition_set:tuple) -> np.ndarray:
        '''
        returns the max yield surface for the given set of condititons for the entire reactant space
        @params:
        condition_set: tuple of tuples of integers that represents a set of conditions in the chemical space

        returns:
        np.ndarray, the maximum yield surface for the given set of conditions over all reactant combinations
        '''
        yield_coverage_surface = np.zeros(self.shape[len(condition_set[0]):])
        for cond in condition_set:
            s = self.get_condition_surface(cond)
            yield_coverage_surface = np.maximum(yield_coverage_surface, s)
        return yield_coverage_surface

    def count_coverage(self, condition_set:tuple, yield_threshold:float) -> int:
        '''
        @params:
        condition_set: tuple of tuples of integers that represents a set of conditions in the chemical space
        yield_threshold: float, the minimum yield to count as a successful reaction

        returns:
        int, the number of reactant combinations that meet the yield threshold for the given set of conditions
        '''
        return self.score_coverage(condition_set, THRESHOLDED_COUNT(np.prod(self.shape[len(condition_set[0]):]))(yield_threshold))
    
    def score_coverage(self, condition_set:tuple, scoring_function) -> float:
        '''
        @params:
        condition_set: tuple of tuples of integers that represents a set of conditions in the chemical space
        scoring_function: function np.ndarray->float, a function that takes in the maximum yield surface of the conditions and returns a score

        returns:
        float, the score of the given set of conditions based on the scoring function
        '''
        yield_coverage_surface = self.get_condition_coverage(condition_set)
        return scoring_function(yield_coverage_surface)
    
    def get_all_set_coverages(self, condition_options:list, scoring_function, max_set_size:int=1, check_subsets=True, num_cpus = 1):
        '''
        @params:
        condition_options: list of tuples of integers, representing all possible or a subset of conditions in the chemical space
        scoring_function: function np.ndarray->float, a function that takes in the maximum yield surface of the conditions and returns a score
        max_set_size: int, the maximum number of conditions to include in a set
        check_subsets: bool, whether to check for sets smaller than the max_set_size
        num_cpus: int, the number of cpus to use for parallel processing

        returns:
        np.ndarray, all sets with their associated coverage with fields "set": tuple, "coverage": float, "size": int, "rand_num": int
        '''
        possible_combos = list(itertools.combinations(condition_options, max_set_size))
        coverages = Parallel(n_jobs=num_cpus)(delayed(self.score_coverage)(s, scoring_function) for s in possible_combos)
        # rand_nums adds randomness to the order of the sets, so a random set will be chosen from sets with the same coverage and size
        rand_nums = list(range(len(possible_combos)))
        np.random.shuffle(rand_nums)
        sets = np.array([(s, coverages[i], -1*len(s), rand_nums[i]) for i, s in enumerate(possible_combos)], dtype=[('set', 'O'), ('coverage', 'f4'),('size', 'i4'),('rand_num', 'i4')])
        if check_subsets and max_set_size > 1:
            smaller_sets = self.get_all_set_coverages(condition_options, scoring_function, max_set_size-1, num_cpus=num_cpus)
            smaller_sets['size'] = -1*smaller_sets['size']
            sets = np.concatenate((sets, smaller_sets[::-1]))
        return sets

    def get_best_set(self, condition_options:list, scoring_function, max_set_size:int=1, check_subsets=True, num_cpus = 1):
        '''
        @params:
        condition_options: list of tuples of integers, representing all possible or a subset of conditions in the chemical space
        scoring_function: function np.ndarray->float, a function that takes in the maximum yield surface of the conditions and returns a score
        max_set_size: int, the maximum number of conditions to include in a set
        check_subsets: bool, whether to check for sets smaller than the max_set_size
        num_cpus: int, the number of cpus to use for parallel processing
        
        returns the best set of conditions based on the scoring function
        '''
        sets = self.get_all_set_coverages(condition_options, scoring_function, max_set_size, check_subsets=check_subsets, num_cpus=num_cpus)
        return np.partition (sets, kth=-1, order=['coverage', 'size', 'rand_num'])[-1:]

    def best_condition_sets(self, condition_options:list, scoring_function, max_set_size:int=1, num_sets:int=None, check_subsets=True, ignore_reduntant_sets=True, num_cpus = 1) -> tuple:
        '''
        @params:
        condition_options: list of tuples of integers, representing all possible or a subset of conditions in the chemical space
        scoring_function: function np.ndarray->float, a function that takes in the maximum yield surface of the conditions and returns a score
        max_set_size: int, the maximum number of conditions to include in a set
        num_sets: int, the number of sets to return
        check_subsets: bool, whether to compute coverages of sets smaller than the max_set_size
        ignore_reduntant_sets: bool, whether to ignore sets that contain a subset with the same coverage
        num_cpus: int, the number of cpus to use for parallel processing

        @returns: list of condition tuples, list of floats, the best condition sets and their corresponding scores in descending order of coverage
        ignores sets that contain a subset with the same coverage
        '''
        sets = self.get_all_set_coverages(condition_options, scoring_function, max_set_size, check_subsets, num_cpus)
        sets.sort(order=['coverage', 'size', 'rand_num'])
        sets['size'] = -1*sets['size']
        
        if not num_sets:
            num_sets = len(sets)
        if not ignore_reduntant_sets:
            return sets[-num_sets:][::-1]
        
        # ignore redundant sets when a subset has already been seen
        subsets_seen = []
        idxs_to_remove = []
        for set_idx in range(len(sets)-1, -1, -1):
            # if set does not contain a set already seen, add it
            unique_set = np.all([not (s.issubset(set(sets[set_idx]['set']))) for s in subsets_seen])
            if unique_set:
                # if set is smaller than max_set_size, add it to subsets_seen
                if len(sets[set_idx]['set']) < max_set_size:
                    subsets_seen.append(set([cond for cond in sets[set_idx]['set']]))
            else:
                idxs_to_remove.append(set_idx)
            if set_idx > 0 and sets[set_idx]['coverage'] != sets[set_idx - 1]['coverage']:
                # reset because coverage of any set will be greater than or equal to the coverage of its subsets
                sets = np.delete(sets, idxs_to_remove)
                subsets_seen = []
                idxs_to_remove = []
        
        return sets[-min(num_sets, len(sets)):][::-1]
    
    def condition_overlap(self, condition_set:tuple, yield_threshold:float) -> float:
        '''
        counts the number of reactant combinations that are covered by more than one condition in the set
        @params:
        condition_set: tuple of tuples of integers that represents a set of conditions in the chemical space
        yield_threshold: float, the minimum yield to count as a successful reaction

        @returns:
        int, the number of reactant combinations that are covered by more than one condition in the set
        '''
        yield_coverage_surface = np.zeros(self.shape[len(condition_set[0]):])
        for cond in condition_set:
            s = self.get_condition_surface(cond) > yield_threshold
            yield_coverage_surface = yield_coverage_surface + s
        return np.sum(yield_coverage_surface > 1)

    
    def rank_conditions(self, condition_options:list, max_set_size:int, cutoff:float) -> dict:
        '''
        @params:
        condition_options: list of tuples of integers, representing all possible or a subset of conditions in the chemical space
        max_set_size: int, the maximum number of conditions to include in a set
        cutoff: float, the minimum yield to count as a successful reaction
        
        @returns:
        list of conditions and list of coverages
        list of conditions is a list of tuples, the conditions in descending order of coverage
        '''
        conditions = [(c,) for c in condition_options]
        for set_size in range(2, max_set_size+1):
            conditions += list(itertools.combinations(condition_options, set_size))
        conditions = conditions[::-1]
        rank_function = lambda x: np.sum(x >= cutoff) 
        conds = np.array([(self.score_coverage(set, rank_function), -1*len(set), self.condition_overlap(set, cutoff), set) for set in conditions],
             dtype=[('coverage', 'f4'),('size', 'i4'), ('overlap', 'f4'), ('condition', 'O')])
        conds.sort(kind = 'stable', order=['coverage', 'size', 'overlap'])
        conds = conds[::-1]
        ranks = np.zeros(len(conds), dtype=int)
        # break ties by uniqueness of conditions -> number covered by only one condition
        rank = 1
        ranks[0] = rank
        for i in range(len(conds) - 1):
            if conds[i]['coverage'] != conds[i+1]['coverage'] or len(conds[i]['condition']) < len(conds[i+1]['condition']):
                rank = i + 2
            ranks[i+1] = rank

        cond_to_rank_map = {conds[i]['condition']: (1 - ranks[i]/len(conditions))*100 for i in range(len(ranks))}
        return cond_to_rank_map