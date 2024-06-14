import matplotlib as mpl
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score
from space_mat import SpaceMatrix
from space_mat import THRESHOLDED_COUNT

class ChemicalSpace:
    def __init__(self, condition_titles:list, reactant_titles:list, data_file:str, target_title='yield') -> None:
        '''
        @params:
        condition_titles: list of strings, the headers of the condition components in the data file ex. ['ligand', 'solvent', 'temperature']
        reactant_titles: list of strings, the headers of the reactants in the data file ex. ['electrophile', 'nucleophile']
        data_file: string, the path to the csv file that contains the data
        target_title: string, the header of the target column in the data file (default is 'yield')

        attributes:
        reactants_dim: int, the number of reactants used in a given reation
        conditions_dim: int, the number of condition components used to define a condition for a given reaction 
        titles: list of strings, the titles of the dimensions of chemical space, in the order of condition_titles followed by reactant_titles
        labels: list of lists of strings, the unique values for each of the condition components and reactants, in the order of the titles
        yield_surface: SpaceMatrix, the matrix of the chemical space indexed in the order of titles (first condition components followed by reactants)
        shape: tuple, the shape of the yield_surface matrix, conditions then reactants
        all_points: list of tuples, all possible points in the chemical space, a point is a single condition and set of reactants
        all_conditions: list of tuples, all possible conditions in the chemical space
        '''
        self.reactants_dim = len(reactant_titles)
        self.conditions_dim = len(condition_titles)
        self.dataset_name = data_file.split('/')[-1].split('.')[0]
        self.yield_surface = SpaceMatrix(self._create_mat(data_file, condition_titles, reactant_titles, target_title))
        self.shape = self.yield_surface.shape
        self.all_points = list(itertools.product(*[range(s) for s in self.shape]))
        self._y_true = np.array([self.yield_surface[point] for point in self.all_points])
        self.all_conditions = list(itertools.product(*[range(s) for s in self.shape[:self.conditions_dim]]))
        self.descriptors = None

    def _create_mat(self, data_file, cond_titles:list, reactant_titles:list,  target_title)->np.ndarray:
        # Create a matrix of the data
        data = pd.read_csv(data_file)
        unique_rs = [data[r_title].unique() for r_title in reactant_titles]
        unique_conds = [data[cond_title].unique() for cond_title in cond_titles]
        self.titles = cond_titles + reactant_titles
        self.labels = unique_conds + unique_rs
        shape = [len(item) for item in self.labels]
        data_mat = np.zeros(tuple(shape))
        for idx, series in data.iterrows():
            point = [np.where(self.labels[i] == (series[title])) for i, title in enumerate(self.titles)]
            # restrict the yields to be between 0 and 100
            data_mat[tuple(point)] = max(min(series[target_title], 100), 0)
        return data_mat

    def measure_reaction_yield(self, point:list) -> float:
        '''
        @params:
        point: list of integers, the indecies of the condition components and reactants (in the order of titles)
        
        returns:
        float, the yield of the reaction at the given point
        '''
        return self.yield_surface[tuple(point)]
    
    # TODO: extend to subset of points
    def score_classifier_prediction(self, y_pred, cutoff) -> tuple:
        '''
        @params:
        y_pred: np.ndarray, the predicted values of the classifier, in the order of all_points
        cutoff: float, the yield threshold (not inclusive) for the classifier to determine a positive result
        
        returns:
        tuple of floats, the accuracy, precision, and recall of the classifier prediction
        '''
        y_pred = y_pred.T[1] > .5
        y_true = self._y_true > cutoff
        return accuracy_score(y_true, y_pred), precision_score(y_true, y_pred), recall_score(y_true, y_pred)

    def best_condition_sets(self, yield_threshold:float, max_set_size:int=1, num_sets:int=10) -> tuple:
        '''
        @params:
        yield_threshold: float, the maximum yield to count as a failed reaction
        max_set_size: int, the maximum number of conditions to include in a set
        num_sets: int, the number of sets to return
        
        returns:
        list of tuples and list of floats, the best condition sets and their corresponding scores
        '''
        return self.yield_surface.best_condition_sets(self.all_conditions, THRESHOLDED_COUNT(np.prod(self.shape[self.conditions_dim:]))(yield_threshold), max_set_size, num_sets)
    
    def format_condition(self, condition:tuple) -> str:
        '''
        @params:
        condition: tuple of integers, the indecies of the condition components
        
        returns:
        string, the formatted condition using the condition labels'''
        return ', '.join([f'{self.labels[i][c]}' for i, c in enumerate(condition)])

    def plot_conditions(self, file_name, conditions:tuple, yield_threshold=0) -> None:
        '''
        @params:
        file_name: string, the file path location where the plot will be saved
        conditions: tuple of condition tuples
        yield_threshold: float, the minimum yield to count as a successful condition, all reactions below this threshold will be set to 0

        requires that the chemical space has exactly 2 reactants

        saves a plot of the max yield surface for the given conditions using the reactant titles as the x and y labels
        '''
        ylabel = self.titles[self.conditions_dim]
        xlabel = self.titles[self.conditions_dim + 1]
        yields = np.zeros(self.shape[self.conditions_dim:])
        for condition in conditions:
            yields = np.maximum(yields, self.yield_surface.get_condition_surface(condition))
        below_threshold = yields < yield_threshold
        yields[below_threshold] = 0
        plt.imshow(yields, vmin=0, vmax=100)
        plt.colorbar(label='% Yield')
        plt.xlabel(xlabel, fontsize=18)
        plt.ylabel(ylabel, fontsize=18)
        condition_str = '; '.join([self.format_condition(c) for c in conditions])
        if len(conditions) == 1:
            condition_str = f"Condition {conditions[0]} Yield"
        else:
            condition_str = f"Conditions {condition_str} Max Yield"
        plt.title(f'{condition_str}', fontsize=20)
        plt.savefig(file_name)

    def plot_surface(self, zlabel = '% Yield', title='Highest Yield Across All Conditions') -> None:    
        '''
        @params:
        dataset_name: string, the prefix of the file where the plot will be saved
        zlabel: string, the label of the z axis
        title: string, the title of the plot

        requires that the chemical space has exactly 2 reactants

        saves a plot of the max yield surface over all conditions using the reactant titles as the x and y labels to dataset_name.png
        '''
        ylabel = self.titles[-2]
        xlabel = self.titles[-1]
        
        max_yield_surface = np.amax(self.yield_surface.mat.T, axis=tuple(range(2, len(self.shape)))).T
        plt.imshow(max_yield_surface)
        plt.colorbar(label=zlabel)
        plt.xlabel(xlabel, fontsize=15)
        plt.ylabel(ylabel, fontsize=15)
        plt.title(title, fontsize=20)
        plt.savefig(f'{self.dataset_name}.png')
    
    def plot_set_coverages(self, set_sizes:list, cutoff:float)->None:
        coverages = []
        for size in set_sizes:
            conditions = itertools.combinations(self.all_conditions, size)
            coverages.append([self.yield_surface.count_coverage(condition, cutoff) for condition in conditions])
        plt.hist(coverages, bins=20, histtype='bar', label=[f"Sets of size {size}" for size in set_sizes], stacked=False, log = True)
        plt.xlabel('Coverage')
        plt.ylabel('Number of Sets')
        plt.legend()
        plt.title('Reaction Space Coverage Distribution over Condition Sets')

    def plot_combination_of_individual(self, cutoff:float)->None:
        ax1:plt.Axes
        ax2:plt.Axes
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        combination_mat = np.zeros((len(self.all_conditions), len(self.all_conditions)))
        ordered_conds, _ = self.best_condition_sets(cutoff, 1, None)
        print(ordered_conds)
        for c in range(len(ordered_conds)):
            for c2 in range(c, len(ordered_conds)):
                combination_mat[c][c2] = self.yield_surface.count_coverage((ordered_conds[c][0], ordered_conds[c2][0]), cutoff)
                combination_mat[c2][c] = combination_mat[c][c2]
        ax1.imshow(combination_mat, origin='lower')
        ax1.set_xlabel('Ranked Individual Conditions')
        ax1.set_ylabel('Ranked Individual Conditions')
        # ax1.colorbar(label='Coverage')
        # # ticks = [f'{cond[0]}' for cond in ordered_conds]
        # # plt.xticks(range(len(ordered_conds)), ticks, rotation=90)
        # # plt.yticks(range(len(ordered_conds)), ticks)
        ax1.set_title('Combinations of Individual Conditions')
        max_coverage = combination_mat.max(axis = 0)
        print(max_coverage)
        avg_coverage = combination_mat.mean(axis = 0)
        print(avg_coverage)
        coverage_std = combination_mat.std(axis = 0)
        ax2.plot(range(len(ordered_conds)), max_coverage, label=['Max Coverage'])
        ax2.plot(range(len(ordered_conds)), avg_coverage, label=['Average Coverage'])
        ax2.plot(range(len(ordered_conds)), coverage_std, label=['Coverage Std'])

    # TODO
    def plot_reaction_difficulty(self, cutoff:float)->None:
        pass

    # TODO: fix for buchwald hartwig
    def plot_subset_overlap(self, set_size:int, cutoff:float, max_num_conditions=None)->None:
        ordered_conds, coverage = self.best_condition_sets(cutoff, 1, None)
        conds_condensed = [cond[0] for cond in ordered_conds]
        # cond_idx = np.argsort(conds_condensed)
        # print(ordered_conds)
        # print(cond_idx)
        ordered_sets, set_coverage = self.best_condition_sets(cutoff, set_size, len(self.all_conditions))
        combination_mat = np.full((len(self.all_conditions) + 1, len(self.all_conditions)), np.nan)
        for i, set in enumerate(ordered_sets):
            for cond in set:
                idx = conds_condensed.index(cond)
                combination_mat[idx][i] = coverage[idx]
                # combination_mat[cond_idx[cond[0]]][i] = 
        combination_mat[-1] = set_coverage
        cmap = mpl.colormaps.get_cmap('viridis')  # viridis is the default colormap for imshow
        cmap.set_bad('white', alpha=1)
        if max_num_conditions:
            combination_mat = combination_mat[:max_num_conditions]+[combination_mat[-1]]
        plt.imshow(combination_mat, origin='lower', cmap=cmap)
        plt.ylabel('Ranked Individual Conditions')
        plt.xlabel(f'Ranked Distinct Condition Sets of at most Size {set_size}')
        plt.colorbar(label='Coverage')
        plt.title('Combinations of Condition Sets')

    def max_possible_coverage(self, cutoff:float)->int:
        return self.yield_surface.count_coverage(self.all_conditions, cutoff)
    
    def plot_coverage_over_yield(self)->None:
        ax1:plt.Axes
        ax2:plt.Axes
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        # plot of coverage vs yield
        max_yield_surface = np.amax(self.yield_surface.mat.T, axis=tuple(range(self.reactants_dim, len(self.shape))))
        yields = np.sort(max_yield_surface.flatten())[::-1]
        coverages = range(1, len(yields) + 1)/(np.prod(self.shape[self.conditions_dim:]))
        # idx = int(np.floor(.6 * (np.prod(self.shape[self.conditions_dim:]))))
        idx = int(np.floor(.6 * (len(yields))))
        y = yields[idx]

        # plot of % of successful reactions vs yield
        all_yields = np.sort(self.yield_surface.mat.flatten())[::-1]
        successful_reactions = range(1, len(all_yields) + 1)/(np.prod(self.shape))
        all_idx = len(all_yields) - np.searchsorted(all_yields[::-1], y, side='right')
        print(all_idx)
        print(y)
        print(all_yields[all_idx])

        # find location where 50% of reactions are successful
        all_idx2 = int(np.floor(.5 * (len(all_yields))))
        y2 = all_yields[all_idx2]
        idx2 = len(yields) - max(np.searchsorted(yields[::-1], y2, side='left'), 1) # handle case where y2 is less than the minimum yield of the max yield surface
        if yields[idx2] < y2 and idx2 > 0:
            idx2 -= 1
        print(idx2)
        print(y2)
        print(yields[idx2])

        ax1.plot(yields, coverages)
        # ax1.plot([y, y], [coverages[idx], 0], 'k-')
        # ax1.plot([min(y2,yields[-1]), yields[0]], [coverages[idx]]*2, 'k-')
        ax1.scatter([y, y2], [coverages[idx], coverages[idx2]])
        ax1.annotate(f'{y:.2f}% yield,\n{coverages[idx]:.3f} coverage', (y, coverages[idx]), textcoords="offset points", xytext=(0,10), ha='left')
        # ax1.plot([y2, y2], [coverages[idx2], 0], 'k-')
        # ax1.plot([min(y2,yields[-1]), yields[0]], [coverages[idx2]]*2, 'k-')
        ax1.annotate(f'{y2:.2f}% yield,\n{coverages[idx2]:.3f} coverage', (y2, coverages[idx2]), textcoords="offset points", xytext=(0,10), ha='left')
        ax1.set_xlabel('Yield')
        ax1.set_ylabel('Coverage')
        ax1.set_title('Coverage vs Yield')

        ax2.plot(all_yields, successful_reactions)
        # ax2.plot([y, y], [successful_reactions[all_idx], 0], 'k-')
        # ax2.plot(all_yields, [successful_reactions[all_idx]]*len(all_yields), 'k-')
        ax2.scatter([y, y2], [successful_reactions[all_idx], successful_reactions[all_idx2]])
        ax2.annotate(f'{y:.2f}% yield\n{successful_reactions[all_idx]:.3f} reactions successful', (y, successful_reactions[all_idx]), textcoords="offset points", xytext=(0,10), ha='left')
        # ax2.plot([y2, y2], [successful_reactions[all_idx2], 0], 'k-')
        # ax2.plot(all_yields, [successful_reactions[all_idx2]]*len(all_yields), 'k-')
        ax2.annotate(f'{y2:.2f}% yield\n{successful_reactions[all_idx2]:.3f} reactions successful', (y2, successful_reactions[all_idx2]), textcoords="offset points", xytext=(0,10), ha='left')
        ax2.set_xlabel('Yield')
        ax2.set_ylabel('Successful Reactions')
        ax2.set_title('Successful Reactions vs Yield')
        

    def plot_set_coverage_size_2(self, cutoff:float)->None:
        coverages = []
        for size in range(1, 5):
            conditions = itertools.combinations(self.all_conditions, size)
            coverages.append([self.yield_surface.count_coverage(condition, cutoff) for condition in conditions])
        plt.hist(coverages, bins=20, histtype='bar', label=[f"Sets of size {size}" for size in range(1, 5)], stacked=False, log = True)
        plt.xlabel('Coverage')
        plt.ylabel('Number of Sets')
        plt.legend()
        plt.title('Reaction Space Coverage Distribution over Condition Sets')

