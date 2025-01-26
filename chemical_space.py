import matplotlib as mpl
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score
from space_mat import SpaceMatrix
from space_mat import THRESHOLDED_COUNT
from matplotlib.colors import ListedColormap
from tools.featurize import convert_to_onehot
import rdkit.Chem as Chem

class ChemicalSpace:
    def __init__(self, condition_titles:list, reactant_titles:list, data_file:str, target_title='yield', condition_parameter_subspace={}, titles_to_fingerprint=[]) -> None:
        '''
        @params:
        condition_titles: list of strings, the headers of the condition components in the data file ex. ['ligand', 'solvent', 'temperature']
        reactant_titles: list of strings, the headers of the reactants in the data file ex. ['electrophile', 'nucleophile']
        data_file: string, the path to the csv file that contains the data
        target_title: string, the header of the target column in the data file (default is 'yield')
        fingerprint_file: string, the path to the csv file that contains the fingerprint data 

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
        self.yield_surface = SpaceMatrix(self._create_mat(data_file, condition_titles, reactant_titles, target_title, condition_parameter_subspace))
        self.features = []
        for i, title in enumerate(self.titles):
            if title in titles_to_fingerprint:
                self.features.append([np.array(Chem.RDKFingerprint(Chem.MolFromSmiles(x))) for x in self.labels[i]])
            else:
                self.features.append(np.diag(np.ones(len(self.labels[i]))))

        self.shape = self.yield_surface.shape
        self.all_points = list(itertools.product(*[range(s) for s in self.shape]))
        self.all_points_featurized = [self.create_feature_vector(point) for point in self.all_points]
        self._y_true = np.array([self.yield_surface[point] for point in self.all_points])
        self.all_conditions = list(itertools.product(*[range(s) for s in self.shape[:self.conditions_dim]]))

    def _create_mat(self, data_file, cond_titles:list, reactant_titles:list,  target_title, condition_params_default)->np.ndarray:
        # Create a matrix of the data
        data = pd.read_csv(data_file)
        unique_rs = [data[r_title].unique() for r_title in reactant_titles]
        unique_conds = [data[cond_title].unique() for cond_title in cond_titles]
        self.titles = cond_titles + reactant_titles
        for param in condition_params_default:
            values = condition_params_default[param]
            idx = cond_titles.index(param)
            unique_conds[idx] = np.array(values)
            data = data.loc[data[param].isin(values)]
        self.labels = unique_conds + unique_rs
        shape = [len(item) for item in self.labels]
        data_mat = np.zeros(tuple(shape))
        for idx, series in data.iterrows():
            point = [np.where(self.labels[i] == (series[title])) for i, title in enumerate(self.titles)]
            # restrict the yields to be between 0 and 100
            data_mat[tuple(point)] = max(min(series[target_title], 100), 0)
        return data_mat
    
    def create_feature_vector(self, point):
        '''Converts a point in the chemical space with the associated fingerprint features, and the rest OHE'''
        result = []
        for i in range(len(point)):
            result.extend(self.features[i][point[i]])
        return result

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
        y_true = self._y_true >= cutoff
        return accuracy_score(y_true, y_pred), precision_score(y_true, y_pred), recall_score(y_true, y_pred)

    def best_condition_sets(self, yield_threshold:float, max_set_size:int=1, num_sets:int=10) -> np.ndarray:
        '''
        @params:
        yield_threshold: float, the maximum yield to count as a failed reaction
        max_set_size: int, the maximum number of conditions to include in a set
        num_sets: int, the number of sets to return
        
        returns:
        np.ndarray(type=('condition set', 'coverage')), the best condition sets and their corresponding scores
        '''
        return self.yield_surface.best_condition_sets(self.all_conditions, THRESHOLDED_COUNT(np.prod(self.shape[self.conditions_dim:]))(yield_threshold), max_set_size, num_sets)
    
    def format_condition(self, condition:tuple) -> str:
        '''
        @params:
        condition: tuple of integers, the indecies of the condition components
        
        returns:
        string, the formatted condition using the condition labels'''
        return ', '.join([f'{self.labels[i][c]}' for i, c in enumerate(condition)])
    
    def max_possible_coverage(self, cutoff:float)->int:
        return self.yield_surface.count_coverage(self.all_conditions, cutoff)

    def get_yield_coverage(self)->tuple:
        max_yield_surface = np.amax(self.yield_surface.mat.T, axis=tuple(range(self.reactants_dim, len(self.shape))))
        yields = np.sort(max_yield_surface.flatten())[::-1]
        coverages = range(1, len(yields) + 1)/(np.prod(self.shape[self.conditions_dim:]))
        for i in range(2, len(yields)+1):
            if yields[1-i] == yields[-i]:
                coverages[-i] = coverages[1-i]
        return yields, coverages
    
    def get_yield_success(self)->tuple:
        all_yields = np.sort(self.yield_surface.mat.flatten())[::-1]
        successful_reactions = range(1, len(all_yields) + 1)/(np.prod(self.shape))
        return all_yields, successful_reactions
    
    def get_individual_conditions_coverage(self)->tuple:
        cond_surfaces = [np.sort(self.yield_surface.get_condition_surface(cond).flatten())[::-1] for cond in self.all_conditions]
        cond_cov_max_yields = np.amax(cond_surfaces, axis=0)
        coverage = range(1, len(cond_cov_max_yields) + 1)/(np.prod(self.shape[self.conditions_dim:]))
        for i in range(2,len(cond_cov_max_yields)+1):
            if cond_cov_max_yields[-i] == cond_cov_max_yields[1-i]:
                coverage[-i] = coverage[1-i]
        return cond_cov_max_yields, coverage

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
        if len(conditions) == 1:
            condition_str = f"Condition {conditions[0]} Coverage"
        else:
            condition_str = ', '.join([f"{c}" for c in conditions])
            condition_str = f"Conditions {condition_str} Coverage"
        plt.title(f'{condition_str}', fontsize=20)
        plt.savefig(file_name)

    def plot_max_surface(self, zlabel = '% Yield', title='Highest Yield Across All Conditions') -> None:    
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
    
    def plot_set_coverages(self, set_sizes:list, cutoff:float, show_legend = True, font_size=15, bins=None, xmin=None)->None:
        '''Plot histogram of condition set coverages across different condition set sizes'''
        coverages = []
        plt.rcParams['font.family'] = "sans-serif"
        plt.rcParams['font.sans-serif'] = "Arial"
        for size in set_sizes:
            conditions = itertools.combinations(self.all_conditions, size)
            covs = [self.yield_surface.count_coverage(condition, cutoff)*100 for condition in conditions]
            print(f"{size} max cov: {np.array(covs).max()}")
            coverages.append(covs) 
        # create custom matplot lib color map
        cmap = ListedColormap(['#FF1F5B', '#009ADE', '#AF58BA', '#FFC61E', '#F28522', '#00CD6C'])
        plt.set_cmap(cmap)
        print(coverages)
        if bins is None:
            bins = np.prod(self.shape[self.conditions_dim:])
        if xmin:
            coverages = [[n for n in c if n >= xmin] for c in coverages]
        plt.hist(coverages, bins=bins, histtype='bar', label=[f"{size}" for size in set_sizes], stacked=False, log = True, color=['#FF1F5B', '#009ADE', '#AF58BA', '#FFC61E', '#F28522', '#00CD6C'][:len(set_sizes)])
        plt.xlabel('Coverage of Reactant Space (%)', fontsize=font_size)
        plt.ylabel('Number of Sets', fontsize=font_size)
        plt.tick_params(axis='both', which='major', labelsize=13)
        if show_legend:
            plt.legend(title='Number of Reaction \nConditions in a Set', loc='center left', bbox_to_anchor=(1, 0.5),  fontsize=11, title_fontsize=13)
        plt.savefig(f"./datasets/{self.dataset_name}_histogram.png")
        plt.show(block=True)

    def plot_condition_pair_coverages(self, cutoff:float, vmin=None, vmax = None, font_size=15)->None:
        ax1:plt.Axes
        fig, (ax1) = plt.subplots(1, 1, figsize=(5, 5))
        combination_mat = np.zeros((len(self.all_conditions), len(self.all_conditions)))
        ordered_conds = self.best_condition_sets(cutoff, 1, None)
        for c in range(len(ordered_conds)):
            for c2 in range(c, len(ordered_conds)):
                combination_mat[c][c2] = self.yield_surface.count_coverage((ordered_conds[c]['set'][0], ordered_conds[c2]['set'][0]), cutoff)
                combination_mat[c2][c] = combination_mat[c][c2]
        ax1.imshow(combination_mat, origin='lower')
        ax1.set_xlabel('Ranked Individual Conditions')
        ax1.set_ylabel('Ranked Individual Conditions')
        ax1.set_title('Combinations of Individual Conditions')

    def plot_subset_overlap(self, set_size:int, cutoff:float, max_num_sets=None, vmin=100, vmax = 0, font_size = 15, cond_cutoff=None, cov_min=0, cov_max=75)->None:
        '''Plot of high coverage sets and the individual conditions that make them up'''
        fig = plt.figure(figsize=(6.5, 8.805))
        gs = fig.add_gridspec(1, 2, width_ratios=[3.6, 1])

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1], sharey=ax1) #

        if max_num_sets is None:
             max_num_sets = len(self.all_conditions)

        individual = self.best_condition_sets(cutoff, 1, None)
        
        conds_condensed = [cond[0] for cond in individual['set']]
        sets = self.best_condition_sets(cutoff, set_size, max_num_sets)
        # conditions over the cutoff number of individual conditions on the x axis
        conds_over_ys = []
        conds_over_txt = []
        if cond_cutoff is None:
            combination_mat = np.full((len(conds_condensed), max_num_sets), np.nan)
        else:
            combination_mat = np.full((cond_cutoff, max_num_sets), np.nan)
        f = lambda s: tuple(np.sort(np.array([-1*conds_condensed.index(c) for c in s]))[::-1])
        converted_sets = np.array([(f(s['set']), s['coverage'], -1*abs(s['size'])) for s in sets],dtype=[('set', 'O'), ('coverage', 'f4'),('size', 'i4')])
        converted_sets.sort(order=['coverage', 'size', 'set'])
        converted_sets = converted_sets[::-1]
        
        for i, set in enumerate(converted_sets['set']):
            for cond in set:
                idx = cond*-1
                if cond_cutoff is None or idx < cond_cutoff - 1:
                    combination_mat[idx][i] = individual[idx]['coverage']*100
                else:
                     conds_over_ys.append(i)
                     conds_over_txt.append(f'{idx+1}')
                     combination_mat[cond_cutoff - 1][i] = individual[idx]['coverage']*100

        sums = np.nansum(combination_mat, axis=1)
        trim_count = np.trim_zeros(sums, 'b')
        print(np.average(sets['size']))
        combination_mat = combination_mat[:len(trim_count) + 1]
        cmap = mpl.colormaps.get_cmap('viridis')
        cmap.set_bad('white', alpha=1)
        cmap.set_under('lightgray')
        cmap.name = 'vir_bad'
        try:
            plt.colormaps.register(cmap)
        except ValueError:
            pass
        plt.set_cmap(cmap)
        vmin = min(np.nanmin(combination_mat), vmin)
        vmax = max(sets['coverage'].max()*100, vmax)
        mat = ax1.imshow(combination_mat.T, aspect='auto', origin='upper', cmap=cmap, vmin=vmin, vmax=vmax)
        ax1.set_xlabel('Ranked Individual Conditions', fontsize=font_size)
        ax1.set_ylabel(f'Ranked Condition Sets', fontsize=font_size)
        ticks = [0]
        ticks.extend(np.arange(4,combination_mat.shape[0], 5))
        ax1.set_xticks(ticks)
        labels = [f"{i + 1}" for i in ticks[:-1]]
        if cond_cutoff is None:
            labels.append(f'{ticks[-1]+1}')
        else:
            labels.append(f'{cond_cutoff}+')
        ax1.set_xticklabels(labels)
        ticks = [0]
        ticks.extend(np.arange(4,combination_mat.shape[1], 5))
        ax1.set_yticks(ticks)
        labels = [i + 1 for i in ticks]
        ax1.set_yticklabels(labels)
        for i, txt in enumerate(conds_over_txt):
            ax1.annotate(txt, (cond_cutoff - 1, conds_over_ys[i]), color='white', fontsize=8, ha='center', va='center')
        print(vmin)
        print(vmax)
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        colors = cmap(norm(sets['coverage'] * 100))
        print(sets['coverage']*100)
        ax2.barh(range(len(sets)), sets['coverage']*100, color = colors)
        ax2.set_xlabel('Coverage (%)', fontsize=font_size)
        cax = plt.axes((0.05, 0.0, 0.9, 0.025))
        cbar = plt.colorbar(label='Coverage of Reactant Space (%)',cax= cax, ax=[ax1, ax2], mappable=mat, orientation="horizontal")
        cbar.ax.tick_params(labelsize=13)
        cbar.ax.set_xlabel('Coverage of Reactant Space (%)', fontsize=font_size)
        ax2.set_xbound(cov_min,cov_max)
        # create grid lines
        ax1.set_xticks(np.arange(-.5, combination_mat.shape[0], 1), minor=True)
        ax1.set_yticks(np.arange(-.5, combination_mat.shape[1], 1), minor=True)

        ax1.tick_params(which='minor', length=0)
        ax2.tick_params(which='minor', length=0)
        ax1.grid(color='lightgray', which='minor', linestyle='-', linewidth=0.5)
        ax1.tick_params(axis='both', which='major', labelsize=13)
        ax2.tick_params(axis='both', which='major', labelsize=13)
        dark_ticks = []
        for i in range(1, combination_mat.shape[1]):
            if sets[i]['coverage'] < sets[i-1]['coverage'] or sets[i]['size'] > sets[i-1]['size']:
                dark_ticks.append(i-.5)
                ax1.hlines(i-.5, xmin=-.5, xmax=combination_mat.shape[0] - .5, colors='black', linestyles='-', linewidth=.5)
        plt.savefig("./set_components.png")
        plt.show(block=True)
    
    def plot_coverage_over_yield(self, coverage_threshold = .6)->None:
        ax1:plt.Axes;ax2:plt.Axes
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        yields, coverages = self.get_yield_coverage()
        all_yields, successful_reactions = self.get_yield_success()
        cond_yields, cond_coverages = self.get_individual_conditions_coverage()
        idx = int(np.floor(coverage_threshold * (len(yields))))
        y = yields[idx]

        all_idx = len(all_yields) - np.searchsorted(all_yields[::-1], y, side='right')
        # plot of coverage vs yield
        ax1.plot(yields, coverages, cond_yields, cond_coverages)
        ax1.scatter([y], [coverages[idx]])
        ax1.annotate(f'{y:.2f}% yield,\n{coverages[idx]:.3f} coverage', (y, coverages[idx]), textcoords="offset points", xytext=(0,10), ha='left')
        ax1.set_xlabel('Yield')
        ax1.set_ylabel('Coverage')
        ax1.set_title('Coverage vs Yield')
        # plot of % of successful reactions vs yield
        ax2.plot(all_yields, successful_reactions)
        ax2.scatter([y], [successful_reactions[all_idx]])
        ax2.annotate(f'{y:.2f}% yield\n{successful_reactions[all_idx]:.3f} reactions successful', (y, successful_reactions[all_idx]), textcoords="offset points", xytext=(0,10), ha='left')
        ax2.set_xlabel('Yield')
        ax2.set_ylabel('Successful Reactions')
        ax2.set_title('Successful Reactions vs Yield')
        

