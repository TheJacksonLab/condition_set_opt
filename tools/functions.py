import os
import pickle
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import seaborn as sns
from chemical_space import ChemicalSpace

def merge_metrics(metric_filepaths:list, output_dir:str=None)->None:
    '''Combines metrics from multiple files with the same config into one file
    @params:
    metric_filepaths: list of filepaths to the metrics files to merge, each file should have a corresponding config file (named by replacing 'metrics' with 'config' in the file path)
    output_dir: directory to save the merged metrics file, if None, saves in the same directory as the first metric file
    '''
    assert len(metric_filepaths) > 1, 'Need more than one metric file to merge'
    config_files = [fp.replace('metrics_', 'config_') for fp in metric_filepaths]
    print(config_files[0])
    with open(config_files[0], 'rb') as f:
        conf = pickle.load(f)
    for cfg_fp in config_files[1:]:
        with open(cfg_fp, 'rb') as f:
            conf_check = pickle.load(f)
            for key in conf.keys() - ['date']:
                print('checking', key)
                assert conf[key] == conf_check[key], f'{key} does not match in config files {cfg_fp} and {config_files[0]}'
    if not output_dir:
        output_dir = '/'.join(metric_filepaths[0].split('/')[:-2])+f'/batch_size={conf["batch_size"]}_cutoff={conf["cutoff"]}_max_set_size={conf["max_set_size"]}_learner={conf["learner_type"]}'
    os.makedirs(output_dir, exist_ok=True)
    print(f'writing to {output_dir}')
    metrics = {}
    opt_count = 0
    for metric_file in metric_filepaths:
        with open(metric_file, 'rb') as f:
            m = pickle.load(f)
            for key in m.keys():
                metrics[opt_count] = m[key]
                opt_count += 1
    with open(f'{output_dir}/metrics_.pkl', 'wb') as f:
        pickle.dump(metrics, f)
    with open(f'{output_dir}/config_.pkl', 'wb') as f:
        pickle.dump(conf, f)

def read_files(files:list)->None:
    '''Outputs key information about config and metrics files
    @params:
    files: list of filepaths to the metrics files to read, each file should have a corresponding config file (named by replacing 'metrics' with 'config' in the file path)
    For each file prints out: model_type, learner_type, batch_size, max_set_size, stochastic_cond_num: (number of optimization runs) file_path'''
    for file in files:
        with open(file.replace('metrics_', 'config_'),'rb') as f:
            conf = pickle.load(f)
        with open(file,'rb') as f:
            try:
                metrics = pickle.load(f)
                print(f"{conf.get('model_type', 'GP')}, {conf['learner_type']}, {conf['batch_size']}, {conf['max_set_size']}, {conf.get('stochastic_cond_num', None)}: ({len(metrics)}) {file}")
            except:
                print(f"{conf['model_type']}, {conf['learner_type']}, {conf['batch_size']}, {conf['max_set_size']}, {conf['stochastic_cond_num']}: Error loading metrics")
                continue

def create_batch_dict(files:list)->dict:
    '''Creates a dictionary of the files organized by the batch size and learner type of each file
    @params:
    files: list of filepaths to the metrics files to read
    each file should have a corresponding config file (named by replacing 'metrics' with 'config' in the file path) and each file should have a dfferent combination of batch size and learner type
    @returns:
    batch_dict: dictionary of the file paths organized by the batch size and learner type of each file by batch_dict[batch_size][learner_type] = file_path
    '''
    idx_map = {-1:0, 0:1, 9:2, 7:3}
    batch_dict = {1:['rand','explore','combined*','exploit*'],
                    10:['rand','explore','combined*','exploit*'],
                    20:['rand','explore','combined*','exploit*'],
                    40:['rand','explore','combined*','exploit*'],
                    80:['rand','explore','combined*','exploit*'],
                    160:['rand','explore','combined*','exploit*']}
    for file in files:
        with open(file.replace('metrics_', 'config_'),'rb') as f:
            conf = pickle.load(f)
        try:
            assert conf['batch_size'] in batch_dict.keys()
            assert conf['model_type'] == 'RF'
            assert conf['stochastic_cond_num'] is None
            assert conf['learner_type'] in idx_map.keys()
        except:
            continue
        with open(file,'rb') as f:
            try:
                metrics = pickle.load(f)
                print(f"{conf['model_type']}, {conf['learner_type']}, {conf['batch_size']}, {conf['max_set_size']}, {conf['stochastic_cond_num']}: ({len(metrics)}) {file}")
            except:
                print(f"{conf['model_type']}, {conf['learner_type']}, {conf['batch_size']}, {conf['max_set_size']}, {conf['stochastic_cond_num']}: Error loading metrics")
                continue
        batch_dict[conf['batch_size']][idx_map[conf['learner_type']]] = file
    return batch_dict

def compare_spaces(spaces:list[ChemicalSpace])->None:
    '''
    Plots the yield vs coverage for each space in the list and yield vs gap in individual and total coverage
    @params:
    spaces: list of ChemicalSpace objects to compare
    '''
    colors = ['#FF1F5B', '#009ADE', '#AF58BA', '#FFC61E', '#F28522']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6)) #ax1, ax2, ax0, ax3
    for n, cs in enumerate(spaces):
        yields, coverage = cs.get_yield_coverage()
        # all_yields, successful_rxns = cs.get_yield_success()
        cond_yield, cond_coverage = cs.get_individual_conditions_coverage()
        diff = [0]*(len(yields)*2)
        diff_yield = [0]*(len(yields)*2)
        diff_total_cov = [0]*(len(yields)*2)
        j, k = 0, 0
        for i in range(len(diff)):
            if j >= len(cond_yield) or (k < len(yields) and yields[k] > cond_yield[j]):
                diff_yield[i] = yields[k]
                diff[i] = coverage[k] - cond_coverage[j-1]
                diff_total_cov[i] = coverage[k]
                k += 1
            else:
                diff_yield[i] = cond_yield[j]
                if k >= len(yields):
                    diff[i] = 1 - cond_coverage[j]
                    diff_total_cov[i] = 1
                else:
                    diff[i] = coverage[k] - cond_coverage[j]
                    diff_total_cov[i] = coverage[k]
                j += 1
        # ax0.plot(diff_yield, diff, label=cs.dataset_name)
        ax1.plot(yields, np.array(coverage) * 100, label=cs.dataset_name, color = colors[n])
        ax1.plot(cond_yield, np.array(cond_coverage) * 100, label=cs.dataset_name, linestyle='dashed', color = colors[n])
        ax2.plot(diff_yield, np.array(diff) *100, label=cs.dataset_name, color = colors[n])
    ax1.set_xlabel('Yield Cutoff', fontsize=20)
    ax1.set_ylabel('Coverage of Reactant Space (%)', fontsize=20)
    dashed_patch = mlines.Line2D([],[],color='black', linestyle='dashed', label='Top Individual Condition')
    solid_patch = mlines.Line2D([],[],color='black', label='All Conditions')
    ax1.legend(handles=[solid_patch, dashed_patch], loc='lower left',fontsize=13)
    ax2.set_xlabel('Yield Cutoff', fontsize=20)
    ax2.set_ylabel('Individual Condition Coverage Gap (Δ)', fontsize=20)
    ax2.legend(fontsize=13, loc='upper left')
    ax1.tick_params(axis='both', which='major', labelsize=13)
    ax2.tick_params(axis='both', which='major', labelsize=13)
    ax1.annotate('',
        xytext=(62, 70),
        xy=(62, 90),
        arrowprops=dict(arrowstyle="->", color='black'),
        # size=size
    )
    ax1.annotate('Δ',
        xytext=(64, 77),
        xy=(64, 77)
    )
    plt.show()

def plot_learner_metrics(chemical_space:ChemicalSpace, metric_filepaths:list, names:list, ax:plt.Axes, title:str, show_legend:bool=True, colors:list|None=None, max_possible_set_cover:int=None)->None:
    '''
    Plots the actual coverage of the best predicted sets over multiple iterations on given axes
    @params:
    chemical_space: ChemicalSpace object
    metric_filepaths: list of filepaths to the metrics files to read
    names: list of names for the learner
    ax: matplotlib axes object
    title: title for the plot
    show_legend: boolean to show the legend
    colors: list of colors for each learner
    max_possible_set_cover: maximum possible coverage of the best
    @returns:
    colors: list of colors used in the plot
    '''
    assert len(metric_filepaths) == len(names), 'Number of filepaths and names must be the same'
    pred_covs = [f'Predicted Coverage' for i in range(len(names))]
    actual_covs = [f'Actual Coverage' for i in range(len(names))]
    config_file = metric_filepaths[0].replace('metrics_', 'config_')
    with open(config_file, 'rb') as f:
        try:
            config = pickle.load(f)
        except:
            print(config_file)
        max_set_size = config['max_set_size']
        yield_cutoff = config['cutoff']
    
    df = pd.DataFrame(columns=["Fraction of Space Tested", 'Predicted Coverage', 'Actual Coverage', 'Accuracy', 'Precision', 'Recall', 'name', 'pred_name', 'actual_name'])
    m = {}
    for i, metric_file in enumerate(metric_filepaths):
        with open(metric_file, 'rb') as f:
            try:
                m = pickle.load(f)
            except:
                print(metric_file)
        conf_file = metric_file.replace('metrics_', 'config_')
        with open(conf_file, 'rb') as f:
            try:
                conf = pickle.load(f)
            except:
                print(config_file)
            batch_size = conf['batch_size']
            evals = np.divide([49 + i*batch_size for i in range(0, 1 + conf['max_experiments']//batch_size)], np.prod(chemical_space.shape)) *100
        for j, metric in m.items():
            coverages = np.array(metric['coverages'])
            best_sets = metric['best_sets']
            best_3 = coverages[:,0]
            if np.any(best_3 > 1):
                best_3 = best_3/np.prod(chemical_space.shape[chemical_space.conditions_dim:])
            best_pred_sets = [s[0] for s in best_sets]
            best3_actual = [chemical_space.yield_surface.count_coverage(set, yield_cutoff) * 100 for set in best_pred_sets]
            iter_df = pd.DataFrame({'Fraction of Space Tested':evals[:len(best_3)], 'Predicted Coverage':best_3, 'Actual Coverage': best3_actual, 'Accuracy':metric['accuracy'], 'Precision':metric['precision'], 'Recall':metric['recall'], 'name':[names[i] for _ in range(len(best_3))], 'pred_name':[pred_covs[i] for _ in range(len(best_3))], 'actual_name':[actual_covs[i] for _ in range(len(best_3))]})
            iter_df.fillna(0, inplace=True)
            if df.empty:
                df  = iter_df
            else:
                df = pd.concat([df, iter_df])
    
    
    cov_labels = [""]*(len(pred_covs)*2)
    for i in range(len(pred_covs)):
        cov_labels[i*2] = pred_covs[i]
        cov_labels[i*2+1] = actual_covs[i]
    print(df.index.is_unique)
    # g = sns.lineplot(data=df, x='Fraction of Space Tested', y='Predicted Coverage', hue='pred_name', hue_order=pred_covs, ax=ax1)
    print(ax)#palette=colors, 
    palette = "tab10"
    if colors is not None:
        matplotlib.colormaps.register(ListedColormap(colors))
        palette = "from_list"
    g = sns.lineplot(data=df, x='Fraction of Space Tested', y='Actual Coverage', hue='name', hue_order=names, palette='from_list', ax=ax)
    g.hlines(y=[chemical_space.max_possible_coverage(yield_cutoff)*100], xmin=df['Fraction of Space Tested'].min(), xmax=df['Fraction of Space Tested'].max(), colors='black', linestyles='solid')
    if max_possible_set_cover:
        g.hlines(y=[max_possible_set_cover*100], xmin=df['Fraction of Space Tested'].min(), xmax=df['Fraction of Space Tested'].max(), colors='gray', linestyles='solid')
    g.hlines(y=[chemical_space.best_condition_sets(yield_cutoff, 1, 1)[0]['coverage']*100], xmin=df['Fraction of Space Tested'].min(), xmax=df['Fraction of Space Tested'].max(), colors='black', linestyles='dashed')
    
    if colors is not None:
        matplotlib.colormaps.unregister('from_list')
    # g2 = sns.lineplot(data=df, x='Fraction of Space Tested', y='Rank', hue='name',palette=colors, ax=ax2)
    # g2.set_ylim(0, 110)
    ax.set_xlabel('Fraction of Space Tested (%)', fontsize=16)
    ax.set_ylabel('Coverage of Reactant Space (%)', fontsize=16)
    ax_handles, ax_labels = ax.get_legend_handles_labels()

    # loc='center left', bbox_to_anchor=(1, 0.5)
    ax.legend(handles=ax_handles, labels=ax_labels, fontsize=14).set_visible(show_legend)
    colors = [handle._color for handle in g.legend_.legend_handles]
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_title(title, loc='left', fontsize=20)

    return colors

def plot_performance(chemical_spaces:list[ChemicalSpace], metric_filepaths:list, names:list, titles:list, colors=None, max_possible_set_cover=None, legend_ax=0, sharey = False)->None:
    '''
    Plots the coverage of the best predicted sets over multiple iterations for each chemical space
    @params:
    chemical_spaces: list of ChemicalSpace objects
    metric_filepaths: list of filepaths to the metrics files to read
    names: list of names for the learner
    titles: list of titles for the plots
    colors: list of colors for each learner
    max_possible_set_cover: maximum possible coverage of sets of the max set size
    legend_ax: index of the axis to show the legend
    sharey: boolean to share the y axis between plots
    '''
    fig_h = int((len(chemical_spaces) +1 )/ 2)
    fig, axs = plt.subplots(fig_h, 2, figsize=(10, fig_h*5), sharey=sharey)
    axs = axs.flatten()
    for i, cs in enumerate(chemical_spaces):
        plot_learner_metrics(cs, metric_filepaths[i], names[i], colors=colors[i], ax=axs[i], title=titles[i], show_legend=(i==legend_ax), max_possible_set_cover=max_possible_set_cover[i])
    if len(chemical_spaces) % 2 == 1:
        axs[-1].remove()
    plt.tight_layout()

def plot_model_perf(chemical_space:ChemicalSpace, metric_filepaths:list, names:list, colors:list, title:str)->list:
    '''Plots the accuracy, precision, recall of the models over multiple iterations
    @params:
    chemical_space: ChemicalSpace optimization runs were performed on
    metric_filepaths: list of filepaths to the metrics files to read (each file should have a corresponding config file (named by replacing 'metrics' with 'config' in the file path))
    names: list of names to use as titles for each filepath in metric_filepaths
    colors: list of colors to use for the results of each filepath in metric_filepaths
    title: title for the plot
    @returns:
    colors: list of colors used in the plot
    '''
    assert len(metric_filepaths) == len(names), 'Number of filepaths and names must be the same'
    config_file = metric_filepaths[0].replace('metrics_', 'config_')
    with open(config_file, 'rb') as f:
        config = pickle.load(f)
        max_set_size = config['max_set_size']
        yield_cutoff = config['cutoff']
    cond_to_rank_map = chemical_space.yield_surface.rank_conditions(chemical_space.all_conditions, max_set_size, yield_cutoff)
    df = pd.DataFrame(columns=["Fraction of Space Tested", 'Predicted Coverage', 'Actual Coverage', 'Rank', 'Accuracy', 'Precision', 'Recall', 'name', 'pred_name', 'actual_name'])
    m = {}
    for i, metric_file in enumerate(metric_filepaths):
        with open(metric_file, 'rb') as f:
            m = pickle.load(f)
        conf_file = metric_file.replace('metrics_', 'config_')
        with open(conf_file, 'rb') as f:
            conf = pickle.load(f)
            batch_size = conf['batch_size']
            evals = np.divide([49 + i*batch_size for i in range(0, 1 + conf['max_experiments']//batch_size)], np.prod(chemical_space.shape))
        for j, metric in m.items():
            coverages = np.array(metric['coverages'])
            best_sets = metric['best_sets']
            best_3 = coverages[:,0]
            if np.any(best_3 > 1):
                best_3 = best_3/np.prod(chemical_space.shape[chemical_space.conditions_dim:])
            best_pred_sets = [s[0] for s in best_sets]
            best3_actual = [chemical_space.yield_surface.count_coverage(set, yield_cutoff) for set in best_pred_sets]
            set_ranks = [cond_to_rank_map[set] for set in best_pred_sets]
            iter_df = pd.DataFrame({'Fraction of Space Tested':evals[:len(best_3)], 'Predicted Coverage':best_3, 'Actual Coverage': best3_actual, 'Rank':set_ranks, 'Accuracy':metric['accuracy'], 'Precision':metric['precision'], 'Recall':metric['recall'], 'name':[names[i] for _ in range(len(best_3))]})
            iter_df.fillna(0, inplace=True)
            if df.empty:
                df  = iter_df
            else:
                df = pd.concat([df, iter_df])
    
    fig2, (ax3, ax4, ax5) = plt.subplots(1, 3, figsize=(15,5))
    sns.lineplot(data=df, x='Fraction of Space Tested', y='Accuracy',
        hue='name',
        hue_order=names,
        ax=ax3)
    sns.lineplot(data=df, x='Fraction of Space Tested', y='Precision', 
        hue='name',
        hue_order=names,
        ax=ax4)
    sns.lineplot(data=df, x='Fraction of Space Tested', y='Recall', 
        hue='name',
        hue_order=names,
        ax=ax5)
    ax3.set_title('Accuracy', fontsize=15)
    ax3.set_xlabel('Fraction of Space Tested', fontsize=15)
    ax4.set_title('Precision', fontsize=15)
    ax4.set_xlabel('Fraction of Space Tested', fontsize=15)
    ax5.set_title('Recall', fontsize=15)
    ax5.set_xlabel('Fraction of Space Tested', fontsize=15)
    ax3.legend().set_visible(True)
    for ax in [ax3, ax4, ax5]:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels)
    fig2.suptitle(title, fontsize=20)
    return colors

def create_batch_df(chemical_space:ChemicalSpace, batch_dict:dict, learner:int)->pd.DataFrame:
    '''Creates a dataframe of the performance metrics for all the optimization runs in batch_dict
    @params:
    chemical_space: ChemicalSpace object optimization runs were performed on
    batch_dict: dictionary of the file paths organized by the batch size and learner type of each file by batch_dict[batch_size][learner_type] = file_path
    learner: index of the learner type to put into the dataframe
    '''
    df = pd.DataFrame(columns=["Fraction of Space Tested", 'Predicted Coverage', 'Actual Coverage', 'Accuracy', 'Precision', 'Recall', 'Batch Size'])
    m = {}
    for batch_size in batch_dict.keys():
        metric_file = batch_dict[batch_size][learner]
        with open(metric_file, 'rb') as f:
            try:
                m = pickle.load(f)
            except:
                print(metric_file)
        conf_file = metric_file.replace('metrics_', 'config_')
        with open(conf_file, 'rb') as f:
            try:
                conf = pickle.load(f)
            except:
                print(conf_file)
            yield_cutoff = conf['cutoff']
            evals = np.divide([49 + i*batch_size for i in range(0, 1 + conf['max_experiments']//batch_size)], np.prod(chemical_space.shape)) *100
        for j, metric in m.items():
            coverages = np.array(metric['coverages'])
            best_sets = metric['best_sets']
            best = coverages[:,0]
            if np.any(best > 1):
                best = best/np.prod(chemical_space.shape[chemical_space.conditions_dim:])
            best_pred_sets = [s[0] for s in best_sets]
            best_actual = [chemical_space.yield_surface.count_coverage(set, yield_cutoff) * 100 for set in best_pred_sets]
            iter_df = pd.DataFrame({'Fraction of Space Tested':evals[:len(best)], 'Predicted Coverage':best, 'Actual Coverage': best_actual, 'Accuracy':metric['accuracy'], 'Precision':metric['precision'], 'Recall':metric['recall'], 
                                'Batch Size': [batch_size] * len(best)})
            iter_df.fillna(0, inplace=True)
            if df.empty:
                df  = iter_df
            else:
                df = pd.concat([df, iter_df])
    return df

def plot_batch_performance(chemical_space:ChemicalSpace, batch_dict:dict, titles:list= ['Random Selection', 'Explore', 'Combined', 'Exploit'], max_possible_set_cover:int|None=None, yield_cutoff:float=.75)->None:
    '''
    plots optimization performance across batch sizes for each learner type
    @params:
    chemical_space: ChemicalSpace object optimization runs were performed on
    batch_dict: dictionary of the file paths organized by the batch size and learner type of each file by batch_dict[batch_size][learner_type] = file_path
    titles: list of titles for the plots
    max_possible_set_cover: maximum possible coverage of sets of the max set size
    yield_cutoff: yield cutoff to use for coverage calculations
    '''
    fig, axs = plt.subplots(1, 4, figsize=(20, 5), sharey=True) #(13, 10)
    axs = axs.flatten()

    for i in range(4):
        df = create_batch_df(chemical_space, batch_dict, i)
        g = sns.lineplot(data=df, x='Fraction of Space Tested', y='Actual Coverage', hue='Batch Size', hue_order=[1, 10, 20, 40, 80, 160], palette='colorblind', ax=axs[i])
        g.hlines(y=[chemical_space.max_possible_coverage(yield_cutoff)*100], xmin=df['Fraction of Space Tested'].min(), xmax=df['Fraction of Space Tested'].max(), colors='black', linestyles='solid')
        if max_possible_set_cover:
            g.hlines(y=[max_possible_set_cover*100], xmin=df['Fraction of Space Tested'].min(), xmax=df['Fraction of Space Tested'].max(), colors='gray', linestyles='solid')
        g.hlines(y=[chemical_space.best_condition_sets(yield_cutoff, 1, 1)[0]['coverage']*100], xmin=df['Fraction of Space Tested'].min(), xmax=df['Fraction of Space Tested'].max(), colors='black', linestyles='dashed')

        axs[i].set_xlabel('Fraction of Space Tested (%)', fontsize=16)
        axs[i].set_ylabel('Coverage of Reactant Space (%)', fontsize=16)
        ax_handles, ax_labels = axs[i].get_legend_handles_labels()

        axs[i].legend(handles=ax_handles, labels=ax_labels, fontsize=14).set_visible((i==0))
        axs[i].tick_params(axis='both', which='major', labelsize=14)
        axs[i].set_title(titles[i], loc='left', fontsize=20)

    plt.tight_layout()

def sort_by_learner_type(files:list)->list:
    '''Sorts the files by learner type
    @params:
    files: list of filepaths to the metrics files to read
    each file should have a corresponding config file (named by replacing 'metrics' with 'config' in the file path)
    @returns:
    sorted_files: list of filepaths sorted by learner type where sorted_files[learner_type] = file_path'''
    sorted_files = [""]*10
    for file in files:
        config_file = file.replace('metrics_', 'config_')
        with open (config_file, 'rb') as f:
            config = pickle.load(f)
            sorted_files[config['learner_type'] + 1] = file
            print(config['learner_type'])
        with open(file, 'rb') as f:
            metrics = pickle.load(f)
            print(len(metrics))
    return sorted_files