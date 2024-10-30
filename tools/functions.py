import os
import pickle
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from chemical_space import ChemicalSpace
import seaborn as sns

def convert_to_onehot(shape: tuple, point:list)-> list:
    '''
    @params
    point: in order of condition, reactant
    '''
    onehot = np.zeros(np.sum(shape))
    for i, p in enumerate(point):
        onehot[int(p + np.sum(shape[:i]))] = 1
    return onehot

def convert_from_onehot(shape: tuple, onehot: int)-> list:
    '''
    returns
    point: in order of condition, reactant
    '''
    point = []
    num = 0
    shape_counter = 0
    for i in range(len(onehot)):
        if onehot[i] == 1:
            point.append(num)
        num += 1
        if num >= shape[shape_counter]:
            num = 0
            shape_counter += 1
    return point

def convert_point_to_idx(shape:tuple, point:list)-> int:
    '''
    @params
    point: in order of condition, reactant
    '''
    idx = 0
    for i, n in enumerate(point):
        idx += n
        if i < len(shape) - 1:
            idx *= shape[i+1]
    return idx

def convert_idx_to_point(shape:tuple, idx:int)-> list:
    '''
    returns:
    point: in order of condition, reactant
    '''
    point = [0]*len(shape)
    for i in range(len(shape)-1, -1, -1):
        point[i] = (idx % shape[i])
        idx = idx // shape[i]
    return point

def merge_metrics(metric_filepaths:list, output_dir:str=None)->None:
    '''Combines metrics from multiple files with the same config into one file'''
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

def read_files(files):
    '''Outputs key information about config and metrics files'''
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

def create_batch_dict(files):
    idx_map = {-1:0, 0:1, 6:2, 7:3}
    batch_dict = {1:['rand','explore','explore-exploit','exploit'],
                    10:['rand','explore','explore-exploit','exploit'],
                    20:['rand','explore','explore-exploit','exploit'],
                    40:['rand','explore','explore-exploit','exploit'],
                    80:['rand','explore','explore-exploit','exploit'],
                    160:['rand','explore','explore-exploit','exploit']}
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
    fig, (ax1, ax2, ax0, ax3) = plt.subplots(1, 4, figsize=(20, 5)) #ax1, ax2, ax0, ax3
    for cs in spaces:
        yields, coverage = cs.get_yield_coverage()
        all_yields, successful_rxns = cs.get_yield_success()
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
        ax0.plot(diff_yield, diff, label=cs.dataset_name)
        ax1.plot(yields, coverage, label=cs.dataset_name)
        ax2.plot(all_yields, successful_rxns, label=cs.dataset_name)
        ax3.plot(diff_total_cov, diff, label=cs.dataset_name)
    ax3.set_xlabel('Total Coverage', fontsize=13)
    ax3.set_ylabel('Gap in Individual and Max Coverage', fontsize=13)
    ax3.set_title('Total Coverage vs Difference', fontsize=13)
    ax0.set_xlabel('Yield Cutoff', fontsize=13)
    ax0.set_ylabel('Gap in Individual and Max Coverage', fontsize=13)
    ax0.set_title('Yield vs Coverage Difference', fontsize=13)
    ax1.set_xlabel('Yield Cutoff', fontsize=13)
    ax1.set_ylabel('Max Coverage', fontsize=13)
    ax1.legend()
    ax1.set_title('Yield vs Coverage', fontsize=13)
    ax2.set_xlabel('Yield Cutoff', fontsize=13)
    ax2.set_ylabel('Successful Reactions', fontsize=13)
    ax2.set_title('Yield vs Successful Reactions', fontsize=13)
    plt.show()

def plot_learner_metrics(chemical_space:ChemicalSpace, metric_filepaths:list, names:list, abbreviations:list=None, output_file_path:str=None, show_legend=True, colors=None)->None:
    assert len(metric_filepaths) == len(names), 'Number of filepaths and names must be the same'
    if np.all(not abbreviations):
        abbreviations = names
    pred_covs = [f'{abbreviations[i]} Predicted Coverage' for i in range(len(names))]
    actual_covs = [f'{abbreviations[i]} Actual Coverage' for i in range(len(names))]
    config_file = metric_filepaths[0].replace('metrics_', 'config_')
    with open(config_file, 'rb') as f:
        config = pickle.load(f)
        max_set_size = config['max_set_size']
        yield_cutoff = config['cutoff']
    cond_to_rank_map = chemical_space.yield_surface.rank_conditions(chemical_space.all_conditions, max_set_size, yield_cutoff)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    df = pd.DataFrame(columns=["Fraction of Space Tested", 'Predicted Coverage', 'Actual Coverage', 'Rank', 'Accuracy', 'Precision', 'Recall', 'name', 'pred_name', 'actual_name'])
    # accuracy, precision, recall = [], [], []
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
            
            # accuracy.append(metric['accuracy'])
            # precision.append(metric['precision'])
            # recall.append(metric['recall'])
            coverages = np.array(metric['coverages'])
            best_sets = metric['best_sets']
            best_3 = coverages[:,0]
            if np.any(best_3 > 1):
                best_3 = best_3/np.prod(chemical_space.shape[chemical_space.conditions_dim:])
            best_pred_sets = [s[0] for s in best_sets]
            best3_actual = [chemical_space.yield_surface.count_coverage(set, yield_cutoff) for set in best_pred_sets]
            # xs = np.arange(len(best_3))
            set_ranks = [cond_to_rank_map[set] for set in best_pred_sets]
            # append to df
            # print(f'label {names[i]}')
            # print(len(evals[:len(best_3)]), len(best_3), len(best3_actual), len(set_ranks), len(metric['accuracy']), len(metric['precision']), len(metric['recall']))
            iter_df = pd.DataFrame({'Fraction of Space Tested':evals[:len(best_3)], 'Predicted Coverage':best_3, 'Actual Coverage': best3_actual, 'Rank':set_ranks, 'Accuracy':metric['accuracy'], 'Precision':metric['precision'], 'Recall':metric['recall'], 'name':[names[i] for _ in range(len(best_3))], 'pred_name':[pred_covs[i] for _ in range(len(best_3))], 'actual_name':[actual_covs[i] for _ in range(len(best_3))]})
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
    g = sns.lineplot(data=df, x='Fraction of Space Tested', y='Actual Coverage', hue='actual_name', hue_order=actual_covs, ax=ax1)
    g.hlines(y=[chemical_space.max_possible_coverage(yield_cutoff)], xmin=0, xmax=df['Fraction of Space Tested'].max(), colors='black', linestyles='solid')
    g.hlines(y=[chemical_space.best_condition_sets(yield_cutoff, 1, 1)[0]['coverage']], xmin=0, xmax=df['Fraction of Space Tested'].max(), colors='black', linestyles='dashed')
    
    g2 = sns.lineplot(data=df, x='Fraction of Space Tested', y='Rank', hue='name',palette=colors, ax=ax2)
    g2.set_ylim(0, 110)
    ax1.set_title('Coverage of Best Predicted Sets', fontsize=15)
    ax1.set_xlabel('Fraction of Space Tested', fontsize=15)
    ax1.set_ylabel('Coverage', fontsize=15)
    ax1_handles, ax1_labels = ax1.get_legend_handles_labels()
    ax1.legend(handles=ax1_handles, labels=ax1_labels)
    colors = [handle._color for handle in g2.legend_.legend_handles]

    ax2.set_title('Best Predicted Sets', fontsize=15)
    ax2.set_xlabel('Fraction of Space Tested', fontsize=15)
    ax2.set_ylabel('Percentile', fontsize=15)
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles=handles, labels=labels)
    ax2.legend().set_visible(show_legend)
    ax1.legend().set_visible(show_legend)
    if output_file_path is not None:
        os.makedirs(output_file_path, exist_ok=True)
        plt.savefig(f'{output_file_path}/set_analysis.png') 
    
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
    ax3.legend().set_visible(show_legend)
    ax4.legend().set_visible(show_legend)
    ax5.legend().set_visible(show_legend)
    for ax in [ax3, ax4, ax5]:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels)
    if output_file_path is not None:
        plt.savefig(f'{output_file_path}/metrics.png')
    return colors

def sort_by_learner_type(files):
    sorted_files = [""]*len(files)
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

def plot_sampled_points(chemical_space:ChemicalSpace, metric_filepath:list, opt_num:int=-1, figsize=(5,5))->None:
    def reactant_to_idx(reactant):
        idx = 0
        for i, n in enumerate(reactant):
            idx += n
            if i < len(chemical_space.shape[chemical_space.conditions_dim:]) - 1:
                idx *= chemical_space.shape[chemical_space.conditions_dim + i+1]
        return idx
    def condition_to_idx(condition):
        idx = 0
        for i, n in enumerate(condition):
            idx += n
            if i < len(chemical_space.shape[:chemical_space.conditions_dim]) - 1:
                idx *= chemical_space.shape[i+1]
        return idx
    with open(metric_filepath, 'rb') as f:
        m = pickle.load(f)
    point_mat = np.full((np.prod(chemical_space.shape[chemical_space.conditions_dim:]), np.prod(chemical_space.shape[:chemical_space.conditions_dim])), np.nan)
    for i, points in enumerate(m[opt_num]['points_suggested']):
        for p in points:
            point_mat[reactant_to_idx(p[chemical_space.conditions_dim:])][condition_to_idx(p[:chemical_space.conditions_dim])] = i + 1
    fig = plt.figure(figsize=figsize)
    plt.imshow(point_mat)
    # plt.colorbar().set_label('Batch Number')
    plt.show()
    