from controller import Controller
from chemical_space import ChemicalSpace
import sys
from pathlib import Path
import os

dataset = sys.argv[1]
num_cpus = sys.argv[2]
batch_size = int(sys.argv[3])
opt_init = int(sys.argv[4])
n_repeats = int(sys.argv[5])
learner_type = int(sys.argv[6])
model_type = sys.argv[7]
stochastic_batch_num = None if sys.argv[8] == -1 else int(sys.argv[8])
cond_subset = int(sys.argv[9])


optimizer:Controller

original_path = os.getcwd()

# load in data
if dataset == 'aryl_scope':
    aryl_scope = ChemicalSpace(['ligand_name'], ['electrophile_id', 'nucleophile_id'], f'{original_path}/datasets/real_datasets/aryl-scope-ligand.csv')
    aryl_scope.titles = ['Ligand', 'Electrophile', 'Nucleophile']
    optimizer = Controller(aryl_scope,21.66, batch_size=batch_size, max_experiments=1000, early_stopping=False, learner_type=learner_type, output_dir=original_path, num_cpus=num_cpus, stochastic_cond_num=stochastic_batch_num, model_type=model_type)
elif dataset == 'borylation':
    borylation = ChemicalSpace(['ligand_name', 'solvent'], ['electrophile_id'], f'{original_path}/datasets/real_datasets/borylation.csv')
    borylation.titles = ['Ligand', 'Solvent', 'Electrophile']
    optimizer = Controller(borylation, 77.24, batch_size=batch_size, max_experiments=1000, early_stopping=False, learner_type=learner_type, output_dir=original_path, num_cpus=num_cpus, stochastic_cond_num=stochastic_batch_num, model_type=model_type, max_set_size=4)
elif dataset == 'deoxy':
    deoxy = ChemicalSpace(['base_name', 'fluoride_name'], ['substrate_name'], f'{original_path}/datasets/real_datasets/deoxyf.csv')
    deoxy.titles = ['Base', 'Fluoride', 'Substrate']
    optimizer = Controller(deoxy, 50, batch_size=batch_size, max_experiments=500, early_stopping=False, learner_type=learner_type, output_dir=original_path, num_cpus=num_cpus, stochastic_cond_num=stochastic_batch_num, model_type=model_type)
elif 'buchwald_hartwig' in dataset:
    dataset_path = f'{original_path}/datasets/real_datasets/buchwald-hartwig.csv'
    # downselect conditions by setting a default value for certain condition parameters
    default_cond_params = {}
    dataset_name_addition = ""
    if cond_subset >= 0:
        cond_defaults = [{'Base':['b'], 'Solvent':[2]}, {'Solvent':[2]}, {'Base':['b']}]
        default_cond_params = cond_defaults[cond_subset]
        included = set(['Catalyst','Solvent','Base']) - set(default_cond_params.keys())
        dataset_name_addition = '_' + '_'.join([p for p in included])
    buchwald_hartwig = ChemicalSpace(['Catalyst','Solvent','Base'], ['Amine','Bromide'], dataset_path, condition_parameter_subspace=default_cond_params)
    buchwald_hartwig.dataset_name = buchwald_hartwig.dataset_name + dataset_name_addition
    optimizer = Controller(buchwald_hartwig, 26.89, batch_size=batch_size, max_experiments=4500, early_stopping=False, learner_type=learner_type, output_dir=original_path, num_cpus=num_cpus, stochastic_cond_num=stochastic_batch_num, model_type=model_type)
else:
    raise ValueError('Invalid dataset')

optimizer.optimization_runs=opt_init
    
optimizer.do_repeats(n_repeats)

