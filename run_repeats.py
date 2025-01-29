from controller import Controller
from chemical_space import ChemicalSpace
import sys
import os

# script to create and run optimization job which is configured from command line arguments

dataset = sys.argv[1] # options: aryl_scope, borylation, deoxy, buchwald_hartwig, buchwald_hartwig_both_1, buchwald_hartwig_rxn_1, buchwald_hartwig_cond_1
num_cpus = sys.argv[2] # number of cpus available to the job
batch_size = int(sys.argv[3]) # batch size for optimization
opt_init = int(sys.argv[4]) # number of initial experiments if using preset seed points
n_repeats = int(sys.argv[5]) # number of repeats to run
learner_type = int(sys.argv[6]) # Acquisition function used, 
                # options: Random Selection:-1, Explore: 0, Combined:8, Exploit:9, see controller for more exploit functions
model_type = sys.argv[7] # options: GP (Gaussian Process Classifier), RF (Random Forest Classifier)
stochastic_batch_num = None if sys.argv[8] == -1 else int(sys.argv[8]) # number of conditions to randomly sample when using stochastic batch
cond_subset = int(sys.argv[9]) # subset of conditions to use (for buchwald_hartwig)
to_fingerprint = int(sys.argv[10]) # titles to fingerprint for deoxy and aryl_scope


optimizer:Controller

original_path = os.getcwd()

def create_bh_chem_space(dataset_name, cond_subset):
    dataset_path = 'datasets/real_datasets/buchwald-hartwig.csv'
    if dataset_name == 'buchwald_hartwig_both_1':
        dataset_path = f'datasets/real_datasets/buchwald-hartwig_both_subset_1.csv'
    elif dataset_name == 'buchwald_hartwig_rxn_1':
        dataset_path = f'datasets/real_datasets/buchwald-hartwig_rxn_subset_1.csv'
    elif dataset_name == 'buchwald_hartwig_cond_1':
        dataset_path = f'datasets/real_datasets/buchwald-hartwig_cond_subset_1.csv'
    # downselect conditions by setting a default value for certain condition parameters
    default_cond_params = {}
    dataset_name_addition = ""
    if cond_subset >= 0:
        cond_defaults = [{'Base':['b'], 'Solvent':[2]}, {'Solvent':[2]}, {'Base':['b']}]
        default_cond_params = cond_defaults[cond_subset]
        included = set(['Catalyst','Solvent','Base']) - set(default_cond_params.keys())
        included = list(included)
        included.sort()
        dataset_name_addition = '_' + '_'.join([p for p in included])
    buchwald_hartwig = ChemicalSpace(['Catalyst','Solvent','Base'], ['Amine','Bromide'], dataset_path, condition_parameter_subspace=default_cond_params)
    buchwald_hartwig.dataset_name = buchwald_hartwig.dataset_name + dataset_name_addition
    return buchwald_hartwig

# load in data
if dataset == 'aryl_scope':
    titles_to_fingerprint = []
    dataset_name_addition = ""
    if to_fingerprint >= 0:
        fingerprint_title_options = [['ligand_smiles'], ['electrophile_smiles'], ['nucleophile_smiles'], ['electrophile_smiles', 'nucleophile_smiles'], ['ligand_smiles', 'electrophile_smiles', 'nucleophile_smiles']]
        titles_to_fingerprint = fingerprint_title_options[to_fingerprint]
        dataset_name_addition = '_' + '_'.join([p for p in titles_to_fingerprint])
    aryl_scope = ChemicalSpace(['ligand_smiles'], ['electrophile_smiles', 'nucleophile_smiles'], f'{original_path}/datasets/real_datasets/aryl-scope-ligand.csv', titles_to_fingerprint=titles_to_fingerprint)
    aryl_scope.titles = ['Ligand', 'Electrophile', 'Nucleophile']
    aryl_scope.dataset_name = aryl_scope.dataset_name + dataset_name_addition
    optimizer = Controller(aryl_scope,21.66, batch_size=batch_size, max_experiments=1000, early_stopping=False, learner_type=learner_type, output_dir=original_path, num_cpus=num_cpus, stochastic_cond_num=stochastic_batch_num, model_type=model_type)
elif dataset == 'borylation':
    borylation = ChemicalSpace(['ligand_name', 'solvent'], ['electrophile_id'], f'{original_path}/datasets/real_datasets/borylation.csv')
    borylation.titles = ['Ligand', 'Solvent', 'Electrophile']
    optimizer = Controller(borylation, 77.24, batch_size=batch_size, max_experiments=1000, early_stopping=False, learner_type=learner_type, output_dir=original_path, num_cpus=num_cpus, stochastic_cond_num=stochastic_batch_num, model_type=model_type, max_set_size=4)
elif dataset == 'deoxy':
    titles_to_fingerprint = []
    dataset_name_addition = ""
    if to_fingerprint >= 0:
        fingerprint_title_options = [['base_smiles'], ['fluoride_smiles'], ['substrate_smiles'], ['base_smiles','fluoride_smiles'], ['base_smiles','fluoride_smiles','substrate_smiles']]
        titles_to_fingerprint = fingerprint_title_options[to_fingerprint]
        dataset_name_addition = '_' + '_'.join([p for p in titles_to_fingerprint])
    deoxy = ChemicalSpace(['base_smiles', 'fluoride_smiles'], ['substrate_smiles'], f'{original_path}/datasets/real_datasets/deoxyf.csv', titles_to_fingerprint=titles_to_fingerprint)
    deoxy.titles = ['Base', 'Fluoride', 'Substrate']
    deoxy.dataset_name = deoxy.dataset_name + dataset_name_addition
    optimizer = Controller(deoxy, 50, batch_size=batch_size, max_experiments=500, early_stopping=False, learner_type=learner_type, output_dir=original_path, num_cpus=num_cpus, stochastic_cond_num=stochastic_batch_num, model_type=model_type)
elif 'buchwald_hartwig' in dataset:
    buchwald_hartwig = create_bh_chem_space(dataset, cond_subset)
    optimizer = Controller(buchwald_hartwig, 26.89, batch_size=batch_size, max_experiments=4500, early_stopping=False, learner_type=learner_type, output_dir=original_path, num_cpus=num_cpus, stochastic_cond_num=stochastic_batch_num, model_type=model_type)
else:
    raise ValueError('Invalid dataset')

optimizer.optimization_runs=opt_init
    
optimizer.do_repeats(n_repeats)

