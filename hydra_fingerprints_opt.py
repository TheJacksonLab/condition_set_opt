from chemical_space import ChemicalSpace
from controller import Controller
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from pathlib import Path
import os

@hydra.main(version_base=None, config_path="conf", config_name="config")
def run(cfg: DictConfig) -> None:
    optimizer:Controller

    original_path = Path(get_original_cwd())

    os.chdir(original_path)

    if cfg.get('stochastic_batch_num', 'None') == 'None':
        cfg.stochastic_batch_num = None
    
    if cfg.get('model_type', 'None') == 'None':
        cfg.model_type = 'GP'
    
    if cfg.get('opt_init', 0) == 0:
        cfg.opt_init = 0
    
    print(cfg.opt_init)

    # load in data
    if cfg.dataset == 'aryl_scope':
        fp_titles = []
        if cfg.fingerprint_titles == 1:
            fp_titles = ['ligand_smiles', 'electrophile_smiles', 'nucleophile_smiles']
        else:
            return
        aryl_scope = ChemicalSpace(['ligand_name'], ['electrophile_id', 'nucleophile_id'], f'{original_path}/datasets/real_datasets/aryl-scope-ligand.csv', titles_to_fingerprint=fp_titles)
        aryl_scope.titles = ['Ligand', 'Electrophile', 'Nucleophile']
        aryl_scope.dataset_name += '-fingerprint_' + ','.join(fp_titles)
        optimizer = Controller(aryl_scope,21.66, batch_size=cfg.batch_size, max_experiments=1000, early_stopping=False, learner_type=cfg.learner_type, output_dir=original_path, num_cpus=cfg.num_cpus, stochastic_cond_num=cfg.get('stochastic_batch_num', None), model_type=cfg.model_type)
    elif cfg.dataset == 'borylation':
        borylation = ChemicalSpace(['ligand_name', 'solvent'], ['electrophile_id'], f'{original_path}/datasets/real_datasets/borylation.csv')
        borylation.titles = ['Ligand', 'Solvent', 'Electrophile']
        borylation.dataset_name += '-fingerprint'
        optimizer = Controller(borylation, 77.24, batch_size=cfg.batch_size, max_experiments=1000, early_stopping=False, learner_type=cfg.learner_type, output_dir=original_path, num_cpus=cfg.num_cpus, stochastic_cond_num=cfg.get('stochastic_batch_num', None), model_type=cfg.model_type, max_set_size=4)
    elif cfg.dataset == 'deoxy':
        deoxy = ChemicalSpace(['base_name', 'fluoride_name'], ['substrate_name'], f'{original_path}/datasets/real_datasets/deoxyf.csv')
        deoxy.titles = ['Base', 'Fluoride', 'Substrate']
        deoxy.dataset_name += '-fingerprint'
        optimizer = Controller(deoxy, 50, batch_size=cfg.batch_size, max_experiments=500, early_stopping=False, learner_type=cfg.learner_type, output_dir=original_path, num_cpus=cfg.num_cpus, stochastic_cond_num=cfg.get('stochastic_batch_num', None), model_type=cfg.model_type)
    elif cfg.dataset == 'buchwald_hartwig':
        buchwald_hartwig = ChemicalSpace(['Catalyst','Solvent','Base'], ['Amine','Bromide'], f'{original_path}/datasets/real_datasets/buchwald-hartwig.csv')
        buchwald_hartwig.dataset_name += '-fingerprint'
        optimizer = Controller(buchwald_hartwig, 26.89, batch_size=cfg.batch_size, max_experiments=4500, early_stopping=False, learner_type=cfg.learner_type, output_dir=original_path, num_cpus=cfg.num_cpus, stochastic_cond_num=cfg.get('stochastic_batch_num', None), model_type=cfg.model_type)
    else:
        raise ValueError('Invalid dataset')

    optimizer.optimization_runs=cfg.opt_init
    
    optimizer.do_repeats(cfg.n_repeats)

if __name__ == "__main__":
    run()

