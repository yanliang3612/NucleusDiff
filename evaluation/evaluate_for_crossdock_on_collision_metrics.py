import argparse
import os


import numpy as np
from rdkit import RDLogger
import torch
from tqdm.auto import tqdm
from glob import glob
from collections import Counter, defaultdict


from utils.evaluation import analyze, eval_bond_length
from utils import misc
from collision_test.utils import trandforms as transforms

thresholds = {
    (1, 1): 0.64,
    (1, 6): 0.92,
    (1, 7): 0.86,
    (1, 8): 0.85,
    (1, 9): 0.85,
    (1, 15): 1.26,
    (1, 16): 1.26,
    (1, 17): 1.25,
    (6, 6): 1.2,
    (6, 7): 1.14,
    (6, 8): 1.13,
    (6, 9): 1.13,
    (6, 15): 1.54,
    (6, 16): 1.54,
    (6, 17): 1.53,
    (7, 7): 1.08,
    (7, 8): 1.07,
    (7, 9): 1.07,
    (7, 15): 1.48,
    (7, 16): 1.48,
    (7, 17): 1.47,
    (8, 8): 1.06,
    (8, 9): 1.06,
    (8, 15): 1.47,
    (8, 16): 1.47,
    (8, 17): 1.46,
    (9, 9): 1.06,
    (9, 15): 1.47,
    (9, 16): 1.47,
    (9, 17): 1.46,
    (15, 15): 1.88,
    (15, 16): 1.88,
    (15, 17): 1.87,
    (16, 16): 1.88,
    (16, 17): 1.87,
    (17, 17): 1.86
}


def distance(point1, point2):
    return np.linalg.norm(point1 - point2)


def find_nearest_atoms(ligand_pos, pocket_pos, ligand_atom_types, pocket_atom_types):
    nearest_atoms = []
    for i, ligand_atom in enumerate(ligand_pos):
        distances = []
        for j, pocket_atom in enumerate(pocket_pos):
            dist = distance(ligand_atom, pocket_atom)
            distances.append((dist, ligand_atom_types[i], pocket_atom_types[j]))

        distances.sort(key=lambda x: x[0])
        nearest_10 = distances[:10]

        for rank, (dist, ligand_type, pocket_type) in enumerate(nearest_10):
            nearest_atoms.append({'ranking': rank, 'atom_pair': (ligand_type, pocket_type), 'distance': dist})
    return nearest_atoms


def check_below_threshold(atom_pair, distance):
    atom_pair = tuple(sorted(map(int, atom_pair)))
    return float(distance) < thresholds.get(atom_pair, float('inf'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_path', type=str, default= "./result_out")
    parser.add_argument('--eval_step', type=int, default=0)
    parser.add_argument('--eval_num_examples', type=int, default=None)
    parser.add_argument('--atom_enc_mode', type=str, default='add_aromatic')
    parser.add_argument('--exhaustiveness', type=int, default=16)
    parser.add_argument('--model', type=str, default="nucleusdiff_test")
    parser.add_argument('--verbose', type=bool, default=False)
    args = parser.parse_args()
    result_path = os.path.join(args.sample_path, f'eval_step_{args.eval_step}_collision_results')
    os.makedirs(result_path, exist_ok=True)
    logger = misc.get_logger('evaluate', log_dir=result_path)
    if not args.verbose:
        RDLogger.DisableLog('rdApp.*')
    # Load generated data
    results_fn_list = glob(os.path.join(args.sample_path, '*result_*.pt'))
    results_fn_list = sorted(results_fn_list, key=lambda x: int(os.path.basename(x)[:-3].split('_')[-1]))
    if args.eval_num_examples is not None:
        results_fn_list = results_fn_list[:args.eval_num_examples]
    num_examples = len(results_fn_list)


    logger.info(f'The model we evaluated is {args.model}')
    logger.info(f'The eval_step is {args.eval_step}')
    logger.info(f'Load generated data done! {num_examples} examples in total.')

    num_samples = 0
    all_mol_stable, all_atom_stable, all_n_atom = 0, 0, 0
    results = []
    all_pair_dist, all_bond_dist = [], []
    all_atom_types = Counter()
    success_pair_dist, success_atom_types = [], Counter()

    atom_level_pocket_ligand__nearest_atoms_info = []
    grouped_info = defaultdict(list)

    for example_idx, r_name in enumerate(tqdm(results_fn_list, desc='Eval')):
        r = torch.load(r_name)
        all_pred_ligand_pos = r['pred_ligand_pos_traj']  # [num_samples, num_steps, num_atoms, 3]
        all_pred_ligand_v = r['pred_ligand_v_traj']
        num_samples += len(all_pred_ligand_pos)

        for sample_idx, (pred_pos, pred_v) in enumerate(zip(all_pred_ligand_pos, all_pred_ligand_v)):
            pred_pos, pred_v = pred_pos[args.eval_step], pred_v[args.eval_step]

            # stability check
            pred_atom_type = transforms.get_atomic_number_from_index(pred_v, mode=args.atom_enc_mode)
            all_atom_types += Counter(pred_atom_type)
            r_stable = analyze.check_stability(pred_pos, pred_atom_type)
            all_mol_stable += r_stable[0]
            all_atom_stable += r_stable[1]
            all_n_atom += r_stable[2]

            pair_dist = eval_bond_length.pair_distance_from_pos_v(pred_pos, pred_atom_type)

            protein_pos = r['data'].protein_pos
            protein_element = r['data'].protein_element

            nearest_atoms_info = find_nearest_atoms(pred_pos, protein_pos.numpy(), pred_atom_type,
                                                    protein_element.numpy())

            for item in nearest_atoms_info:
                atom_level_pocket_ligand__nearest_atoms_info.append(
                    {"ranking": item["ranking"], "atom_pair": item["atom_pair"], "distance": item["distance"]}
                )
                grouped_info[100 * example_idx + sample_idx].append(
                    {"ranking": item["ranking"], "atom_pair": item["atom_pair"], "distance": item["distance"]}
                )

    grouped_list = [atom_level_pocket_ligand__nearest_atoms_info[i:i + 10] for i in
                    range(0, len(atom_level_pocket_ligand__nearest_atoms_info), 10)]
    final_mol_grouped_list = list(grouped_info.values())

    atom_pair_distance_below_threshold = 0
    atom_distance_below_threshold = 0
    mol_distance_below_threshold = 0

    for i in range(len(atom_level_pocket_ligand__nearest_atoms_info)):
           atom_pair = atom_level_pocket_ligand__nearest_atoms_info[i]["atom_pair"]
           distance =  atom_level_pocket_ligand__nearest_atoms_info[i]["distance"]
           if check_below_threshold(atom_pair, distance):
              atom_pair_distance_below_threshold += 1
           else:
              atom_pair_distance_below_threshold += 0

    logger.info(f'atom_pair_number_in_all:{len(atom_level_pocket_ligand__nearest_atoms_info)}')
    logger.info(f'Metric:PLCR:{atom_pair_distance_below_threshold}')

    for i in range(len(grouped_list)):
        nearest_atoms_info_for_one = grouped_list[i]
        for j in range(len(nearest_atoms_info_for_one)):
            atom_pair = nearest_atoms_info_for_one[j]["atom_pair"]
            distance = nearest_atoms_info_for_one[j]["distance"]
            if check_below_threshold(atom_pair, distance):
                atom_distance_below_threshold += 1
                break

    logger.info(f'atom_number_in_all:{len(grouped_list)}')
    logger.info(f'Metric:ALCR:{atom_distance_below_threshold}')

    for i in range(len(final_mol_grouped_list)):
        nearest_mol_info_for_one = final_mol_grouped_list[i]
        for j in range(len(nearest_mol_info_for_one)):
            atom_pair = nearest_mol_info_for_one[j]["atom_pair"]
            distance = nearest_mol_info_for_one[j]["distance"]
            if check_below_threshold(atom_pair, distance):
                mol_distance_below_threshold += 1
                break

    logger.info(f'mol_number_in_all:{len(final_mol_grouped_list)}')
    logger.info(f'Metric:MLCR:{mol_distance_below_threshold}')

