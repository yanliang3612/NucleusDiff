import argparse
import os
import shutil

import torch
from torch_geometric.transforms import Compose

import utils.misc as misc
import utils.transforms as trans
from datasets.pl_data import ProteinLigandData, torchify_dict
from models.molopt_score_model import ScorePosNet3D
from sample_for_crossdock import sample_diffusion_ligand
from utils.data import PDBProtein
from utils import reconstruct
from rdkit import Chem
from types import SimpleNamespace


def pdb_to_pocket_data(pdb_path):
    pocket_dict = PDBProtein(pdb_path).to_dict_atom()
    data = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict=torchify_dict(pocket_dict),
        ligand_dict={
            'element': torch.empty([0, ], dtype=torch.long),
            'pos': torch.empty([0, 3], dtype=torch.float),
            'atom_feature': torch.empty([0, 8], dtype=torch.float),
            'bond_index': torch.empty([2, 0], dtype=torch.long),
            'bond_type': torch.empty([0, ], dtype=torch.long),
        }
    )

    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default="./checkpoints/nucleusdiff_pretrained_model.pt")
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--pdb_path', type=str, default="./real_world_test_extract_pockets/CDK2/cdk2_ligand_pocket10.pdb")
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--result_path', type=str, default='./read_world_cdk2_test')
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--num_steps', type=int, default=1000)
    parser.add_argument('--pos_only', type=bool, default=False)
    parser.add_argument('--center_pos_mode', type=str, default="protein")
    parser.add_argument('--sample_num_atoms', type=str, default="real_world_testing")
    parser.add_argument('--inference_num_atoms', type=int, default=30)
    parser.add_argument('--test_time', type=int, default=4)
    args = parser.parse_args()

    logger = misc.get_logger('evaluate')

    # Load config
    misc.seed_all(args.seed)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    logger.info(f"Training Config: {ckpt['config']}")

    # Transforms
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_atom_mode = SimpleNamespace(**ckpt['config']).ligand_atom_mode
    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
    transform = Compose([
        protein_featurizer,
    ])

    # Load model
    model = ScorePosNet3D(
        SimpleNamespace(**ckpt['config']),
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim
    ).to(args.device)

    model.load_state_dict(ckpt['model_mol'], strict=True)
    logger.info(f'Successfully load the model! {args.checkpoint}')

    # Load pocket
    data = pdb_to_pocket_data(args.pdb_path)
    data = transform(data)

    all_pred_pos, all_pred_v, pred_pos_traj, pred_v_traj, pred_v0_traj, pred_vt_traj, time_list = sample_diffusion_ligand(
        model, data, args.num_samples,
        batch_size=args.batch_size, device=args.device,
        num_steps=args.num_steps,
        pos_only=args.pos_only,
        center_pos_mode=args.center_pos_mode,
        sample_num_atoms=args.sample_num_atoms,
        inference_num_atoms = args.inference_num_atoms
    )
    result = {
        'data': data,
        'pred_ligand_pos': all_pred_pos,
        'pred_ligand_v': all_pred_v,
        'pred_ligand_pos_traj': pred_pos_traj,
        'pred_ligand_v_traj': pred_v_traj
    }

    logger.info('Sample done!')

    # reconstruction
    gen_mols = []
    n_recon_success, n_complete = 0, 0
    for sample_idx, (pred_pos, pred_v) in enumerate(zip(all_pred_pos, all_pred_v)):
        pred_atom_type = trans.get_atomic_number_from_index(pred_v, mode='add_aromatic')
        try:
            pred_aromatic = trans.is_aromatic_from_index(pred_v, mode='add_aromatic')
            mol = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic)
            smiles = Chem.MolToSmiles(mol)
        except reconstruct.MolReconsError:
            gen_mols.append(None)
            continue
        n_recon_success += 1

        if '.' in smiles:
            gen_mols.append(None)
            continue
        n_complete += 1
        gen_mols.append(mol)
    result['mols'] = gen_mols
    logger.info('Reconstruction done!')
    logger.info(f'n recon: {n_recon_success} n complete: {n_complete}')

    result_path = args.result_path
    os.makedirs(result_path, exist_ok=True)
    # shutil.copyfile(args.config, os.path.join(result_path, 'sample.yml'))
    torch.save(result, os.path.join(result_path, f'sample_{args.test_time}.pt'))
    mols_save_path = os.path.join(result_path, f'sdf')
    os.makedirs(mols_save_path, exist_ok=True)
    for idx, mol in enumerate(gen_mols):
        if mol is not None:
            sdf_writer = Chem.SDWriter(os.path.join(mols_save_path, f'{idx:03d}.sdf'))
            sdf_writer.write(mol)
            sdf_writer.close()
    logger.info(f'Results are saved in {result_path}')
