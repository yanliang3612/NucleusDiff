import os
import argparse
import shutil

from utils.data import PDBProtein, parse_sdf_file


def load_item(item, path):
    pdb_path = os.path.join(path, item[0])
    sdf_path = os.path.join(path, item[1])
    with open(pdb_path, 'r') as f:
        pdb_block = f.read()
    with open(sdf_path, 'r') as f:
        sdf_block = f.read()
    return pdb_block, sdf_block


def process_item(item, args):
    try:
        pdb_block, sdf_block = load_item(item, args.source)
        protein = PDBProtein(pdb_block)
        # ligand = parse_sdf_block(sdf_block)
        ligand = parse_sdf_file(os.path.join(args.source, item[1]))

        pdb_block_pocket = protein.residues_to_pdb_block(
            protein.query_residues_ligand(ligand, args.radius)
        )
        
        ligand_fn = item[1]
        pocket_fn = ligand_fn[:-4] + '_pocket%d.pdb' % args.radius
        ligand_dest = os.path.join(args.dest, ligand_fn)
        pocket_dest = os.path.join(args.dest, pocket_fn)
        os.makedirs(os.path.dirname(ligand_dest), exist_ok=True)

        shutil.copyfile(
            src=os.path.join(args.source, ligand_fn),
            dst=os.path.join(args.dest, ligand_fn)
        )
        with open(pocket_dest, 'w') as f:
            f.write(pdb_block_pocket)
        return pocket_fn, ligand_fn, item[0], item[1] # item[0]: original protein filename; item[2]: rmsd.
    except Exception:
        print('Exception occurred.', item)
        return None, item[1], item[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default="./data/real_world")
    parser.add_argument('--dest', type=str, default="./real_world_test_extract_pockets")
    parser.add_argument('--radius', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=True)

    item_list = [["3CL/3cl_processed.pdb", "3CL/3cl_ligand.sdf"], ["AKT1/akt1_protein_processed.pdb", "AKT1/akt1_ligand_refine.sdf"], ["CDK2/cdk2_protein.pdb", "CDK2/cdk2_ligand.sdf"]]

    for item in item_list:
       process_item(item, args)
