import argparse
import wandb
import os

import numpy as np
import torch
import torch.utils.tensorboard
from sklearn.metrics import roc_auc_score
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from tqdm.auto import tqdm
from rdkit import RDLogger
from rdkit import Chem
from pytorch_lightning import seed_everything

import utils.train as utils_train
import utils.transforms as trans
from collections import Counter
from datasets import get_dataset, get_mesh_dataset
from datasets.pl_data import FOLLOW_BATCH
from models.molopt_score_model import calculate_vdw_loss
from models.molopt_score_model import ScorePosNet3D_mol
from models.molopt_score_model import ScorePosNet3D_mesh
from utils import misc, reconstruct, transforms
from utils.evaluation import eval_atom_type, scoring_func, analyze, eval_bond_length
from collision_test.utils.data import vdw_radii_dict




def get_auroc(y_true, y_pred, feat_mode):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    avg_auroc = 0.
    possible_classes = set(y_true)
    for c in possible_classes:
        auroc = roc_auc_score(y_true == c, y_pred[:, c])
        avg_auroc += auroc * np.sum(y_true == c)
        mapping = {
            'basic': trans.MAP_INDEX_TO_ATOM_TYPE_ONLY,
            'add_aromatic': trans.MAP_INDEX_TO_ATOM_TYPE_AROMATIC,
            'full': trans.MAP_INDEX_TO_ATOM_TYPE_FULL
        }
        print(f'atom: {mapping[feat_mode][c]} \t auc roc: {auroc:.4f}')
    return avg_auroc / len(y_true)



def parser_args_sweep():
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument('--data_name', type=str, default="pl")
    parser.add_argument('--data_path', type=str,
                        default='./data/crossdocked_v1.1_rmsd1.0_pocket10')
    parser.add_argument('--data_split', type=str,
                        default='./data/crossdocked_pocket10_pose_w_manifold_data_split.pt')
    parser.add_argument('--ligand_atom_mode', type=str, default='add_aromatic')
    parser.add_argument('--random_rot', default=False)

    # loss weight
    parser.add_argument('--loss_v_weight', type=float, default=100.)
    parser.add_argument('--loss_mesh_constained_weight', type=float, default=1.)
    parser.add_argument('--loss_pos_mesh_weight', type=float, default=1.)
    parser.add_argument('--loss_v_mesh_weight', type=float, default=100.)

    # model
    parser.add_argument('--model_mean_type', type=str, default="C0")
    parser.add_argument('--beta_schedule', type=str, default="sigmoid")
    parser.add_argument('--beta_start', type=float, default=1.e-7)
    parser.add_argument('--beta_end', type=float, default=2.e-3)
    parser.add_argument('--v_beta_schedule', type=str, default='cosine')
    parser.add_argument('--v_beta_s', type=float, default=0.01)
    parser.add_argument('--num_diffusion_timesteps', type=int, default=1000)
    parser.add_argument('--sample_time_method', type=str, default='symmetric')
    parser.add_argument('--time_emb_dim', type=int, default=0)
    parser.add_argument('--time_emb_mode', type=str, default='simple')
    parser.add_argument('--center_pos_mode', type=str, default='protein')

    # model setting
    parser.add_argument('--node_indicator', default=True)
    parser.add_argument('--model_type', type=str, default='uni_o2')
    parser.add_argument('--num_blocks', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=9)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=16)
    parser.add_argument('--edge_feat_dim', type=int, default=4)
    parser.add_argument('--num_r_gaussian', type=int, default=20)
    parser.add_argument('--knn', type=int, default=32)
    parser.add_argument('--num_node_types', type=int, default=8)

    parser.add_argument('--act_fn', type=str, default='relu')
    parser.add_argument('--norm', default=True)
    parser.add_argument('--cutoff_mode', type=str, default='knn')
    parser.add_argument('--ew_net_type', type=str, default='global')
    parser.add_argument('--num_x2h', type=int, default=1)
    parser.add_argument('--num_h2x', type=int, default=1)
    parser.add_argument('--r_max', type=float, default=10.)
    parser.add_argument('--x2h_out_fc', default=False)
    parser.add_argument('--sync_twoup', default=False)

    # train
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--n_acc_batch', type=int, default=1)
    parser.add_argument('--max_iters', type=int, default=500000)
    parser.add_argument('--val_freq', type=int, default=100)
    parser.add_argument('--pos_noise_std', type=float, default=0.1)
    parser.add_argument('--max_grad_norm', type=float, default=8.0)
    parser.add_argument('--bond_loss_weight', type=float, default=1.0)

    # optimizer
    parser.add_argument('--optimizer_type', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--beta1', type=float, default=0.95)
    parser.add_argument('--beta2', type=float, default=0.999)

    # scheduler
    parser.add_argument('--scheduler_type', type=str, default='plateau')
    parser.add_argument('--factor', type=int, default=0.6)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--min_lr', type=int, default=1.e-6)

    # sample
    parser.add_argument('--sample_num_diffusion_timesteps', type=int, default=1000)
    parser.add_argument('--sample_num_samples', type=int, default=100)
    parser.add_argument('--sample_num_atoms', type=str, default="prior")
    parser.add_argument('--pos_only', default=False)
    parser.add_argument('--sample_batch_size', type=int, default=100)

    # evaluate
    parser.add_argument('--evaluate_verbose', default=False)
    parser.add_argument('--eval_step', type=int, default=-1)
    parser.add_argument('--eval_num_examples', type=int, default=None)

    # more
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--logdir', type=str, default='./logs_diffusion')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--train_report_iter', type=int, default=200)
    parser.add_argument('--exp_name', type=str, default="test")
    parser.add_argument('--use_wandb', default=False)
    parser.add_argument("--sweep_id", type=str, default='yanliangfdu/targetdiff_mesh_2/d72sxrin')
    parser.add_argument('--wandb_project_name', type=str, default="sbdd_test_1")
    return parser.parse_args()



def evaluate(results_fn_list, args, logger):
    if not args.evaluate_verbose:
        RDLogger.DisableLog('rdApp.*')

    num_examples = len(results_fn_list)
    logger.info(f'Load generated data done! {num_examples} examples in total.')

    num_samples = 0
    all_mol_stable, all_atom_stable, all_n_atom = 0, 0, 0
    n_recon_success, n_complete = 0, 0
    results = []
    all_pair_dist, all_bond_dist = [], []

    all_atom_types = Counter()
    success_pair_dist, success_atom_types = [], Counter()
    for example_idx, r in enumerate(tqdm(results_fn_list, desc='Eval')):
        all_pred_ligand_pos = r['pred_ligand_pos_traj']
        all_pred_ligand_v = r['pred_ligand_v_traj']
        num_samples += len(all_pred_ligand_pos)

        for sample_idx, (pred_pos, pred_v) in enumerate(zip(all_pred_ligand_pos, all_pred_ligand_v)):
            pred_pos, pred_v = pred_pos[args.eval_step], pred_v[args.eval_step]

            # stability check
            pred_atom_type = transforms.get_atomic_number_from_index(pred_v, mode=args.ligand_atom_mode)
            all_atom_types += Counter(pred_atom_type)

            r_stable = analyze.check_stability(pred_pos, pred_atom_type)
            all_mol_stable += r_stable[0]
            all_atom_stable += r_stable[1]
            all_n_atom += r_stable[2]

            pair_dist = eval_bond_length.pair_distance_from_pos_v(pred_pos, pred_atom_type)
            all_pair_dist += pair_dist

            try:
                pred_aromatic = transforms.is_aromatic_from_index(pred_v, mode=args.ligand_atom_mode)
                mol = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic)
                smiles = Chem.MolToSmiles(mol)
            except reconstruct.MolReconsError:
                if args.evaluate_verbose:
                    logger.warning('Reconstruct failed %s' % f'{example_idx}_{sample_idx}')
                continue
            n_recon_success += 1

            if '.' in smiles:
                continue
            n_complete += 1

            # now we only consider complete molecules as success
            bond_dist = eval_bond_length.bond_distance_from_mol(mol)
            all_bond_dist += bond_dist

            success_pair_dist += pair_dist
            success_atom_types += Counter(pred_atom_type)

            results.append({
                'mol': mol,
                'smiles': smiles,
                'ligand_filename': r['data'].ligand_filename,
                'pred_pos': pred_pos,
                'pred_v': pred_v,
            })

    logger.info(f'Evaluate done! {num_samples} samples in total.')

    fraction_mol_stable = all_mol_stable / num_samples
    fraction_atm_stable = all_atom_stable / all_n_atom
    fraction_recon = n_recon_success / num_samples
    fraction_complete = n_complete / num_samples

    logger.info('Number of reconstructed mols: %d, complete mols: %d, evaluated mols: %d' % (
        n_recon_success, n_complete, len(results)))

    evaluate_results = {
        'mol_stable': fraction_mol_stable,
        'atm_stable': fraction_atm_stable,
        'recon_success': fraction_recon,
        'complete': fraction_complete,
        'Number of reconstructed mols': n_recon_success,
        'complete mols': n_complete,
        'evaluated mols': len(results),
    }
    return evaluate_results



def main():
    args = parser_args_sweep()
    # import pdb; pdb.set_trace()
    if args.use_wandb:
        wandb.init(project=args.wandb_project_name)
        wandb.config.update(args)

    config_name = 'training'
    seed_everything(args.seed)

    # Logging
    log_dir = misc.get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag, project=args.wandb_project_name)

    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    vis_dir = os.path.join(log_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)
    logger = misc.get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)

    # Transforms
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_featurizer = trans.FeaturizeLigandAtom(args.ligand_atom_mode)
    transform_list = [
        protein_featurizer,
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
    ]
    if args.random_rot:
        transform_list.append(trans.RandomRotation())
    transform = Compose(transform_list)

    # Datasets and loaders
    logger.info('Loading dataset...')

    dataset, subsets = get_mesh_dataset(
        name=args.data_name,
        path=args.data_path,
        split_path=args.data_split,
        transform=transform
    )

    train_set, val_set = subsets['train'], subsets['val']
    logger.info(f'Training: {len(train_set)} Validation: {len(val_set)}')

    collate_exclude_keys = ['ligand_nbh_list']
    train_iterator = utils_train.inf_iterator(DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        follow_batch=FOLLOW_BATCH,
        exclude_keys=collate_exclude_keys
    ))
    val_loader = DataLoader(val_set, args.batch_size, shuffle=False,
                            follow_batch=FOLLOW_BATCH, exclude_keys=collate_exclude_keys)

    # Model
    logger.info('Building model...')
    model_mol = ScorePosNet3D_mol(
        args,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim
    ).to(args.device)

    model_mesh = ScorePosNet3D_mesh(
        args,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim +1
    ).to(args.device)

    # print model
    print(
        f'protein feature dim: {protein_featurizer.feature_dim} ligand feature dim: {ligand_featurizer.feature_dim}')
    logger.info(f'# model_mol trainable parameters: {misc.count_parameters(model_mol) / 1e6:.4f} M')
    logger.info(f'# model_mesh trainable parameters: {misc.count_parameters(model_mesh) / 1e6:.4f} M')

    # Optimizer and scheduler
    optimizer = utils_train.get_new_optimizer(args, model_mol, model_mesh)
    scheduler = utils_train.get_scheduler(args, optimizer)

    def train(it):
        model_mol.train()
        model_mesh.train()
        optimizer.zero_grad()
        for _ in range(args.n_acc_batch):
            batch = next(train_iterator).to(args.device)

            protein_noise = torch.randn_like(batch.protein_pos) * args.pos_noise_std
            gt_protein_pos = batch.protein_pos + protein_noise

            results_mol, pred = model_mol.get_diffusion_loss(
                protein_pos=gt_protein_pos,
                protein_v=batch.protein_atom_feature.float(),
                batch_protein=batch.protein_element_batch,

                ligand_pos=batch.ligand_pos,
                ligand_v=batch.ligand_atom_feature_full,
                batch_ligand=batch.ligand_element_batch
            )

            repeat_times = int(batch.ligand_mesh_pos.shape[0] / batch.ligand_pos.shape[0])
            repeated_shape = torch.repeat_interleave(batch.ligand_atom_feature_full, repeat_times).shape
            data_type = batch.ligand_atom_feature_full.dtype
            ligand_mesh_v= torch.full(repeated_shape, 13, dtype=data_type).to(args.device)

            results_mesh, pred_mesh = model_mesh.get_diffusion_loss(
                protein_pos=gt_protein_pos,
                protein_v=batch.protein_atom_feature.float(),
                batch_protein=batch.protein_element_batch,

                ligand_pos=batch.ligand_mesh_pos,
                ligand_v= ligand_mesh_v,
                batch_ligand=torch.repeat_interleave(batch.ligand_element_batch, int(
                    batch.ligand_mesh_pos.shape[0] / batch.ligand_pos.shape[0])),
            )

            loss_mesh_constrained = calculate_vdw_loss(pred_mesh, pred, batch.ligand_element, vdw_radii_dict,
                                batch.ligand_element_batch, int((batch.ligand_mesh_pos).shape[0] / (batch.ligand_pos).shape[0]))
            loss_mesh_constrained = torch.mean(loss_mesh_constrained)

            loss_mol, loss_mol_pos, loss_mol_v = results_mol['loss'], results_mol['loss_pos'], results_mol['loss_v']
            loss_mesh, loss_mesh_pos, loss_mesh_v = results_mesh['loss'], results_mesh['loss_pos'], results_mesh['loss_v']

            loss = loss_mol + loss_mesh + args.loss_mesh_constained_weight * loss_mesh_constrained
            loss = loss / args.n_acc_batch
            loss.backward()

        orig_grad_norm = clip_grad_norm_(list(model_mol.parameters())+list(model_mesh.parameters()), args.max_grad_norm)
        optimizer.step()

        logger.info(
            '[Train] Iter %d | Loss %.6f (mol_loss %.6f | mol_pos_loss %.6f | mol_v_loss %.6f | loss_mesh_contrained %.6f | loss_mesh %.6f | loss_mesh_pos %.6f | loss_mesh_v %.6f) | Lr: %.6f | Grad Norm: %.6f' % (
                it, loss, loss_mol, loss_mol_pos, loss_mol_v, loss_mesh_constrained, loss_mesh, loss_mesh_pos, loss_mesh_v,
                optimizer.param_groups[0]['lr'], orig_grad_norm
            )
        )

        if args.use_wandb:
            wandb.log({
                "Iter": it,
                "Total Loss": loss,
                "Mol Loss": loss_mol,
                "Mol Position Loss": loss_mol_pos,
                "Mol V Loss": loss_mol_v * 1000,
                "Mesh Loss": loss_mesh,
                "Mesh Position Loss": loss_mesh_pos,
                "Mesh Atom Type Loss": loss_mesh_v* 1000,
                "Mesh Constrained Loss": loss_mesh_constrained,
                "Learning Rate": optimizer.param_groups[0]['lr'],
                "Gradient Norm": orig_grad_norm
            })

        if it % args.train_report_iter == 0:
            for k, v in results_mol.items():
                if torch.is_tensor(v) and v.squeeze().ndim == 0:
                    writer.add_scalar(f'train/{k}', v, it)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
            writer.add_scalar('train/grad', orig_grad_norm, it)
            writer.flush()

            for k, v in results_mesh.items():
                if torch.is_tensor(v) and v.squeeze().ndim == 0:
                    writer.add_scalar(f'train/{k}', v, it)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
            writer.add_scalar('train/grad', orig_grad_norm, it)
            writer.flush()


    def validate(it):

        sum_loss, sum_loss_mesh_constrained, sum_mol_loss, sum_mol_loss_pos, \
        sum_mol_loss_v, sum_mesh_loss, sum_mesh_loss_pos, sum_mesh_loss_v, sum_n = 0, 0, 0, 0, 0, 0, 0, 0, 0

        all_pred_v, all_true_v = [], []
        with torch.no_grad():
            model_mol.eval()
            model_mesh.eval()
            for batch in tqdm(val_loader, desc='Validate'):
                batch = batch.to(args.device)
                batch_size = batch.num_graphs
                for t in np.linspace(0, model_mol.num_timesteps - 1, 10).astype(int):
                    time_step = torch.tensor([t] * batch_size).to(args.device)

                    results_mol, val_pred = model_mol.get_diffusion_loss(
                        protein_pos=batch.protein_pos,
                        protein_v=batch.protein_atom_feature.float(),
                        batch_protein=batch.protein_element_batch,

                        ligand_pos=batch.ligand_pos,
                        ligand_v=batch.ligand_atom_feature_full,
                        batch_ligand=batch.ligand_element_batch
                    )

                    repeat_times = int(batch.ligand_mesh_pos.shape[0] / batch.ligand_pos.shape[0])
                    repeated_shape = torch.repeat_interleave(batch.ligand_atom_feature_full, repeat_times).shape
                    data_type = batch.ligand_atom_feature_full.dtype
                    val_ligand_mesh_v = torch.full(repeated_shape, 13, dtype=data_type).to(args.device)

                    results_mesh, val_pred_mesh = model_mesh.get_diffusion_loss(
                        protein_pos=batch.protein_pos,
                        protein_v=batch.protein_atom_feature.float(),
                        batch_protein=batch.protein_element_batch,

                        ligand_pos=batch.ligand_mesh_pos,
                        ligand_v=val_ligand_mesh_v,
                        batch_ligand=torch.repeat_interleave(batch.ligand_element_batch, int(
                            batch.ligand_mesh_pos.shape[0] / batch.ligand_pos.shape[0])),
                    )

                    val_loss_mesh_constrained =calculate_vdw_loss(val_pred_mesh,val_pred, batch.ligand_element, vdw_radii_dict,
                                           batch.ligand_element_batch, int((batch.ligand_mesh_pos).shape[0] / (batch.ligand_pos).shape[0]))
                    val_loss_mesh_constrained = torch.mean(val_loss_mesh_constrained)
                    val_mol_loss, val_mol_loss_pos, val_mol_loss_v = results_mol['loss'], results_mol['loss_pos'], results_mol['loss_v']

                    val_loss = 2 * val_mol_loss + args.loss_mesh_constained_weight * val_loss_mesh_constrained

                    sum_loss += float(val_loss) * batch_size
                    sum_mol_loss += float(val_mol_loss) * batch_size
                    sum_mol_loss_pos += float(val_mol_loss_pos) * batch_size
                    sum_mol_loss_v += float(val_mol_loss_v) * batch_size


                    sum_loss_mesh_constrained += float(val_loss_mesh_constrained) * batch_size


                    sum_n += batch_size
                    all_pred_v.append(results_mol['ligand_v_recon'].detach().cpu().numpy())
                    all_true_v.append(batch.ligand_atom_feature_full.detach().cpu().numpy())


        avg_loss = sum_loss / sum_n
        avg_mol_loss = sum_mol_loss / sum_n
        avg_mol_loss_pos = sum_mol_loss_pos / sum_n
        avg_mol_loss_v = sum_mol_loss_v / sum_n

        avg_loss_mesh_constrained= sum_loss_mesh_constrained / sum_n


        atom_auroc = get_auroc(np.concatenate(all_true_v), np.concatenate(all_pred_v, axis=0),
                               feat_mode=args.ligand_atom_mode)

        if args.scheduler_type == 'plateau':
            scheduler.step(avg_loss)
        elif args.scheduler_type == 'warmup_plateau':
            scheduler.step_ReduceLROnPlateau(avg_loss)
        else:
            scheduler.step()

        logger.info(
            '[Validate] Iter %05d | Loss %.6f | Loss mol %.6f |Loss mol pos %.6f | Loss mol v %.6f e-3 | loss_mesh_constrained %.6f e-3 | Avg atom auroc %.6f' % (
                it, avg_loss, avg_mol_loss, avg_mol_loss_pos, avg_mol_loss_v * 1000, avg_loss_mesh_constrained, atom_auroc
            )
        )

        if args.use_wandb:
            wandb.log({
                "Iter": it,
                "Val Loss": avg_loss,
                "Val Mol Loss": avg_mol_loss,
                "Val Mol Position Loss": avg_mol_loss_pos,
                "Val Mol V Loss": avg_mol_loss_v * 1000,
                "Val Mesh Constrained Loss": avg_loss_mesh_constrained,
                "Atom Auroc": atom_auroc,
            })

        writer.add_scalar('val/loss', avg_loss, it)
        writer.add_scalar('val/loss_mol', avg_mol_loss, it)
        writer.add_scalar('val/loss_mol_pos', avg_mol_loss_pos, it)
        writer.add_scalar('val/loss_mol_v', avg_mol_loss_v, it)
        writer.add_scalar('val/loss_mesh_constrained', avg_loss_mesh_constrained, it)

        writer.flush()
        return avg_loss

    try:
        for it in range(1, args.max_iters + 1):
            train(it)
            if it % args.val_freq == 0 or it == args.max_iters:
                val_loss = validate(it)
                ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                torch.save({
                    'config': vars(args),
                    'model_mol': model_mol.state_dict(),
                    'model_mesh': model_mesh.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'iteration': it,
                }, ckpt_path)

    except KeyboardInterrupt:
        logger.info('Terminating...')



if __name__ == '__main__':
   main()




