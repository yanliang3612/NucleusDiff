<h1 align="center">NucleusDiff: Manifold-Constrained Nucleus-Level Denoising Diffusion Model for Structure-Based Drug Design</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2409.10584"><img src="https://img.shields.io/badge/arXiv-2409.10584-b31b1b" alt="arXiv" /></a>
  <a href="https://yanliang3612.github.io/NucleusDiff/"><img src="https://img.shields.io/badge/Project-Page-0A66C2" alt="Project Page" /></a>
  <a href="https://www.pnas.org/doi/10.1073/pnas.2415666122"><img src="https://img.shields.io/badge/Paper-PNAS-B31B1B" alt="PNAS Paper" /></a>
  <a href="https://www.caltech.edu/about/news/new-ai-model-for-drug-design-brings-more-physics-to-bear-in-predictions"><img src="https://img.shields.io/badge/Caltech-News-FF6C0C" alt="Caltech News" /></a>
  <a href="https://doi.org/10.5281/zenodo.17093932"><img src="https://img.shields.io/badge/DOI-Zenodo-1682D4" alt="Zenodo DOI" /></a>
  <a href="https://join.slack.com/t/matdiscoverai/shared_invite/zt-32kktcuk0-XaaJT2P9qZTfNdaCzJUGAg"><img src="https://img.shields.io/badge/Slack-Join_SciGenAI-4A154B?logo=slack&amp;logoColor=white" alt="Join SciGenAI on Slack" /></a>
</p>

<p align="center">
  The official implementation of the PNAS 2025 paper <em>Manifold-Constrained Nucleus-Level Denoising Diffusion Model for Structure-Based Drug Design</em>.
</p>

<p align="center">
  Shengchao Liu<sup>*</sup>, Liang Yan<sup>*</sup>, Weitao Du, Weiyang Liu, Zhuoxinran Li, Hongyu Guo,<br />
  Christian Borgs, Jennifer Chayes, Anima Anandkumar
</p>

<p align="center">
  <strong>Proceedings of the National Academy of Sciences (PNAS), 2025</strong><br />
  <sup>*</sup>Equal contribution
</p>

<p align="center">
  <a href="#community">
    <img src="https://readme-typing-svg.demolab.com?font=Inter&amp;weight=700&amp;size=18&amp;pause=1200&amp;color=0A66C2&amp;center=true&amp;vCenter=true&amp;width=860&amp;lines=Join+the+SciGenAI+Community+for+NucleusDiff;Real-time+Q%26A+%E2%80%A2+code+contributions+%E2%80%A2+pull+requests;Discuss+%26+collaborate+across+generative+AI+for+science" alt="Join the SciGenAI Community for NucleusDiff" />
  </a>
</p>

<p align="center">
  <a href="https://join.slack.com/t/matdiscoverai/shared_invite/zt-32kktcuk0-XaaJT2P9qZTfNdaCzJUGAg">
    <img src="assets/scigenai-slack-qr.png" alt="QR code to join the SciGenAI Slack community" width="180" />
  </a>
  <br />
  <a href="https://join.slack.com/t/matdiscoverai/shared_invite/zt-32kktcuk0-XaaJT2P9qZTfNdaCzJUGAg"><strong>Scan the QR code or join via the Slack invitation link</strong></a>
</p>

<p align="center">
  <img src="assets/inference.gif" width="100%" alt="NucleusDiff inference process" />
</p>

## Update & News

- 🌐 **2024-06-18 — Project page launched.** Explore the model, visualizations, paper, and code on the [NucleusDiff project page](https://yanliang3612.github.io/NucleusDiff/).
- 📄 **2024-09-16 — arXiv v1 released.** The first preprint of NucleusDiff is available as [arXiv:2409.10584v1](https://arxiv.org/abs/2409.10584v1).
- 🔄 **2024-09-30 — arXiv v2 released.** The revised preprint is available as [arXiv:2409.10584v2](https://arxiv.org/abs/2409.10584v2).
- 🎉 **2025-08-11 — Accepted by PNAS.** NucleusDiff was accepted by the *Proceedings of the National Academy of Sciences*; read the [PNAS paper](https://www.pnas.org/doi/10.1073/pnas.2415666122).
- 💻 **2025-09-10 — Code open sourced.** We released the official NucleusDiff implementation as [v1.0.0 on GitHub](https://github.com/yanliang3612/NucleusDiff/releases/tag/v1.0.0).
- 📰 **2025-10-20 — Featured by Caltech News.** Caltech highlighted NucleusDiff in [“New AI Model for Drug Design Brings More Physics to Bear in Predictions”](https://www.caltech.edu/about/news/new-ai-model-for-drug-design-brings-more-physics-to-bear-in-predictions).
- 💬 **2026-07-16 — SciGenAI Slack channel launched.** We opened a dedicated NucleusDiff channel for real-time questions, discussions, code contributions, and collaboration. [Join the SciGenAI Slack community](https://join.slack.com/t/matdiscoverai/shared_invite/zt-32kktcuk0-XaaJT2P9qZTfNdaCzJUGAg)—everyone is welcome!

## Overview

NucleusDiff is a manifold-constrained denoising diffusion model for structure-based drug design. It jointly models atomic nuclei and their surrounding electron-cloud manifolds to reduce atomic collisions while generating high-affinity ligands.

## Contents

- [Update & News](#update--news)
- [Overview](#overview)
- [Installation](#1-installation)
- [Data Preparation](#2-data-preparation)
- [CrossDocked2020 Experiments](#3-crossdocked2020-experiments)
- [Therapeutic Target Experiments](#4-therapeutic-target-experiments)
- [Universal Inference](#5-universal-inference-for-a-specified-protein)
- [SciGenAI Community](#community)
- [Citation](#citation)

## 1. Installation

We recommend using the Conda environments below to reproduce our results. For full evaluation or training logs, contact [yanliangfdu@gmail.com](mailto:yanliangfdu@gmail.com).

### 1.1 Main experiment dependencies

The code has been tested in the following environment:

| Package           | Version   |
|-------------------|-----------|
| Python            | 3.8.13    |
| PyTorch           | 1.12.1    |
| CUDA              | 11.0      |
| PyTorch Geometric | 2.5.2     |
| RDKit             | 2021.03.1b1 |


Install via Conda and Pip:

```bash
conda create -n "nucleusdiff" python=3.8.13
conda activate nucleusdiff
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install torch_geometric
pip install https://data.pyg.org/whl/torch-1.12.0%2Bcu113/pyg_lib-0.3.1%2Bpt112cu113-cp38-cp38-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_cluster-1.6.0%2Bpt112cu113-cp38-cp38-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_scatter-2.1.0%2Bpt112cu113-cp38-cp38-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_sparse-0.6.16%2Bpt112cu113-cp38-cp38-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_spline_conv-1.2.1%2Bpt112cu113-cp38-cp38-linux_x86_64.whl
conda install rdkit/label/nightly::rdkit
conda install openbabel tensorboard pyyaml easydict python-lmdb -c conda-forge
pip install wandb
pip install pytorch-lightning==2.1.3
pip install matplotlib
pip install numpy==1.23
pip install accelerate
pip install transformers


# For Vina Docking
pip install meeko==0.1.dev3 scipy pdb2pqr vina==1.2.2
python -m pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3
```
The code should work with PyTorch >= 1.9.0 and PyG >= 2.0. You can adjust the package versions for your environment.


### 1.2 Manifold preprocessing dependencies

Use this separate environment only if you want to process the CrossDocked manifold dataset from scratch.

```bash
# We recommend using conda for environment management
conda create -n Manifold python=3.7.3
conda activate Manifold

pip install -r ./crossdock_manifold_data_preparation/requirements.txt
# install PyMesh for surface mesh processing
PYMESH_PATH="~/PyMesh" # substitute with your own PyMesh path
git clone https://github.com/PyMesh/PyMesh.git $PYMESH_PATH 
cd $PYMESH_PATH 
git submodule update --init
apt-get update
# make sure you have these libraries installed before building PyMesh
apt-get install cmake libgmp-dev libmpfr-dev libgmpxx4ldbl libboost-dev libboost-thread-dev libopenmpi-dev
cd $PYMESH_PATH/third_party
python build.py all # build third party dependencies
cd $PYMESH_PATH
mkdir build
cd build
cmake ..
make -j # check for missing third-party dependencies if failed to make
cd $PYMESH_PATH
python setup.py install
python -c "import pymesh; pymesh.test()"

# install meshplot
conda install -c conda-forge meshplot

# install libigl
conda install -c conda-forge igl

# download MSMS
MSMS_PATH="~/MSMS" # substitute with your own MSMS path
wget https://ccsb.scripps.edu/msms/download/933/ -O msms_i86_64Linux2_2.6.1.tar.gz
mkdir -p $MSMS_PATH # mark this directory as your $MSMS_bin for later use
tar zxvf msms_i86_64Linux2_2.6.1.tar.gz -C $MSMS_PATH

# install PyTorch 1.10.0 (e.g., with CUDA 11.3)
conda install pytorch==1.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu113.html

# install Manifold
pip install -e . 
```


---

## 2. Data Preparation

### 2.1 CrossDocked2020 data

1. The data used for training / evaluating the model are organized in the [nucleusdiff_data_and_checkpoint](https://drive.google.com/drive/folders/1boX4IOC-WVJ5zWLy2ulRGvDClN7ukUOe?usp=sharing) Google Drive folder.

2. To train the model from scratch, download the preprocessed LMDB and split files:

   - `crossdocked_v1.1_rmsd1.0_pocket10_processed_w_manifold_data_version.lmdb`
   - `crossdocked_pocket10_pose_w_manifold_data_split.pt`

3. To evaluate the model on the test set, you need to download _and_ unzip the `test_set.zip`. It includes the original PDB files that will be used in Vina Docking.

4. If you want to process the dataset from scratch, you need to download CrossDocked2020 v1.1 from [here](https://bits.csb.pitt.edu/files/crossdock2020/), save it into `./data/CrossDocked2020`, and run the scripts in `./crossdock_data_preparation`:

#### Process CrossDocked2020 from scratch

1. [`step1_clean_crossdocked.py`](./crossdock_data_preparation/step1_clean_crossdocked.py) filters the original dataset and keeps entries with RMSD < 1 Å. It generates `index.pkl` and a directory containing the filtered data (corresponding to `crossdocked_v1.1_rmsd1.0.tar.gz` in Google Drive). You can skip this step if you downloaded the preprocessed LMDB file.

```bash
python ./crossdock_data_preparation/step1_clean_crossdocked.py \
       --source "./data/CrossDocked2020" \
       --dest "./data/crossdocked_v1.1_rmsd1.0" \
       --rmsd_thr 1.0
```

2. [`step2_extract_pockets.py`](./crossdock_data_preparation/step2_extract_pockets.py) clips each protein file to a 10 Å region around the binding molecule.

```bash
python ./crossdock_data_preparation/step2_extract_pockets.py \
       --source "./data/crossdocked_v1.1_rmsd1.0" \
       --dest "./data/crossdocked_v1.1_rmsd1.0_pocket10"
```

3. [`step3_split_pl_dataset.py`](./crossdock_data_preparation/step3_split_pl_dataset.py) creates the training and test splits. We use the same `split_by_name.pt` as [AR](https://arxiv.org/abs/2203.10446) and [Pocket2Mol](https://arxiv.org/abs/2205.07249); it is also available in the Google Drive data folder.

```bash
python ./crossdock_data_preparation/step3_split_pl_dataset.py \
       --path "./data/crossdocked_v1.1_rmsd1.0_pocket10" \
       --dest "./data/crossdocked_pocket10_pose_split.pt" \
       --fixed_split "./data/split_by_name.pt"
```

### 2.2 CrossDocked manifold data

1. Activate the manifold preprocessing environment:

```bash
conda activate Manifold
```

2. Prepare input for MSMS:

```bash
python step1_convert_npz_to_xyzrn.py \
       --crossdock_source [path/to/crossdock_pocket10_auxdata/] \
       --out_root "./data/crossdocked_pocket10_mesh"
```

3. Run MSMS to generate molecular surfaces:

```bash
python step2_compute_msms.py \
       --data_root "./data/crossdocked_pocket10_mesh" \
       --msms-bin [path/to/MSMS/dir]/msms.x86_64Linux2.2.6.1 
```

4. Refine the surface meshes:

```bash
python step3_refine_mesh.py \
       --data_root "./data/crossdocked_pocket10_mesh"
```

### 2.3 Generate the final LMDB and split files

```bash
python ./datasets/pl_pair_dataset.py \
       --data_root "./data/crossdocked_v1.1_rmsd1.0_pocket10"
```

---

## 3. CrossDocked2020 Experiments

### 3.1 Training

```bash
python train.py \
       --lr 0.001 \
       --device "cuda:0" \
       --wandb_project_name "nucleusdiff_train" \
       --loss_mesh_constained_weight 1
```

**Note:** Our pretrained models are available in the [nucleusdiff_data_and_checkpoint](https://drive.google.com/drive/folders/1boX4IOC-WVJ5zWLy2ulRGvDClN7ukUOe?usp=sharing) Google Drive folder.


### 3.2 Inference

```bash
python sample_for_crossdock.py \
       --ckpt_path "./logs_diffusion/nucleusdiff_train" \
       --ckpt_it 100000 \
       --cuda_device 0 \
       --data_id 0 
```

You can also speed up sampling with multiple GPUs, e.g.:

```bash
python sample_for_crossdock.py \
       --ckpt_path "./logs_diffusion/nucleusdiff_train" \
       --ckpt_it 100000 \
       --cuda_device 0 \
       --data_id 0

python sample_for_crossdock.py \
       --ckpt_path "./logs_diffusion/nucleusdiff_train" \
       --ckpt_it 100000 \
       --cuda_device 1 \
       --data_id 1 

python sample_for_crossdock.py \
       --ckpt_path "./logs_diffusion/nucleusdiff_train" \
       --ckpt_it 100000 \
       --cuda_device 2 \
       --data_id 2

python sample_for_crossdock.py \
       --ckpt_path "./logs_diffusion/nucleusdiff_train" \
       --ckpt_it 100000 \
       --cuda_device 3 \
       --data_id 3 
```

### 3.3 General metrics

```bash
python ./evaluation/evaluate_for_crossdock_on_general_metrics.py \
        --sample_path "./result_output" \
        --eval_step -1 \
        --protein_root "./data/test_set" \
        --docking_mode "vina_dock"
```

### 3.4 Collision metrics

```bash
python ./evaluation/evaluate_for_crossdock_on_collision_metrics.py \
        --sample_path "./result_output" \
        --eval_step -1
```

---

## 4. Therapeutic Target Experiments

### 4.1 Data preparation

If you want to process the dataset from scratch, you need to download `real_world.zip` from [nucleusdiff_data_and_checkpoint](https://drive.google.com/drive/folders/1boX4IOC-WVJ5zWLy2ulRGvDClN7ukUOe?usp=sharing), save it into `./data`, and run the scripts in `./covid_19_data_preparation`:

```bash
python ./covid_19_data_preparation/extract_pockets_for_real_world.py \
        --source "./data/real_world" \
        --dest "./real_world_test_extract_pockets"
```

### 4.2 Inference

```bash
python sample_for_covid_19.py \
        --checkpoint [path/to/nucleusdiff/checkpoint] \
        --pdb_path "./real_world_test_extract_pockets/CDK2/cdk2_ligand_pocket10.pdb" \
        --result_path "./read_world_cdk2_test" \
        --sample_num_atoms "real_world_testing" \
        --inference_num_atoms 30
```

### 4.3 General metrics

```bash
python ./evaluation/evaluate_for_covid_19_on_general_metrics.py \
        --sample_path "./read_world_cdk2_test" \
        --protein_root "./real_world/cdk2_processed.pdb" \
        --ligand_filename "CDK2" \
        --docking_mode "vina_dock"
```

### 4.4 Collision metrics

```bash
python ./evaluation/evaluate_for_covid_19_on_collision_metrics.py \
        --sample_path "./read_world_cdk2_test" \
        --model "nucleusdiff_train" \
        --target "cdk2_test"
```

## 5. Universal Inference for a Specified Protein

Use `sample_for_specific_protein.py` to generate ligands for an arbitrary single protein pocket PDB.

### 5.1 Input preparation

1. Prepare a pocket PDB centered at the binding site (e.g., 10 Å around the ligand or binding residues).  
   You may reuse the script in 4.1: `./covid_19_data_preparation/extract_pockets_for_real_world.py`.
2. Example pocket file: `./specific_protein/3cl_ligand_pocket10.pdb`.

### 5.2 Inference

```bash
python sample_for_specific_protein.py \
        --checkpoint ./checkpoints/nucleusdiff_pretrained_model.pt \
        --pdb_path ./specific_protein/3cl_ligand_pocket10.pdb \
        --result_path ./results_specific_protein \
        --sample_num_atoms real_world_testing \
        --inference_num_atoms 30 \
        --num_samples 1000 \
        --num_steps 1000 \
        --device cuda:0
```

Key arguments:

- `--checkpoint`: path to a NucleusDiff checkpoint (`.pt`).
- `--pdb_path`: pocket PDB for your target protein.
- `--result_path`: output directory.
- `--sample_num_atoms`: set to `real_world_testing` to use a fixed atom count.
- `--inference_num_atoms`: atoms per generated ligand when using `real_world_testing`.
- `--num_samples`: number of ligands to generate.
- `--num_steps`: diffusion steps (trade-off between quality and speed).
- `--device`: GPU device, e.g., `cuda:0`.

### 5.3 Outputs

- `${result_path}/sample_{test_time}.pt`: raw tensors and sampling trajectories.
- `${result_path}/sdf/*.sdf`: reconstructed molecules in SDF format.

Run `python sample_for_specific_protein.py --help` for the complete list of options and defaults.

---

<a id="community"></a>

## SciGenAI Community

### Join the NucleusDiff Slack channel

SciGenAI is a community platform for discussion and collaboration around **generative AI for science**. It connects researchers, students, engineers, and open-source contributors working across NucleusDiff and other AI4Science projects.

The workspace includes a dedicated **`NucleusDiff` channel** where you can:

- Ask NucleusDiff questions and get real-time community support.
- Discuss the paper, implementation details, experiments, and reproducibility.
- Coordinate code contributions, commits, issues, code reviews, and pull requests.
- Connect with other generative-AI-for-science projects and find collaborators.

**Option 1 — Invitation link:** [Join the SciGenAI Slack community](https://join.slack.com/t/matdiscoverai/shared_invite/zt-32kktcuk0-XaaJT2P9qZTfNdaCzJUGAg)

**Option 2 — QR code:** Scan the code below with your phone. The QR image is also clickable.

<p align="center">
  <a href="https://join.slack.com/t/matdiscoverai/shared_invite/zt-32kktcuk0-XaaJT2P9qZTfNdaCzJUGAg">
    <img src="assets/scigenai-slack-qr.png" alt="QR code to join the SciGenAI Slack community" width="220" />
  </a>
</p>

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{liu2025manifold,
  title={Manifold-constrained nucleus-level denoising diffusion model for structure-based drug design},
  author={Liu, Shengchao and Yan, Liang and Du, Weitao and Liu, Weiyang and Li, Zhuoxinran and Guo, Hongyu and Borgs, Christian and Chayes, Jennifer and Anandkumar, Anima},
  journal={Proceedings of the National Academy of Sciences},
  volume={122},
  number={41},
  pages={e2415666122},
  year={2025},
  publisher={National Academy of Sciences}
}
```
