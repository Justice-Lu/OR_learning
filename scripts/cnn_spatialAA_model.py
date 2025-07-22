import os 
import sys
import pandas as pd 
import numpy as np 


OR_LEARNING_PATH = os.path.join(os.getcwd().split('OR_learning')[0], 'OR_learning/')
sys.path.insert(0, os.path.join(OR_LEARNING_PATH, 'utils/'))

import plot_functions as pf
import color_function as cf 
import voxel_functions as vf 




Cbc_cav_coords = pd.read_pickle('/mnt/data2/Justice/OR_learning/files/binding_cavity/dict_Cbc_cav_coords.pkl')
Cbc_res_coords = pd.read_pickle('/mnt/data2/Justice/OR_learning/files/binding_cavity/dict_Cbc_res_coords.pkl')

voxels, grid_shape = vf.voxelize_cavity(cavity_coords = Cbc_cav_coords,
                                      residue_coords = Cbc_res_coords,
                                      resolution = 1, 
                                      encode_method='ohe', 
                                      vdw_radius = True, 
                                      sparse_mode = False)
print(f'Grid shape: {grid_shape}')

# Save memory
del Cbc_cav_coords


"""
Prepare cid features 
"""

# Load pS6-IP data 
ps6_df = pd.read_csv('/mnt/data2/Justice/OR_learning/files/pS6IP/pS6IP_MASTER_HL_Annotated_2025.csv', index_col = 0) 
# Subset for concentration in percentages 
ps6_df = ps6_df[ps6_df.concentration.str.contains('p')]
ps6_df = ps6_df.sort_values(['Family', 'DL_OR', 'odor_category', 'odor', 'concentration', 'FDR', 'logFC_adj_zscore']).dropna()
# Subset for ORs in voxel ORs.  
ps6_df = ps6_df[ps6_df.DL_OR.isin(Cbc_res_coords.keys())]




import pubchempy as pcp
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.decomposition import PCA
import torch 


def cid_to_smiles(cid):
    try:
        compound = pcp.Compound.from_cid(cid)
        return compound.isomeric_smiles
    except:
        return None
    
def smiles_to_ecfp4(smiles, n_bits=1028):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits))
    else:
        return None


if not os.path.exists('/mnt/data2/Justice/OR_learning/output/ligand_fp_tensor.pt'): 
    # Generate smiles from pS6-IP cid 
    unique_cids = ps6_df.cid.unique()
    cid_smiles_dict = {cid: cid_to_smiles(str(cid)) for cid in tqdm(unique_cids)}

    # Create fingerprint from smiles 
    ligand_fp_dict = {}
    for cid, smiles in tqdm(cid_smiles_dict.items()):
        if smiles:
            fp = smiles_to_ecfp4(smiles)
            if fp is not None:
                ligand_fp_dict[cid] = fp

    ligand_fp_df = pd.DataFrame.from_dict(ligand_fp_dict, orient='index')
    ligand_fp_df.index.name = 'cid'
    pca = PCA(n_components=32)

    ligand_fp_tensor = torch.tensor(pca.fit_transform(ligand_fp_df.values), dtype=torch.float32)

    print(f'PCA explained variance: {np.sum(pca.explained_variance_ratio_)}')
else: 
    cid_smiles_dict = np.load('/mnt/data2/Justice/OR_learning/output/cid_smiles_dict.npy', allow_pickle=True).item()
    ligand_fp_tensor = torch.load('/mnt/data2/Justice/OR_learning/output/ligand_fp_tensor.pt')


# CNN MODEL
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from datetime import datetime

# ---------- Set Up Data ----------
class ORLigandIndexDataset(Dataset):
    def __init__(self, pairs_df, voxel_tensor, ligand_fp_tensor, or_index_map, ligand_index_map):
        self.voxel_tensor = voxel_tensor
        self.ligand_fp_tensor = ligand_fp_tensor
        self.targets = []
        self.indices = []

        for _, row in pairs_df.iterrows():
            or_id = row['DL_OR']
            cid = row['cid']
            if or_id in or_index_map and cid in ligand_index_map:
                self.indices.append((or_index_map[or_id], ligand_index_map[cid]))
                self.targets.append(row['logFC_adj_zscore'])

        self.targets = torch.tensor(self.targets, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        or_idx, ligand_idx = self.indices[idx]
        voxel = self.voxel_tensor[or_idx]           # [D, H, W, C]
        ligand = self.ligand_fp_tensor[ligand_idx]  # [F]
        target = self.targets[idx]                  # scalar
        return voxel, ligand, target

    
import torch.nn as nn
import torch.nn.functional as F

# ---------- Model ----------
class ORLigandCNN(nn.Module):
    def __init__(self, voxel_shape=(41, 42, 65), ligand_dim=64,
                 kernel_size=3, hidden_dim=256,
                 num_conv_layers=2, conv_channels=(16, 32),
                 dropout=0.0, pool_type='AVG'):
        super().__init__()
        C, D, H, W = voxel_shape
        layers = []
        in_channels = C
        
        # Select pooling class
        if pool_type == 'AVG':
            pool = nn.AvgPool3d
        elif pool_type == 'MAX':
            pool = nn.MaxPool3d
        else:
            raise ValueError(f"Unsupported pool_type: {pool_type}")

        for i in range(num_conv_layers):
            out_channels = conv_channels[i]
            layers.append(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2))
            layers.append(nn.ReLU())
            layers.append(pool(2))  # Dynamic pooling
            in_channels = out_channels

        self.cnn = nn.Sequential(*layers)

        # Compute CNN output size
        dummy = torch.zeros(1, C, D, H, W)
        with torch.no_grad():
            out = self.cnn(dummy)
        self.cnn_out_dim = out.reshape(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(self.cnn_out_dim + ligand_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, voxel, ligand):
        voxel = voxel.permute(0, 4, 1, 2, 3)  # [B, D, H, W, C] → [B, C, D, H, W]
        cnn_out = self.cnn(voxel)
        cnn_out = cnn_out.reshape(cnn_out.size(0), -1)
        combined = torch.cat([cnn_out, ligand], dim=1)
        return self.fc(combined)

# ---------- Prepare Data ----------
# voxel_tensor: [N_or, D, H, W, C]
# ligand_fp_tensor: [N_ligand, F]
# ps6_df: full OR-ligand response table with DL_OR, cid, logFC_adj_zscore
# Cbc_res_coords, cid_smiles_dict: OR and ligand reference maps

or_ids = list(Cbc_res_coords.keys())
ligand_cids = list(cid_smiles_dict.keys())
or_index_map = {or_id: i for i, or_id in enumerate(or_ids)}
ligand_index_map = {cid: i for i, cid in enumerate(ligand_cids)}
voxel_tensor = torch.stack(voxels)

# Set your desired target ratio
# POSITIVE_WEIGHT = 0.5  

# # Separate positives and zeros
# positive_df = ps6_df[ps6_df['logFC_adj_zscore'] >= 2]
# negative_df = ps6_df[ps6_df['logFC_adj_zscore'] < 2]

# # Number of samples to take
# total_samples = 1000
# n_pos = int(total_samples * POSITIVE_WEIGHT)
# n_neg = total_samples - n_pos

# # Sample with controlled balance
# sampled_positives = positive_df.sample(n=min(n_pos, len(positive_df)), random_state=0)
# sampled_negatives = negative_df.sample(n=n_neg, random_state=0)
# subset_df = pd.concat([sampled_positives, sampled_negatives]).sample(frac=1.0, random_state=0)

# # Now split train/val
# train_df, test_df = train_test_split(subset_df, test_size=0.2, random_state=0)
# train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=0)

# train_dataset = ORLigandIndexDataset(train_df, voxel_tensor, ligand_fp_tensor, or_index_map, ligand_index_map)
# val_dataset = ORLigandIndexDataset(val_df, voxel_tensor, ligand_fp_tensor, or_index_map, ligand_index_map)
# test_dataset = ORLigandIndexDataset(test_df, voxel_tensor, ligand_fp_tensor, or_index_map, ligand_index_map)

def create_balanced_datasets(ps6_df, voxel_tensor, ligand_fp_tensor,
                             or_index_map, ligand_index_map,
                             total_samples=1000, positive_weight=0.5, seed=None):

    positive_df = ps6_df[ps6_df['logFC_adj_zscore'] >= 2]
    negative_df = ps6_df[ps6_df['logFC_adj_zscore'] < 2]

    n_pos = int(total_samples * positive_weight)
    n_neg = total_samples - n_pos

    sampled_pos = positive_df.sample(n=min(n_pos, len(positive_df)), random_state=seed)
    sampled_neg = negative_df.sample(n=n_neg, random_state=seed)

    subset_df = pd.concat([sampled_pos, sampled_neg]).sample(frac=1.0, random_state=seed)
    train_df, test_df = train_test_split(subset_df, test_size=0.2, random_state=seed)
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=seed)

    train_dataset = ORLigandIndexDataset(train_df, voxel_tensor, ligand_fp_tensor, or_index_map, ligand_index_map)
    val_dataset = ORLigandIndexDataset(val_df, voxel_tensor, ligand_fp_tensor, or_index_map, ligand_index_map)
    test_dataset = ORLigandIndexDataset(test_df, voxel_tensor, ligand_fp_tensor, or_index_map, ligand_index_map)

    return train_dataset, val_dataset, test_dataset

# # Dataloaders
# train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=8)

# ---------- Hyperparameter Random Sampling ----------
import random
from itertools import product
import matplotlib.pyplot as plt
import json
import os


param_choices = {
        'pooling': ['MAX', 'AVG'], 
        'lr': [1e-2, 5e-3, 1e-3, 5e-4, 1e-4],  
        'batch_size': [4, 8, 16, 32],
        'hidden_dim': [64, 128, 256, 512, 1024],
        'kernel_size': [1, 3, 5, 7],
        'num_conv_layers': [1, 2, 3, 4],
        'dropout': [0.0, 0.2, 0.3, 0.5, 0.7]
}

def make_run_id(param_dict):
    # Abbreviation map
    abbr = {
        'lr': 'lr',
        'batch_size': 'bs',
        'hidden_dim': 'hd',
        'kernel_size': 'ks',
        'num_conv_layers': 'nl',
        'conv_channels': 'cc',
        'dropout': 'do',
        'pooling': 'pt',
    }
    def format_val(v):
        if isinstance(v, (list, tuple)):
            return "-".join(map(str, v))
        return str(v)

    parts = [f"{abbr.get(k, k)}{format_val(v)}" for k, v in param_dict.items()]
    return "_".join(parts)


num_trials = 10  # number of random configs to test
param_list = list(product(*param_choices.values()))
random.shuffle(param_list)
sampled_params = param_list[:num_trials]
param_keys = list(param_choices.keys())

voxel_shape = voxel_tensor.shape  # [N, D, H, W, C]
C, D, H, W = voxel_shape[-1], voxel_shape[1], voxel_shape[2], voxel_shape[3]

# Saving directory
save_path = '/mnt/data2/Justice/OR_learning/output/cnn_spatialAA/'

for i, param_values in enumerate(sampled_params):
    param_dict = dict(zip(param_keys, param_values))
    lr = param_dict['lr']
    batch_size = param_dict['batch_size']
    hidden_dim = param_dict['hidden_dim']
    kernel_size = param_dict['kernel_size']

    # Check if this hyperparameter combination already exists
    run_id = make_run_id(param_dict)
    # run_id = f"lr{lr}_bs{batch_size}_hd{hidden_dim}_ks{kernel_size}"
    if any(run_id in dirname for dirname in os.listdir(save_path)):
        print(f"Skipping existing run with {run_id}")
        continue
    
    # Otherwise, proceed
    run_name = os.path.join(save_path, 
                            f"{run_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_name, exist_ok=True)
    with open(os.path.join(run_name, "params.json"), "w") as f:
        json.dump(param_dict, f, indent=4)

    # Model, optimizer, loss
    model = ORLigandCNN(
        voxel_shape=(C, D, H, W),
        ligand_dim=ligand_fp_tensor.shape[1],
        kernel_size=kernel_size,
        hidden_dim=hidden_dim,
        # num_conv_layers=param_dict['num_conv_layers'],
        # conv_channels=param_dict['conv_channels'],
        num_conv_layers = param_dict['num_conv_layers'],
        conv_channels = tuple([16 * 2**i for i in range(param_dict['num_conv_layers'])]),
        dropout=param_dict['dropout'], 
        pool_type=param_dict['pooling']
    )   
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size)

    train_dataset, val_dataset, test_dataset = create_balanced_datasets(
        ps6_df, voxel_tensor, ligand_fp_tensor, or_index_map, ligand_index_map,
        total_samples=1000, positive_weight=0.5, seed=i  # seed varies per trial
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    best_val_loss = float('inf')
    best_model_path = os.path.join(run_name, "best_model.pt")
    train_losses = []
    val_losses = []

    print(f"\n Training {run_name}")
    for epoch in range(20):
        
        # Dynamic training settings
        initial_epochs = 20
        max_epochs = 100
        patience = 5
        min_delta = 1e-4  # Minimum change in val loss to be considered an improvement

        best_val_loss = float('inf')
        best_model_path = os.path.join(run_name, "best_model.pt")
        train_losses = []
        val_losses = []
        epochs_no_improve = 0
        epoch = 0

        print(f"\n Training {run_name}")
        while epoch < max_epochs:
            epoch += 1
            model.train()
            total_train_loss = 0
            train_preds_epoch, train_targets_epoch = [], []

            for vox, lig, y in train_loader:
                pred = model(vox, lig)
                loss = loss_fn(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

                train_preds_epoch.append(pred.detach().squeeze().cpu().numpy())
                train_targets_epoch.append(y.squeeze().cpu().numpy())

            model.eval()
            total_val_loss = 0
            val_preds_epoch, val_targets_epoch = [], []

            with torch.no_grad():
                for vox, lig, y in val_loader:
                    pred = model(vox, lig)
                    loss = loss_fn(pred, y)
                    total_val_loss += loss.item()
                    val_preds_epoch.append(pred.squeeze().cpu().numpy())
                    val_targets_epoch.append(y.squeeze().cpu().numpy())

            avg_train_loss = total_train_loss / len(train_loader)
            avg_val_loss = total_val_loss / len(val_loader)
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            print(f"Epoch {epoch:2d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

            # Save best model
            if avg_val_loss + min_delta < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), best_model_path)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # Stop early if no improvement
            if epoch >= initial_epochs and epochs_no_improve >= patience:
                print(f"Early stopping after {epoch} epochs. No significant improvement in {patience} epochs.")
                break

    # Plot and save training/validation curves
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training vs Validation Loss")
    plt.savefig(os.path.join(run_name, "loss_curve.png"))
    plt.close()
    
    # Final scatter plot for last epoch val predictions
    val_preds = np.concatenate(val_preds_epoch)
    val_targets = np.concatenate(val_targets_epoch)

    # Save raw arrays
    train_preds = np.concatenate(train_preds_epoch)
    train_targets = np.concatenate(train_targets_epoch)
    val_preds = np.concatenate(val_preds_epoch)
    val_targets = np.concatenate(val_targets_epoch)

    # Save arrays
    np.save(os.path.join(run_name, "train_preds.npy"), train_preds)
    np.save(os.path.join(run_name, "train_targets.npy"), train_targets)
    np.save(os.path.join(run_name, "val_preds.npy"), val_preds)
    np.save(os.path.join(run_name, "val_targets.npy"), val_targets)

    # Combined scatter plot
    plt.figure(figsize=(12, 5))

    # Training
    plt.subplot(1, 2, 1)
    plt.scatter(train_targets, train_preds, alpha=0.5)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("Train: Predictions vs Targets")
    plt.plot([train_targets.min(), train_targets.max()],
            [train_targets.min(), train_targets.max()], 'r--')

    # Validation
    plt.subplot(1, 2, 2)
    plt.scatter(val_targets, val_preds, alpha=0.5)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("Validation: Predictions vs Targets")
    plt.plot([val_targets.min(), val_targets.max()],
            [val_targets.min(), val_targets.max()], 'r--')

    plt.tight_layout()
    plt.savefig(os.path.join(run_name, "scatter_train_val.png"))
    plt.close()
    

# ---------- Final Evaluation ----------
# print("\nEvaluating best saved model...")
# model = ORLigandCNN(voxel_shape=(C, D, H, W), ligand_dim=ligand_fp_tensor.shape[1])
# model.load_state_dict(torch.load(save_path))
# model.eval()

# test_loader = DataLoader(test_dataset, batch_size=16)
# all_preds, all_targets = [], []

# with torch.no_grad():
#     for vox, lig, y in test_loader:
#         pred = model(vox, lig)
#         all_preds.append(pred.squeeze().numpy())
#         all_targets.append(y.squeeze().numpy())

# preds = np.concatenate(all_preds)
# targets = np.concatenate(all_targets)

# from scipy.stats import pearsonr
# r, _ = pearsonr(preds, targets)
# mse = np.mean((preds - targets) ** 2)

# print(f"✅ Final Test MSE: {mse:.4f}")
# print(f"✅ Final Test Pearson r: {r:.3f}")

