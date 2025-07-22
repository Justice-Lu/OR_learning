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
                # Convert to binary label: 1 if binding (zscore >= 2), else 0
                self.targets.append(1.0 if row['logFC_adj_zscore'] >= 2 else 0.0)

        self.targets = torch.tensor(self.targets, dtype=torch.float32).unsqueeze(1)  # [N, 1]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        or_idx, ligand_idx = self.indices[idx]
        voxel = self.voxel_tensor[or_idx]           # [D, H, W, C]
        ligand = self.ligand_fp_tensor[ligand_idx]  # [F]
        target = self.targets[idx]                  # [1]
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
        
        pool = nn.AvgPool3d if pool_type == 'AVG' else nn.MaxPool3d

        for i in range(num_conv_layers):
            out_channels = conv_channels[i]
            layers.append(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2))
            layers.append(nn.ReLU())
            layers.append(pool(2))
            in_channels = out_channels

        self.cnn = nn.Sequential(*layers)

        # Compute CNN output size
        dummy = torch.zeros(1, C, D, H, W)
        with torch.no_grad():
            out = self.cnn(dummy)
        self.cnn_out_dim = out.reshape(1, -1).size(1)  

        self.fc = nn.Sequential(
            nn.Linear(self.cnn_out_dim + ligand_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)  # Output: logit for binary classification
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
import argparse
import matplotlib.pyplot as plt
import json
import os

from sklearn.metrics import accuracy_score, roc_auc_score
from datetime import datetime
import numpy as np

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

def main():
    
    """
    Parameter search for the hyperparameters below. 
    
    Unless, directory for param directory path is specified. . . 
    """
    param_choices = {
        'pooling': ['MAX', 'AVG'], 
        'lr': [1e-2, 5e-3, 1e-3, 5e-4, 1e-4],  
        'batch_size': [4, 8, 16, 32],
        'hidden_dim': [64, 128, 256, 512, 1024],
        'kernel_size': [1, 3, 5, 7],
        'num_conv_layers': [1, 2, 3, 4],
        'dropout': [0.0, 0.2, 0.3, 0.5, 0.7]
    }
    
    

    # Compute all combinations
    all_params = list(product(*param_choices.values()))
    total_combinations = len(all_params)

    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_trials", type=int, default=total_combinations,
                        help=f"Number of hyperparameter trials (default: {total_combinations})")
    parser.add_argument("--patience", type=int, default=10,
                        help=f"Number of no improvement epochs to tolerate (default: 10)")
    parser.add_argument("--min_epoch", type=int, default=40,
                        help=f"Minumum number of epochs to run (default: {40})")
    parser.add_argument("--max_epoch", type=int, default=100,
                        help=f"Maximum number of epochs to learn up to (default: {100})")
    parser.add_argument("--param_dir", default=None,
                        help=f"Absolute Path to the directory with params to test. ")
    parser.add_argument("--out_dir", type=str, default = '/mnt/data2/Justice/OR_learning/output/cnn_spatialAA_binary/', 
                        help=f"Absolute Path to output directory")
    args = parser.parse_args()

    
    if args.param_dir: 
        import json 
        param_dir = str(args.param_dir)
        param_list = []
        for _param in os.listdir(param_dir): 
            with open(os.path.join(param_dir, _param), 'r') as file:
                data = json.load(file)
            param_list.append(data)
            
        num_trials = len(param_list)
        param_list = [_param.values() for _param in param_list] # convert to just list form 

    else: 
        param_list = list(product(*param_choices.values()))
        random.shuffle(param_list)
        num_trials = args.num_trials  # number of random configs to test

        
    sampled_params = param_list[:num_trials]
    param_keys = list(param_choices.keys())

    voxel_shape = voxel_tensor.shape  # [N, D, H, W, C]
    C, D, H, W = voxel_shape[-1], voxel_shape[1], voxel_shape[2], voxel_shape[3]

    # Saving directory
    save_path = str(args.out_dir)
    if not os.path.exists(save_path): 
        os.mkdir(save_path) 
    for i, param_values in enumerate(sampled_params):
        param_dict = dict(zip(param_keys, param_values))
        lr = param_dict['lr']
        batch_size = param_dict['batch_size']
        hidden_dim = param_dict['hidden_dim']
        kernel_size = param_dict['kernel_size']

        run_id = make_run_id(param_dict)
        if not args.param_dir: 
            if any(run_id in dirname for dirname in os.listdir(save_path)):
                print(f"Skipping existing run with {run_id}")
                continue

        run_name = os.path.join(save_path, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{run_id}")
        os.makedirs(run_name, exist_ok=True)
        with open(os.path.join(run_name, "params.json"), "w") as f:
            json.dump(param_dict, f, indent=4)

        model = ORLigandCNN(
            voxel_shape=(C, D, H, W),
            ligand_dim=ligand_fp_tensor.shape[1],
            kernel_size=kernel_size,
            hidden_dim=hidden_dim,
            num_conv_layers=param_dict['num_conv_layers'],
            conv_channels=tuple([16 * 2**i for i in range(param_dict['num_conv_layers'])]),
            dropout=param_dict['dropout'],
            pool_type=param_dict['pooling']
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.BCEWithLogitsLoss()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        train_dataset, val_dataset, test_dataset = create_balanced_datasets(
            ps6_df, voxel_tensor, ligand_fp_tensor, or_index_map, ligand_index_map,
            total_samples=1000, positive_weight=0.5, seed=i
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        initial_epochs = args.min_epoch
        max_epochs = args.max_epoch
        patience = args.patience
        min_delta = 1e-4
        best_val_loss = float('inf')
        best_model_path = os.path.join(run_name, "best_model.pt")
        train_losses, val_losses = [], []
        val_aucs = []
        epochs_no_improve = 0
        epoch = 0

        print(f"\nTraining {run_name}")
        while epoch < max_epochs:
            epoch += 1
            model.train()
            total_train_loss = 0
            train_probs, train_labels = [], []

            for vox, lig, y in train_loader:
                vox, lig, y = vox.to(device), lig.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(vox, lig)
                loss = loss_fn(logits, y)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
                train_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
                train_labels.append(y.cpu().numpy())

            model.eval()
            total_val_loss = 0
            val_probs, val_labels = [], []

            with torch.no_grad():
                for vox, lig, y in val_loader:
                    vox, lig, y = vox.to(device), lig.to(device), y.to(device)
                    logits = model(vox, lig)
                    loss = loss_fn(logits, y)
                    total_val_loss += loss.item()
                    val_probs.append(torch.sigmoid(logits).cpu().numpy())
                    val_labels.append(y.cpu().numpy())

            avg_train_loss = total_train_loss / len(train_loader)
            avg_val_loss = total_val_loss / len(val_loader)
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            # Metrics
            val_probs_flat = np.concatenate(val_probs).flatten()
            val_labels_flat = np.concatenate(val_labels).flatten()
            val_preds_binary = (val_probs_flat > 0.5).astype(np.float32)
            val_acc = accuracy_score(val_labels_flat, val_preds_binary)
            try:
                val_auc = roc_auc_score(val_labels_flat, val_probs_flat)
            except ValueError:
                val_auc = float("nan")  # Only one class present
            val_aucs.append(val_auc)

            print(f"Epoch {epoch:2d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Acc: {val_acc:.4f} | AUC: {val_auc:.4f}")

            # Save best model
            if avg_val_loss + min_delta < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), best_model_path)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epoch >= initial_epochs and epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

        # Plot training curves
        plt.figure()
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Val Loss")
        plt.plot(val_aucs, label="Val AUC")
        plt.xlabel("Epoch")
        plt.legend()
        plt.title("Loss and AUC")
        plt.savefig(os.path.join(run_name, "loss_auc_curve.png"))
        plt.close()

        # Save arrays
        np.save(os.path.join(run_name, "train_losses.npy"), np.array(train_losses))
        np.save(os.path.join(run_name, "val_losses.npy"), np.array(val_losses))
        np.save(os.path.join(run_name, "val_aucs.npy"), np.array(val_aucs))

        # Save final validation predictions
        np.save(os.path.join(run_name, "val_probs.npy"), val_probs_flat)
        np.save(os.path.join(run_name, "val_labels.npy"), val_labels_flat)


        from sklearn.metrics import confusion_matrix, roc_curve
        import seaborn as sns

        # Confusion matrix
        conf = confusion_matrix(val_labels_flat, val_preds_binary)
        plt.figure()
        sns.heatmap(conf, annot=True, fmt="d", cmap="Blues")
        plt.title("Validation Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(os.path.join(run_name, "confusion_matrix.png"))
        plt.close()

        # ROC curve
        fpr, tpr, _ = roc_curve(val_labels_flat, val_probs_flat)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {val_auc:.2f}")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("Validation ROC Curve")
        plt.legend()
        plt.savefig(os.path.join(run_name, "roc_curve.png"))
        plt.close()
        
if __name__ == "__main__":
    main()