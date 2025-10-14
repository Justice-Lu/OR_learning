import os 
import sys
import pandas as pd 
import numpy as np 


OR_LEARNING_PATH = os.path.join(os.getcwd().split('OR_learning')[0], 'OR_learning/')
sys.path.insert(0, os.path.join(OR_LEARNING_PATH, 'utils/'))

import plot_functions as pf
import color_function as cf 
import voxel_functions as vf 


# Load in ESM embeddings
res_esm = np.load('/mnt/data2/Justice/OR_learning/files/ESM/residue_embeddings.npy', 
                  allow_pickle=True)
OR_label = np.load('/mnt/data2/Justice/OR_learning/files/ESM/esm_OR_order.npy', 
                 allow_pickle=True)


from sklearn.decomposition import TruncatedSVD

# esm_embeddings: list of arrays (shape = [L, 1280])
all_residues = np.vstack(res_esm)  # shape = [total_residues, 1280]

svd = TruncatedSVD(n_components=32, random_state=42)
res_esm_reduced = svd.fit_transform(all_residues)

print(f'SVD explained variance: {np.sum(svd.explained_variance_ratio_)}')

# Track the number of residues (length) in each OR
lengths = [arr.shape[0] for arr in res_esm]
# Use np.split to break the reduced array back into original groups
res_esm_reduced = np.split(res_esm_reduced, np.cumsum(lengths)[:-1])
OR_esm = {str(_OR[0]): res_esm_reduced[i] for i, _OR in enumerate(OR_label)} # Re-Join OR label and ESM 


Cbc_cav_coords = pd.read_pickle('/mnt/data2/Justice/OR_learning/files/binding_cavity/AF3_dict_Cbc_cav_coords2.pkl')
Cbc_res_coords = pd.read_pickle('/mnt/data2/Justice/OR_learning/files/binding_cavity/AF3_dict_Cbc_res_coords2.pkl')

# For troubleshooting, using only 100 voxels 
# Cbc_cav_coords = {_key: Cbc_cav_coords[_key] for _key in list(Cbc_cav_coords)[0:100]}
# Cbc_res_coords = {_key: Cbc_res_coords[_key] for _key in list(Cbc_res_coords)[0:100]}

# Filter Cbc coordinates for ESM keys 
exclude_OR = [_key for _key in Cbc_cav_coords if _key.split('_')[0] not in OR_label]
if exclude_OR: 
    Cbc_cav_coords = {_key: _values for _key, _values in Cbc_cav_coords.items() if _key not in exclude_OR}
    Cbc_res_coords = {_key: _values for _key, _values in Cbc_res_coords.items() if _key not in exclude_OR}

voxels, grid_shape = vf.voxelize_cavity(cavity_coords = Cbc_cav_coords,
                                      residue_coords = Cbc_res_coords,
                                      resolution = 1, 
                                      encode_method='esm', 
                                      OR_esm = OR_esm, 
                                      esm_order = list(Cbc_cav_coords), 
                                      vdw_radius = True, 
                                      sparse_mode = False)
print(f'Grid shape: {grid_shape}')

del OR_esm, res_esm, OR_label, res_esm_reduced
del Cbc_cav_coords


"""
Prepare cid features 
"""

# Load pS6-IP data 
ps6_df = pd.read_csv('/mnt/data2/Justice/OR_learning/files/pS6IP/pS6IP_MASTER_HL_Annotated_2025.csv', index_col = 0) 
# Subset for concentration in percentages 
ps6_df = ps6_df[ps6_df.concentration.str.contains('p')]
ps6_df = ps6_df.sort_values(['Family', 'DL_OR', 'odor_category', 'odor', 'concentration', 'FDR', 'activation_zscore']).dropna()
# Subset for ORs in voxel ORs.  
unique_ORs = np.unique([_keys.split('_')[0] for _keys in list(Cbc_res_coords)])
ps6_df = ps6_df[ps6_df.DL_OR.isin(unique_ORs)]



# import pubchempy as pcp
# from tqdm import tqdm
# from rdkit import Chem
# from rdkit.Chem import AllChem
# from sklearn.decomposition import PCA
# import torch 

# def cid_to_smiles(cid):
#     try:
#         compound = pcp.Compound.from_cid(cid)
#         return compound.isomeric_smiles
#     except:
#         return None
    
# def smiles_to_ecfp4(smiles, n_bits=1028):
#     mol = Chem.MolFromSmiles(smiles)
#     if mol:
#         return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits))
#     else:
#         return None

# # Simply load ligand tensor if previously created. 
# if not os.path.exists('/mnt/data2/Justice/OR_learning/output/ligand_fp_tensor.pt'): 
#     # Generate smiles from pS6-IP cid 
#     unique_cids = ps6_df.cid.unique()
#     cid_smiles_dict = {cid: cid_to_smiles(str(cid)) for cid in tqdm(unique_cids)}

#     # Create fingerprint from smiles 
#     ligand_fp_dict = {}
#     for cid, smiles in tqdm(cid_smiles_dict.items()):
#         if smiles:
#             fp = smiles_to_ecfp4(smiles)
#             if fp is not None:
#                 ligand_fp_dict[cid] = fp

#     ligand_fp_df = pd.DataFrame.from_dict(ligand_fp_dict, orient='index')
#     ligand_fp_df.index.name = 'cid'
#     pca = PCA(n_components=32)

#     ligand_fp_tensor = torch.tensor(pca.fit_transform(ligand_fp_df.values), dtype=torch.float32)

#     print(f'PCA explained variance: {np.sum(pca.explained_variance_ratio_)}')
# else: 
#     cid_smiles_dict = np.load('/mnt/data2/Justice/OR_learning/output/cid_smiles_dict.npy', allow_pickle=True).item()
#     ligand_fp_tensor = torch.load('/mnt/data2/Justice/OR_learning/output/ligand_fp_tensor.pt')



# CNN MODEL
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from datetime import datetime

import torch
import random
import scipy.ndimage as ndimage

def augment_voxel(voxel, rotation_prob=0.5, noise_std=0.01, mask_prob=0.01):
    """
    Apply augmentation to voxel tensor [D, H, W, C].
    Returns augmented voxel of same shape.
    """

    # --- 1. Random rotation around z-axis (membrane axis) ---
    if random.random() < rotation_prob:
        angle = random.choice([0, 90, 180, 270])  # 90° increments
        voxel = torch.tensor(
            np.rot90(voxel.numpy(), k=angle // 90, axes=(0, 1)).copy()
        )

    # --- 2. Add Gaussian noise to features ---
    if noise_std > 0:
        noise = torch.randn_like(voxel) * noise_std
        voxel = voxel + noise

    # --- 3. Random masking ---
    if mask_prob > 0:
        mask = torch.rand_like(voxel[..., 0]) < mask_prob
        voxel[mask] = 0.0

    return voxel


# ---------- Set Up Data ----------
from torch.utils.data import Dataset
import torch
import re 

class ORVoxelDataset(Dataset):
    def __init__(self, voxel_tensor, or_index_map, augment=False):
        """
        voxel_tensor: [N_or, D, H, W, C]
        or_index_map: dict mapping OR name -> index in voxel_tensor
        family_to_idx: dict mapping OR family -> class index
        """
        self.voxel_tensor = voxel_tensor
        self.or_names = list(or_index_map.keys())
        self.or_indices = list(or_index_map.values())
        self.targets = []
        self.augment = augment

        for or_name in self.or_names:
            match = re.match(r'Or(\d+)', or_name, re.IGNORECASE)
            family = match.group(1) if match else "None"
            self.targets.append(family)

        self.targets = torch.tensor(self.targets, dtype=torch.long)

    def __len__(self):
        return len(self.or_indices)

    def __getitem__(self, idx):
        or_idx = self.or_indices[idx]
        voxel = self.voxel_tensor[or_idx]      # [D, H, W, C]
        target = self.targets[idx]             # int class label

        if self.augment:
            voxel = augment_voxel(voxel)

        return voxel, target

    
import torch.nn as nn
import torch.nn.functional as F

# ---------- Model ----------
class ORFamilyCNN(nn.Module):
    def __init__(self, voxel_shape=(41, 42, 65), num_classes=18,
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
            layers.append(nn.Conv3d(in_channels, out_channels,
                                    kernel_size=kernel_size,
                                    padding=kernel_size // 2))
            layers.append(nn.ReLU())
            layers.append(pool(2))
            in_channels = out_channels

        self.cnn = nn.Sequential(*layers)

        # Get flattened CNN output size
        dummy = torch.zeros(1, C, D, H, W)
        with torch.no_grad():
            out = self.cnn(dummy)
        self.cnn_out_dim = out.reshape(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(self.cnn_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)  # multi-class logits
        )

    def forward(self, voxel):
        voxel = voxel.permute(0, 4, 1, 2, 3)  # [B, D, H, W, C] → [B, C, D, H, W]
        cnn_out = self.cnn(voxel)
        cnn_out = cnn_out.reshape(cnn_out.size(0), -1)
        return self.fc(cnn_out)

# ---------- Prepare Data ----------
# voxel_tensor: [N_or, D, H, W, C]
# ligand_fp_tensor: [N_ligand, F]
# ps6_df: full OR-ligand response table with DL_OR, cid, activation_zscore
# Cbc_res_coords, cid_smiles_dict: OR and ligand reference maps

or_ids = list(Cbc_res_coords.keys())
# ligand_cids = list(cid_smiles_dict.keys())
or_index_map = {or_id: i for i, or_id in enumerate(or_ids)}
# ligand_index_map = {cid: i for i, cid in enumerate(ligand_cids)}
voxel_tensor = torch.stack(voxels)

OR_family = [
    re.match(r'Or(\d+)', _OR).group(1) if re.match(r'Or(\d+)', _OR) else 'None'
    for _OR in or_index_map.keys()
]
family_set = sorted(set(OR_family))
num_classes = len(family_set)


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
        # 'hidden_dim': [128, 256, 512, 1024],
        'hidden_dim': [8, 16, 32, 64],
        'kernel_size': [1, 3, 5, 7],
        'num_conv_layers': [1, 2, 3],
        'dropout': [0.0, 0.2, 0.3, 0.5], 
        # 'total_sample': [2000, 1000], 
        # 'pos_ratio': [0.5, 0.3]
    }
    
        # Compute all combinations
    all_params = list(product(*param_choices.values()))
    total_combinations = len(all_params)

    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_trials", type=int, default=total_combinations)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min_epoch", type=int, default=40)
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--param_dir", default=None)
    parser.add_argument("--out_dir", type=str, 
        default='/mnt/data2/Justice/OR_learning/output/cnn_family/')
    for key in param_choices.keys():
        parser.add_argument(f"--{key}", type=str, nargs="+")
    args = parser.parse_args()

    # Load param_list
    if args.param_dir: 
        param_list = []
        for _param in os.listdir(args.param_dir): 
            with open(os.path.join(args.param_dir, _param), 'r') as file:
                data = json.load(file)
            param_list.append(list(data.values()))
    else: 
        param_list = list(product(*param_choices.values()))
        random.shuffle(param_list)
    
    import ast
    def parse_override(val):
        try:
            # Try to parse Python literal (list, number, etc.)
            parsed = ast.literal_eval(val)
            if isinstance(parsed, (list, tuple)):
                return list(parsed)
            else:
                return [parsed]
        except (SyntaxError, ValueError):
            # Fallback: just return as string in list
            return [val]
        
    # Apply overrides from CLI
    overrides = {}
    for k in param_choices.keys():
        raw_val = getattr(args, k)
        if raw_val is not None:
            parsed_values = []
            for v in raw_val:
                parsed_values.extend(parse_override(v))
            parsed_values = [type(param_choices[k][0])(v) for v in parsed_values]
            overrides[k] = parsed_values

            # Update param_choices
            param_choices[k] = parsed_values

            # Update param_list configs
            key_idx = list(param_choices.keys()).index(k)
            if len(parsed_values) == 1:
                param_list = [
                    tuple(val if i != key_idx else parsed_values[0] for i, val in enumerate(cfg))
                    for cfg in param_list
                ]
            else:
                param_list = list(product(*param_choices.values()))

    # Finalize trial count
    if args.num_trials is None:
        num_trials = len(param_list)
    else:
        num_trials = args.num_trials

    print(f"Total configs to test: {num_trials}")
    sampled_params = param_list[:num_trials]
    param_keys = list(param_choices.keys())

    # ---------------- Save Path ----------------
    save_path = args.out_dir
    os.makedirs(save_path, exist_ok=True)

    param_keys = list(param_choices.keys())
    sampled_params = param_list[:args.num_trials]

    # ---------------- Loop over configs ----------------
    voxel_shape = voxel_tensor.shape
    C, D, H, W = voxel_shape[-1], voxel_shape[1], voxel_shape[2], voxel_shape[3]
        
    # ---------------- Dataset ----------------
    dataset = ORVoxelDataset(voxel_tensor, or_index_map, augment=True)
    # num_classes = len(family_to_idx)
    
    # ---------------- Train/Val Split ----------------
    from torch.utils.data import random_split
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
    for i, param_values in enumerate(sampled_params):
        param_dict = dict(zip(param_keys, param_values))
        lr = param_dict['lr']
        batch_size = param_dict['batch_size']

        # Check if this hyperparameter combination already exists
        run_id = make_run_id(param_dict)
        if not args.param_dir: 
            if any(run_id in dirname for dirname in os.listdir(save_path)):
                print(f"Skipping existing run with {run_id}")
                continue
        
        # Otherwise, proceed
        run_name = os.path.join(save_path, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{run_id}")
        os.makedirs(run_name, exist_ok=True)
        print(param_dict)
        with open(os.path.join(run_name, "params.json"), "w") as f:
            json.dump(param_dict, f, indent=4)


        # ---------------- Model ----------------
        model = ORFamilyCNN(
            voxel_shape=(C, D, H, W),
            # num_classes=num_classes,
            kernel_size=param_dict['kernel_size'],
            hidden_dim=param_dict['hidden_dim'],
            num_conv_layers=param_dict['num_conv_layers'],
            conv_channels=tuple([16 * 2**i for i in range(param_dict['num_conv_layers'])]),
            dropout=param_dict['dropout'], 
            pool_type=param_dict['pooling']
        )   

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        
        # ---------------- Training Loop ----------------
        min_delta = 1e-4
        patience = args.patience
        initial_epochs = args.min_epoch
        max_epochs = args.max_epoch

        best_val_loss = float('inf')
        epochs_no_improve = 0
        train_losses, val_losses, val_accs = [], [], []

        best_model_path = os.path.join(run_name, "best_model.pt")

        print(f"\nTraining {run_name}")
        for epoch in range(1, max_epochs+1):
            model.train()
            total_train_loss = 0
            for vox, y in train_loader:
                vox, y = vox.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(vox)
                loss = loss_fn(logits, y)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            model.eval()
            total_val_loss, correct, total = 0, 0, 0
            with torch.no_grad():
                for vox, y in val_loader:
                    vox, y = vox.to(device), y.to(device)
                    logits = model(vox)
                    loss = loss_fn(logits, y)
                    total_val_loss += loss.item()
                    preds = torch.argmax(logits, dim=1)
                    correct += (preds == y).sum().item()
                    total += y.size(0)

            avg_train_loss = total_train_loss / len(train_loader)
            avg_val_loss = total_val_loss / len(val_loader)
            val_acc = correct / total
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            val_accs.append(val_acc)

            print(f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

            # Early stopping
            if avg_val_loss + min_delta < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), best_model_path)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epoch >= initial_epochs and epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break


        # ---------------- Save Results ----------------
        np.save(os.path.join(run_name, "train_losses.npy"), np.array(train_losses))
        np.save(os.path.join(run_name, "val_losses.npy"), np.array(val_losses))
        np.save(os.path.join(run_name, "val_accs.npy"), np.array(val_accs))

        plt.figure()
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Val Loss")
        plt.plot(val_accs, label="Val Acc")
        plt.legend()
        plt.title("Training Curves")
        plt.savefig(os.path.join(run_name, "training_curves.png"))
        plt.close()

        from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
        import seaborn as sns

        # ---------------- Evaluate on Validation Set ----------------
        model.eval()
        all_logits, all_labels = [], []

        with torch.no_grad():
            for vox, y in val_loader:
                vox, y = vox.to(device), y.to(device)
                logits = model(vox)
                all_logits.append(logits.cpu())
                all_labels.append(y.cpu())

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # ---------------- Predicted Classes ----------------
        val_preds = torch.argmax(all_logits, dim=1)
        val_labels = all_labels

        # ---------------- Confusion Matrix ----------------
        conf = confusion_matrix(val_labels, val_preds)
        plt.figure(figsize=(10,8))
        sns.heatmap(conf, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Validation Confusion Matrix")
        plt.savefig(os.path.join(run_name, "confusion_matrix.png"))
        plt.close()

        # ---------------- Classification Report ----------------
        report = classification_report(val_labels, val_preds, target_names=list(family_set))
        with open(os.path.join(run_name, "classification_report.txt"), "w") as f:
            f.write(report)

        # ---------------- ROC Curves ----------------
        # Convert labels to one-hot for multiclass ROC
        val_labels_onehot = F.one_hot(val_labels, num_classes=(num_classes)).numpy()
        val_probs = F.softmax(all_logits, dim=1).numpy()

        plt.figure(figsize=(10,8))
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(val_labels_onehot[:, i], val_probs[:, i])
            auc = roc_auc_score(val_labels_onehot[:, i], val_probs[:, i])
            plt.plot(fpr, tpr, label=f"{family_set[i]} (AUC={auc:.2f})")

        plt.plot([0,1],[0,1], 'k--')
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC Curves (One-vs-Rest)")
        plt.legend(loc="lower right", fontsize=8)
        plt.savefig(os.path.join(run_name, "roc_curves.png"))
        plt.close()
        
if __name__ == "__main__":
    main()