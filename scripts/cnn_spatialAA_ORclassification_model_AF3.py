import os 
import sys
import pandas as pd 
import numpy as np 


OR_LEARNING_PATH = os.path.join(os.getcwd().split('OR_learning')[0], 'OR_learning/')
sys.path.insert(0, os.path.join(OR_LEARNING_PATH, 'utils/'))

import plot_functions as pf
import color_function as cf 
import voxel_functions as vf 


Cbc_cav_coords = pd.read_pickle('/mnt/data2/Justice/OR_learning/files/binding_cavity/AF3_dict_Cbc_cav_coords2.pkl')
Cbc_res_coords = pd.read_pickle('/mnt/data2/Justice/OR_learning/files/binding_cavity/AF3_dict_Cbc_res_coords2.pkl')

# For troubleshooting, using only 100 voxels 
Cbc_cav_coords = {_key: Cbc_cav_coords[_key] for _key in list(Cbc_cav_coords)[::2]}
Cbc_res_coords = {_key: Cbc_res_coords[_key] for _key in list(Cbc_res_coords)[::2]}

voxels, grid_shape = vf.voxelize_cavity(cavity_coords = Cbc_cav_coords,
                                        residue_coords = Cbc_res_coords,
                                        resolution = 1, 
                                        encode_method='ohe', 
                                        vdw_radius = True)
print(f'Grid shape: {grid_shape}')

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



# CNN MODEL
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import re
import os
import json
import random
from itertools import product
from datetime import datetime
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns

def augment_voxel(voxel, noise_std=0.05, mask_prob=0.05):
    """
    Augmentation for voxel tensor [D, H, W, C].
    - Adds noise only to occupied voxels
    - Masks only occupied voxels
    """

    voxel_aug = voxel.clone()

    # Occupied voxels mask (where any feature > 0)
    occupied = voxel_aug.sum(dim=-1) > 0

    # --- 1. Add Gaussian noise only to occupied voxels ---
    if noise_std > 0:
        noise = torch.randn_like(voxel_aug) * noise_std
        voxel_aug[occupied] += noise[occupied]

    # --- 2. Random masking of occupied voxels ---
    if mask_prob > 0:
        mask = (torch.rand_like(voxel_aug[..., 0]) < mask_prob) & occupied
        voxel_aug[mask] = 0.0

    return voxel_aug


# ---------------- Dataset ----------------
# Predefine the family classes once
FAMILIES = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,51,52,55,56]
FAMILY_TO_IDX = {fam: i for i, fam in enumerate(FAMILIES)}
IDX_TO_FAMILY = {i: fam for i, fam in enumerate(FAMILIES)}
NUM_CLASSES = len(FAMILIES)

class ORVoxelDataset(Dataset):
    def __init__(self, voxel_tensor, or_index_map, augment=False):
        """
        voxel_tensor: [N_or, D, H, W, C]
        or_index_map: dict mapping OR name -> index in voxel_tensor
        """
        self.voxel_tensor = voxel_tensor
        self.or_names = list(or_index_map.keys())
        self.or_indices = list(or_index_map.values())
        self.augment = augment

        # Map OR name → family index using predefined FAMILY_TO_IDX
        OR_family = []
        for or_name in self.or_names:
            match = re.match(r'Or(\d+)', or_name, re.IGNORECASE)
            fam = int(match.group(1)) if match else 0
            OR_family.append(fam)

        # Convert families → zero-indexed labels
        self.targets = [FAMILY_TO_IDX[f] for f in OR_family]
        self.targets = torch.tensor(self.targets, dtype=torch.long)

    def __len__(self):
        return len(self.or_indices)

    def __getitem__(self, idx):
        or_idx = self.or_indices[idx]
        voxel = self.voxel_tensor[or_idx]  # [D, H, W, C]
        target = self.targets[idx]          # class index
        if self.augment:
            voxel = augment_voxel(voxel)
        return voxel, target

# ---------------- Model ----------------
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
            layers.append(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2))
            layers.append(nn.ReLU())
            layers.append(pool(2))
            in_channels = out_channels

        self.cnn = nn.Sequential(*layers)

        # Compute flattened CNN output size
        dummy = torch.zeros(1, C, D, H, W)
        with torch.no_grad():
            out = self.cnn(dummy)
        self.cnn_out_dim = out.reshape(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(self.cnn_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)  # output logits for each class
        )

    def forward(self, voxel):
        voxel = voxel.permute(0, 4, 1, 2, 3)  # [B, D, H, W, C] → [B, C, D, H, W]
        cnn_out = self.cnn(voxel)
        cnn_out = cnn_out.reshape(cnn_out.size(0), -1)
        return self.fc(cnn_out)



# ---------------- Training Loop ----------------
def train_model(model, train_loader, val_loader, device, run_name,
                lr=1e-3, max_epochs=100, min_epoch=40, patience=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.to(device)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_losses, val_losses, val_accs = [], [], []

    best_model_path = os.path.join(run_name, "best_model.pt")

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
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0
        all_probs, all_labels = [], []
        with torch.no_grad():
            for vox, y in val_loader:
                vox, y = vox.to(device), y.to(device)
                logits = model(vox)
                
                # Compute loss
                loss = loss_fn(logits, y)
                total_val_loss += loss.item()

                probs = torch.softmax(logits, dim=1)  # get probabilities
                all_probs.append(probs.cpu().numpy())
                all_labels.append(y.cpu().numpy())

        # flatten
        val_probs = np.vstack(all_probs)          # shape: (N_samples, num_classes)
        val_labels = np.concatenate(all_labels)   # shape: (N_samples,)

        # Accuracy
        val_preds = np.argmax(val_probs, axis=1)
        val_acc = accuracy_score(val_labels, val_preds)
        val_accs.append(val_acc)
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epoch >= min_epoch or epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    # ---------------- Save Results ----------------
    np.save(os.path.join(run_name, "train_losses.npy"), np.array(train_losses))
    np.save(os.path.join(run_name, "val_losses.npy"), np.array(val_losses))
    np.save(os.path.join(run_name, "val_accs.npy"), np.array(val_accs))
    np.save(os.path.join(run_name, "val_labels.npy"), val_labels)
    np.save(os.path.join(run_name, "val_probs.npy"), val_probs)

    
    # ---------------- Metrics ----------------
    
    # ---------------- Confusion matrix ----------------
    val_label_family = [IDX_TO_FAMILY.get(i) for i in val_labels]
    val_preds_family = [IDX_TO_FAMILY.get(i) for i in val_preds]
    
    conf = confusion_matrix(val_label_family, val_preds_family, labels = FAMILIES )

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        conf,
        annot=True,
        fmt='d',
        cmap="Blues",
        xticklabels=FAMILIES,
        yticklabels=FAMILIES
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(run_name, "confusion_matrix.png"))
    plt.close()

    # Classification report
    report = classification_report(val_labels, val_preds)
    with open(os.path.join(run_name, "classification_report.txt"), "w") as f:
        f.write(report)


    # ---------------- ROC-AUC ----------------
    y_true_bin = label_binarize(val_labels, classes=range(NUM_CLASSES))  # one-hot

    # Overall macro-average ROC-AUC (multi-class)
    overall_auc = roc_auc_score(y_true_bin, val_probs, average="macro", multi_class="ovr")
    print(f"Overall ROC-AUC (macro, multi-class): {overall_auc:.4f}")

    # ---------------- Overall Accuracy ROC ----------------
    # Compute "overall correctness" as binary labels
    val_preds_class = np.argmax(val_probs, axis=1)
    y_correct = (val_preds_class == val_labels).astype(int)  # 1 if correct, 0 if incorrect
    y_score = np.max(val_probs, axis=1)  # confidence of predicted class

    fpr_overall, tpr_overall, _ = roc_curve(y_correct, y_score)
    roc_auc_overall = auc(fpr_overall, tpr_overall)

    plt.figure(figsize=(6,6))
    plt.plot(fpr_overall, tpr_overall, lw=2, label=f'Overall Accuracy ROC (AUC={roc_auc_overall:.2f})')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Overall Accuracy ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(run_name, "roc_overall_accuracy.png"))
    plt.close()

    # ---------------- Per-Family ROC Curves ----------------
    plt.figure(figsize=(6, 6))
    for i, fam in enumerate(FAMILIES):
        if np.sum(y_true_bin[:, i]) == 0:
            continue  # skip classes not present in this validation set
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], val_probs[:, i])
        roc_auc_i = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{fam} (AUC={roc_auc_i:.2f})")

    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Per-Family ROC Curves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(run_name, "roc_per_family.png"))
    plt.close()

    # ---------------- Training curves ----------------
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.plot(val_accs, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig(os.path.join(run_name, "training_curves.png"))
    plt.close()

    return best_model_path, train_losses, val_losses, val_accs


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
    int(re.match(r'Or(\d+)', _OR, re.IGNORECASE).group(1)) if re.match(r'Or(\d+)', _OR, re.IGNORECASE) else 'None'
    for _OR in or_index_map.keys()
]
family_set = sorted(set(OR_family))


# ---------- Hyperparameter Random Sampling ----------
import random
from itertools import product
import argparse
import matplotlib.pyplot as plt
import json
import os


from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
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
        'augment': 'agmnt'
    }
    def format_val(v):
        if isinstance(v, (list, tuple)):
            return "-".join(map(str, v))
        return str(v)

    parts = [f"{abbr.get(k, k)}{format_val(v)}" for k, v in param_dict.items()]
    return "_".join(parts)

# ---------------- Hyperparameter Sweep ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="./output_cnn_family")
    parser.add_argument("--num_trials", type=int, default=10)
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--min_epoch", type=int, default=40)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

    voxel_tensor = torch.stack(voxels)  # your voxels
    or_ids = list(Cbc_res_coords.keys())
    or_index_map = {or_id: i for i, or_id in enumerate(or_ids)}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    voxel_shape = voxel_tensor.shape
    C, D, H, W = voxel_shape[-1], voxel_shape[1], voxel_shape[2], voxel_shape[3]

    # ---------------- Hyperparameter Sweep ----------------
    param_choices = {
        'lr': [1e-2, 1e-3],
        'batch_size': [8, 16],
        'hidden_dim': [16, 32],
        'kernel_size': [3, 5],
        'num_conv_layers': [1, 2],
        'dropout': [0.0, 0.2],
        'pooling': ['AVG', 'MAX'],
        'augment': [True, False]
    }
    all_params = list(product(*param_choices.values()))
    random.shuffle(all_params)
    sampled_params = all_params[:args.num_trials]
    param_keys = list(param_choices.keys())

    os.makedirs(args.out_dir, exist_ok=True)

    for param_values in sampled_params:
        param_dict = dict(zip(param_keys, param_values))
        run_id = make_run_id(param_dict)
        run_name = os.path.join(
            args.out_dir,
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{run_id}"
        )
        os.makedirs(run_name, exist_ok=True)
        with open(os.path.join(run_name, "params.json"), "w") as f:
            json.dump(param_dict, f, indent=4)

        # Dataset instantiated *per run* with chosen augment setting
        dataset = ORVoxelDataset(voxel_tensor, 
                                 or_index_map, 
                                 augment=param_dict['augment'])

        labels = [dataset[i][1] for i in range(len(dataset))]
        train_indices, val_indices = train_test_split(
            range(len(dataset)),
            test_size=0.1,
            stratify=labels,
            random_state=0
        )
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)

        train_loader = DataLoader(train_dataset, 
                                  batch_size=param_dict["batch_size"], 
                                  shuffle=True)
        val_loader = DataLoader(val_dataset, 
                                batch_size=param_dict["batch_size"])

        model = ORFamilyCNN(
            voxel_shape=(C, D, H, W),
            num_classes=NUM_CLASSES,
            kernel_size=param_dict['kernel_size'],
            hidden_dim=param_dict['hidden_dim'],
            num_conv_layers=param_dict['num_conv_layers'],
            conv_channels=tuple([16 * 2**i for i in range(param_dict['num_conv_layers'])]),
            dropout=param_dict['dropout'],
            pool_type=param_dict['pooling']
        )

        train_model(
            model, train_loader, val_loader, device, run_name,
            lr=param_dict['lr'],
            max_epochs=args.max_epoch,
            min_epoch=args.min_epoch,
            patience=args.patience
        )
        
if __name__ == "__main__":
    main()