import os 
import sys
import pandas as pd 
import numpy as np 
import random

random.seed(0)

OR_LEARNING_PATH = os.path.join(os.getcwd().split('OR_learning')[0], 'OR_learning/')
sys.path.insert(0, os.path.join(OR_LEARNING_PATH, 'utils/'))

import plot_functions as pf
import color_function as cf 
import voxel_functions as vf 


# Load pS6-IP data 
ps6_df = pd.read_csv('/mnt/data2/Justice/OR_learning/files/pS6IP/pS6IP_MASTER_HL_Annotated_2025.csv', index_col = 0) 
ps6_df.odor_category = ps6_df.odor_category.astype(pd.CategoricalDtype(categories=['Alcohols', 'Aldehydes', 'Ketones','CarboxylicAcids', 'Esters',
                                                                                   'Tiglates', 'Pyradine', 'Thiazole', 'Sulfurous', 'Others'], ordered=True))
# Subset for concentration in percentages 
ps6_df = ps6_df[ps6_df.concentration.str.contains('p')]
ps6_df = ps6_df.sort_values(['Family', 'DL_OR', 'odor_category', 'odor', 'concentration', 'FDR', 'activation_zscore']).dropna()

# IMPORTANT DROPPING OUT ODOR CATEGORY OF 'OTHERS'
ps6_df = ps6_df[ps6_df.odor_category != 'Others']

# Build odor category vocab
ODOR_CATEGORIES = ps6_df.odor_category.sort_values().unique()
ODORS = [_odor for _cat in ODOR_CATEGORIES for _odor in ps6_df[ps6_df.odor_category == _cat].dropna().sort_values(['cid_mw', 'odor']).odor.unique()]
ODOR_TO_CAT = dict(zip(ps6_df.odor, ps6_df.odor_category))
CATEGORY_COLORS = cf.distinct_colors(ODOR_CATEGORIES, category = 'tab10')
ODOR_COLORS = {}
for _category in ODOR_CATEGORIES:     
    ODOR_COLORS.update(cf.generate_faded_shades(CATEGORY_COLORS[_category],
                                     list(ps6_df[ps6_df.odor_category == _category].sort_values(['cid_mw', 'odor']).odor.unique()), 
                                     (0.4,0.8)))
    
ODOR_TO_IDX = {cat: i for i, cat in enumerate(ODORS)}
IDX_TO_ODOR = {i: cat for i, cat in enumerate(ODORS)}
NUM_ODORS = len(ODORS)

# Subset for ORs in voxel ORs.  
# unique_ORs = np.unique([_keys.split('_')[0] for _keys in list(Cbc_res_coords)])
# ps6_df = ps6_df[ps6_df.DL_OR.isin(unique_ORs)]

activated_ps6_df = ps6_df[ps6_df.activation_zscore >= 2]
zero_ps6_df = ps6_df[((ps6_df.activation_zscore <= 0.2) | 
                     (ps6_df.activation_zscore >= -0.2)) & 
                     (~ps6_df.DL_OR.isin(list(activated_ps6_df.DL_OR.unique())))]


Cbc_cav_coords = pd.read_pickle('/mnt/data2/Justice/OR_learning/files/binding_cavity/AF3_dict_Cbc_cav_coords2.pkl')
Cbc_res_coords = pd.read_pickle('/mnt/data2/Justice/OR_learning/files/binding_cavity/AF3_dict_Cbc_res_coords2.pkl')


# For troubleshooting, using only positive activation_zscore 
# Take x number of none activating ORs proportion to activated ORs. 
# sample_ORs = random.sample(list(zero_ps6_df.DL_OR.unique()), 
#                            len(list(activated_ps6_df.DL_OR.unique())) // 6) + list(activated_ps6_df.DL_OR.unique())

sample_ORs = list(activated_ps6_df.DL_OR.unique()) # No non-activated ORs
sample_ORs = [_ORs.lower() for _ORs in sample_ORs]

Cbc_cav_coords = {_key: Cbc_cav_coords[_key] for _key in list(Cbc_cav_coords) if _key.split('_')[0].lower() in sample_ORs}
Cbc_res_coords = {_key: Cbc_res_coords[_key] for _key in list(Cbc_res_coords) if _key.split('_')[0].lower() in sample_ORs}
# For troubleshooting, using only 100 voxels 
# Cbc_cav_coords = {_key: Cbc_cav_coords[_key] for _key in list(Cbc_cav_coords)[:100]}
# Cbc_res_coords = {_key: Cbc_res_coords[_key] for _key in list(Cbc_res_coords)[:100]}

# Taking only x prediction of each AF3
# Cbc_cav_coords = {_key: Cbc_cav_coords[_key] for _key in list(Cbc_cav_coords) if _key.split('_')[1] in (['0','1','2','3'])}
# Cbc_res_coords = {_key: Cbc_res_coords[_key] for _key in list(Cbc_res_coords) if _key.split('_')[1] in (['0','1','2','3'])}
print('Cbc coords files loaded. . .')


########### Compute and load ESM embeddings
# Load in ESM embeddings
# print("loading ESM")
# res_esm = np.load('/mnt/data2/Justice/OR_learning/files/ESM/residue_embeddings.npy', 
#                   allow_pickle=True)
# OR_label = np.load('/mnt/data2/Justice/OR_learning/files/ESM/esm_OR_order.npy', 
#                  allow_pickle=True)

# from sklearn.decomposition import TruncatedSVD
# # esm_embeddings: list of arrays (shape = [L, 1280])
# all_residues = np.vstack(res_esm)  # shape = [total_residues, 1280]
# svd = TruncatedSVD(n_components=32, random_state=42)
# res_esm_reduced = svd.fit_transform(all_residues)
# print(f'SVD explained variance: {np.sum(svd.explained_variance_ratio_)}')

# # Track the number of residues (length) in each OR
# lengths = [arr.shape[0] for arr in res_esm]
# # Use np.split to break the reduced array back into original groups
# res_esm_reduced = np.split(res_esm_reduced, np.cumsum(lengths)[:-1])
# OR_esm = {str(_OR[0]): res_esm_reduced[i] for i, _OR in enumerate(OR_label)} # Re-Join OR label and ESM 

OR_esm = np.load('/mnt/data2/Justice/OR_learning/files/ESM/combined/reduced_OR_esm_mmhs.npy', allow_pickle= True).item()
OR_esm =  {_key: _values for _key, _values in OR_esm.items() if not _key.isupper()} # Exclude human OR 

# Filter Cbc coordinates for ESM keys 
exclude_OR = []
exclude_OR.extend([_k for _k in Cbc_res_coords.keys() if _k.isupper()]) # Exclude human OR 
exclude_OR.extend([_key for _key in Cbc_cav_coords if _key.split('_')[0] not in OR_esm.keys()]) # Exclude none overlapping OR names in ESM vs coords
if exclude_OR: 
    Cbc_cav_coords = {_key: _values for _key, _values in Cbc_cav_coords.items() if _key not in exclude_OR}
    Cbc_res_coords = {_key: _values for _key, _values in Cbc_res_coords.items() if _key not in exclude_OR}

center_reference = np.load('/mnt/data2/Justice/OR_learning/files/binding_cavity/AF3_center_reference.npy')
voxels, grid_shape = vf.voxelize_cavity(cavity_coords = Cbc_cav_coords,
                                      residue_coords = Cbc_res_coords,
                                      resolution = 1, 
                                      encode_method='esm', 
                                      OR_esm = OR_esm, 
                                      esm_order = list(Cbc_cav_coords), 
                                      cube_dim = 32, reference_center = center_reference, 
                                      vdw_radius = True, 
                                      sparse_mode = False)
print(f'Grid shape: {grid_shape}')

del OR_esm
del Cbc_cav_coords


# CNN MODEL
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import re
import os
import json
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

    # --- Add Gaussian noise only to occupied voxels ---
    if noise_std > 0:
        noise = torch.randn_like(voxel_aug) * noise_std
        voxel_aug[occupied] += noise[occupied]
    # --- Random masking of occupied voxels ---
    if mask_prob > 0:
        mask = (torch.rand_like(voxel_aug[..., 0]) < mask_prob) & occupied
        voxel_aug[mask] = 0.0

    return voxel_aug

def inflate_voxels(voxel_tensor, or_index_map, ps6_df,
                   n_augments=5, noise_std=0.05, mask_prob=0.05):
    """
    Inflate dataset by creating augmented copies of each OR voxel.
    Returns new_voxel_tensor, new_index_map.
    """
    new_voxels = []
    new_names = []

    for or_name, idx in or_index_map.items():
        voxel = voxel_tensor[idx]
        # Keep original
        new_voxels.append(voxel)
        new_names.append(or_name)

        # Generate augmented versions
        for _ in range(n_augments):
            aug_voxel = augment_voxel(voxel, noise_std=noise_std, mask_prob=mask_prob)
            new_voxels.append(aug_voxel)
            new_names.append(or_name)

    new_voxels = torch.stack(new_voxels)
    new_index_map = {name: i for i, name in enumerate(new_names)}
    return new_voxels, new_index_map


# ---------------- Dataset ----------------

class ORVoxelDataset(Dataset):
    def __init__(self, voxel_tensor, or_index_map, ps6_df, score_threshold=2.0):
        """
        voxel_tensor: [N_or, D, H, W, C]
        or_index_map: dict mapping OR name -> index in voxel_tensor
        ps6_df: dataframe containing DL_OR, odor_category, activation_zscore
        score_threshold: only include categories with zscore >= threshold
        """
        self.voxel_tensor = voxel_tensor
        self.or_names = list(or_index_map.keys())
        self.or_indices = list(or_index_map.values())

        self.targets = []

        for or_name in self.or_names:
            clean_or_name = or_name.split('_')[0].lower()
            df_sub = ps6_df[(ps6_df.DL_OR.str.lower() == clean_or_name) &
                            (ps6_df.activation_zscore >= score_threshold)]
            
            # print(clean_or_name, ps6_df.DL_OR.str.lower().unique())

            # Build binary target vector: 1 if activation_zscore >= threshold else 0
            target_vec = np.zeros(NUM_ODORS, dtype=np.float32)
            for _, row in df_sub.iterrows():
                idx = ODOR_TO_IDX[row.odor]
                target_vec[idx] = 1.0 

            self.targets.append(target_vec)

        self.targets = torch.tensor(np.array(self.targets), dtype=torch.float32)

    def __len__(self):
        return len(self.or_names)

    def __getitem__(self, idx):
        voxel = self.voxel_tensor[self.or_indices[idx]]
        target = self.targets[idx]

        voxel_names = self.or_names[idx]

        return voxel, target, voxel_names
    
    
# ---------------- Model ----------------
class OROdorCNN(nn.Module):
    def __init__(self, voxel_shape=(41, 42, 65), num_classes=None,
                 kernel_size=3, hidden_dim=256,
                 num_conv_layers=2, conv_channels=(16, 32),
                 dropout=0.0, pool_type='AVG'):
        super().__init__()
        C, D, H, W = voxel_shape
        assert num_classes is not None, "num_classes must be set to number of odor categories"

        layers = []
        in_channels = C
        pool = nn.AvgPool3d if pool_type == 'AVG' else nn.MaxPool3d

        for i in range(num_conv_layers):
            out_channels = conv_channels[i]
            layers.append(
                nn.Conv3d(in_channels, out_channels,
                          kernel_size=kernel_size,
                          padding=kernel_size // 2)
            )
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
            nn.Linear(hidden_dim, num_classes)  # multi-label logits
        )

    def forward(self, voxel):
        # voxel: [B, D, H, W, C] → reorder for Conv3d
        voxel = voxel.permute(0, 4, 1, 2, 3)  # [B, C, D, H, W]
        cnn_out = self.cnn(voxel)
        cnn_out = cnn_out.reshape(cnn_out.size(0), -1)
        return self.fc(cnn_out)  # logits [B, num_classes]


# ---------------- Training Loop ----------------
def train_model(model, train_loader, val_loader, device, run_name,
                lr=1e-3, max_epochs=100, min_epoch=40, patience=10,
                multi_label=True, threshold=0.4):
    """
    Train model for single-label (multi-class) or multi-label classification,
    while keeping all original validation plots.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss() if multi_label else nn.CrossEntropyLoss()
    model.to(device)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_losses, val_losses, val_accs = [], [], []

    best_model_path = os.path.join(run_name, "best_model.pt")

    for epoch in range(1, max_epochs+1):
        # -------- Training --------
        model.train()
        total_train_loss = 0
        train_labels = []
        train_voxel_names = []
        for vox, y, voxel_name in train_loader:
            vox, y = vox.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(vox)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            
            train_labels.append(y.cpu().numpy())
            train_voxel_names.extend(voxel_name)
            
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        train_labels = np.vstack(train_labels) if multi_label else np.concatenate(train_labels)

        # -------- Validation --------
        model.eval()
        total_val_loss = 0
        val_probs, val_labels = [], []
        val_voxel_names = []
        with torch.no_grad():
            for vox, y, voxel_name in val_loader:
                vox, y = vox.to(device), y.to(device)
                logits = model(vox)
                loss = loss_fn(logits, y)
                total_val_loss += loss.item()

                probs = torch.sigmoid(logits) if multi_label else torch.softmax(logits, dim=1)
                val_probs.append(probs.cpu().numpy())
                val_labels.append(y.cpu().numpy())
                
                val_voxel_names.extend(voxel_name)


        val_probs = np.vstack(val_probs)
        val_labels = np.vstack(val_labels) if multi_label else np.concatenate(val_labels)
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # -------- Metrics --------
        if multi_label:
            val_preds = (val_probs > threshold).astype(int)
            # val_acc = f1_score(val_labels, val_preds, average="micro", zero_division=0)
            # metric_name = "Val F1 (micro)"
            val_acc = f1_score(val_labels, val_preds, average="macro", zero_division=0)
            metric_name = "Val F1 (macro)"
            
        else:
            val_preds = np.argmax(val_probs, axis=1)
            val_acc = accuracy_score(val_labels, val_preds)
            metric_name = "Val Acc"

        val_accs.append(val_acc)
        print(f"Epoch {epoch:02d} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"{metric_name}: {val_acc:.4f}")

        # -------- Early stopping --------
        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epoch >= min_epoch and epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    # -------- Save Results --------
    np.save(os.path.join(run_name, "train_losses.npy"), np.array(train_losses))
    np.save(os.path.join(run_name, "val_losses.npy"), np.array(val_losses))
    np.save(os.path.join(run_name, "val_accs.npy"), np.array(val_accs))
    
    np.save(os.path.join(run_name, "train_labels.npy"), train_labels)
    np.save(os.path.join(run_name, "val_labels.npy"), val_labels)
    np.save(os.path.join(run_name, "val_preds.npy"), val_preds)
    np.save(os.path.join(run_name, "val_probs.npy"), val_probs)
    
    np.save(os.path.join(run_name, "train_voxel_names.npy"), train_voxel_names)
    np.save(os.path.join(run_name, "val_voxel_names.npy"), val_voxel_names)
    
    # -------- Confusion Matrix --------
    if multi_label:
        # Compute per-class metrics
        per_class_tp = np.sum((val_preds == 1) & (val_labels == 1), axis=0)
        per_class_fp = np.sum((val_preds == 1) & (val_labels == 0), axis=0)
        per_class_fn = np.sum((val_preds == 0) & (val_labels == 1), axis=0)
        per_class_tn = np.sum((val_preds == 0) & (val_labels == 0), axis=0)
        
        metrics = np.stack([per_class_tp, per_class_fp, per_class_fn, per_class_tn], axis=0)
        plt.figure(figsize=(14, 6))
        sns.heatmap(metrics, annot=True, fmt='d',
                    xticklabels=ODORS,
                    yticklabels=["TP", "FP", "FN", "TN"],
                    cmap="Blues")
        plt.title("Per-class Confusion Counts")
        plt.xlabel("Odor Category")
        plt.ylabel("Metric")
        plt.savefig(os.path.join(run_name, "confusion_matrix.png"))
        plt.close()
        
    else:
        val_label_names = [IDX_TO_ODOR.get(i) for i in val_labels]
        val_pred_names = [IDX_TO_ODOR.get(i) for i in val_preds]

        conf = confusion_matrix(val_label_names, val_pred_names, labels=ODORS)
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf, annot=True, fmt='d', cmap="Blues",
                    xticklabels=ODORS, yticklabels=ODORS)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(os.path.join(run_name, "confusion_matrix.png"))
        plt.close()
    

    # -------- Classification Report --------
    report = classification_report(val_labels, val_preds, target_names=ODORS, zero_division=0)
    with open(os.path.join(run_name, "classification_report.txt"), "w") as f:
        f.write(report)

    # -------- ROC-AUC --------
    if multi_label:
        y_true_bin = val_labels
    else:
        y_true_bin = label_binarize(val_labels, classes=range(NUM_ODORS))  # one-hot

    overall_auc = roc_auc_score(y_true_bin, val_probs, average="macro")
    print(f"Overall ROC-AUC (macro): {overall_auc:.4f}")

    # -------- Overall Accuracy ROC --------
    if multi_label:
        # Micro-average ROC
        fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), val_probs.ravel())
        roc_auc_micro = auc(fpr_micro, tpr_micro)

        # Macro-average ROC
        fpr_dict, tpr_dict, roc_auc_dict = {}, {}, {}
        for i in range(NUM_ODORS):
            if np.sum(y_true_bin[:, i]) > 0:  # skip empty classes
                fpr_dict[i], tpr_dict[i], _ = roc_curve(y_true_bin[:, i], val_probs[:, i])
                roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])

        all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in roc_auc_dict]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in roc_auc_dict:
            mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
        mean_tpr /= len(roc_auc_dict)
        roc_auc_macro = auc(all_fpr, mean_tpr)

        # --- Plot overall ROC ---
        plt.figure(figsize=(6, 6))
        # plt.plot(fpr_micro, tpr_micro, label=f"Micro-average ROC (AUC={roc_auc_micro:.2f})", lw=2)
        plt.plot(all_fpr, mean_tpr, label=f"Macro-average ROC (AUC={roc_auc_macro:.2f})", lw=2)
        plt.plot([0, 1], [0, 1], "k--", lw=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Overall ROC Curve (Micro & Macro Average)")
        plt.legend(loc="lower right", frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(run_name, "roc_overall_multilabel.png"))
        plt.close()
        
    else:
        val_preds_class = np.argmax(val_probs, axis=1)
        y_correct = (val_preds_class == val_labels).astype(int)
        y_score = np.max(val_probs, axis=1)
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

    # -------- Per-Family ROC Curves --------
    category_curves = {cat: [] for cat in ODOR_CATEGORIES}

    plt.figure(figsize=(6, 6))
    for i, _odor in enumerate(ODORS):
        if np.sum(y_true_bin[:, i]) == 0:
            continue
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], val_probs[:, i])
        roc_auc_i = auc(fpr, tpr)
        plt.plot(fpr, tpr, alpha=0.3, 
                #  label=f"{_odor} (AUC={roc_auc_i:.2f})"
                 )
        
        # Collect for category averaging
        category_curves[ODOR_TO_CAT.get(_odor)].append((fpr, tpr))

    # Plot category-averaged ROC for this run
    for cat, curves in category_curves.items():
        if not curves:
            continue
        mean_fpr = np.linspace(0, 1, 100)
        tprs_cat = []
        for fpr, tpr in curves:
            tpr_interp = np.interp(mean_fpr, fpr, tpr)
            tpr_interp[0] = 0.0
            tprs_cat.append(tpr_interp)
        mean_tpr_cat = np.mean(tprs_cat, axis=0)
        mean_tpr_cat[-1] = 1.0
        mean_auc_cat = auc(mean_fpr, mean_tpr_cat)
        plt.plot(mean_fpr, mean_tpr_cat, color=CATEGORY_COLORS[cat], lw=3,
                 label=f"{cat} Avg (AUC={mean_auc_cat:.2f})"
                 )
        
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Per-Family ROC Curves")
    plt.legend(fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(run_name, "roc_per_Odor.png"))
    plt.close()

    # -------- Training Curves --------
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.plot(val_accs, label=metric_name)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig(os.path.join(run_name, "training_curves.png"))
    plt.close()

    return best_model_path, train_losses, val_losses, val_accs


def evaluate_model(model, test_loader, device, run_name,
                   multi_label=True, threshold=0.4, model_path=None):
    """
    Evaluate a trained model on a test set. If model_path is provided, load weights from it.
    Saves test_preds, test_probs, test_labels, test_voxel_names and common metric plots/reports.
    """
    import os
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
    from sklearn.preprocessing import label_binarize
    import matplotlib.pyplot as plt
    import seaborn as sns

    if model_path is not None:
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)

    model.to(device)
    model.eval()

    loss_fn = nn.BCEWithLogitsLoss() if multi_label else nn.CrossEntropyLoss()

    test_probs, test_labels = [], []
    test_voxel_names = []
    with torch.no_grad():
        for vox, y, voxel_name in test_loader:
            vox, y = vox.to(device), y.to(device)
            logits = model(vox)
            probs = torch.sigmoid(logits) if multi_label else torch.softmax(logits, dim=1)
            test_probs.append(probs.cpu().numpy())
            test_labels.append(y.cpu().numpy())
            test_voxel_names.extend(voxel_name)

    test_probs = np.vstack(test_probs)
    test_labels = np.vstack(test_labels) if multi_label else np.concatenate(test_labels)

    if multi_label:
        test_preds = (test_probs > threshold).astype(int)
        # Ensure integer dtype
        test_labels = test_labels.astype(int)
        test_preds = test_preds.astype(int)
        # Use macro F1 for multi-label summary
        try:
            test_f1 = f1_score(test_labels, test_preds, average='macro', zero_division=0)
        except Exception:
            test_f1 = None
        print(f"Test F1 (macro): {test_f1}")
    else:
        # If labels are multi-hot vectors, convert to single class indices
        if test_labels.ndim > 1:
            try:
                test_labels = np.argmax(test_labels, axis=1)
            except Exception:
                # fallback: flatten
                test_labels = test_labels.ravel()
        test_preds = np.argmax(test_probs, axis=1)
        test_labels = test_labels.astype(int)
        test_preds = test_preds.astype(int)
        test_acc = accuracy_score(test_labels, test_preds)
        print(f"Test Accuracy: {test_acc:.4f}")

    # Save arrays
    np.save(os.path.join(run_name, "test_probs.npy"), test_probs)
    np.save(os.path.join(run_name, "test_preds.npy"), test_preds)
    np.save(os.path.join(run_name, "test_labels.npy"), test_labels)
    np.save(os.path.join(run_name, "test_voxel_names.npy"), np.array(test_voxel_names))

    # Classification report and confusion matrix (handle multi-label and single-label)
    try:
        # Save classification report for both multi-label and single-label
        report = classification_report(test_labels, test_preds, target_names=ODORS, zero_division=0)
        with open(os.path.join(run_name, "test_classification_report.txt"), "w") as f:
            f.write(report)

        if multi_label:
            # Compute per-class confusion counts (TP, FP, FN, TN)
            per_class_tp = np.sum((test_preds == 1) & (test_labels == 1), axis=0)
            per_class_fp = np.sum((test_preds == 1) & (test_labels == 0), axis=0)
            per_class_fn = np.sum((test_preds == 0) & (test_labels == 1), axis=0)
            per_class_tn = np.sum((test_preds == 0) & (test_labels == 0), axis=0)

            metrics = np.stack([per_class_tp, per_class_fp, per_class_fn, per_class_tn], axis=0)
            plt.figure(figsize=(14, 6))
            sns.heatmap(metrics, annot=True, fmt='d',
                        xticklabels=ODORS,
                        yticklabels=["TP", "FP", "FN", "TN"],
                        cmap="Blues")
            plt.title("Per-class Confusion Counts (Test)")
            plt.xlabel("Odor Category")
            plt.ylabel("Metric")
            plt.savefig(os.path.join(run_name, "test_confusion_matrix.png"))
            plt.close()
        else:
            # Single-label confusion matrix with odor names
            true_names = [IDX_TO_ODOR.get(int(i), str(i)) for i in test_labels]
            pred_names = [IDX_TO_ODOR.get(int(i), str(i)) for i in test_preds]
            conf = confusion_matrix(true_names, pred_names, labels=ODORS)
            plt.figure(figsize=(10, 8))
            sns.heatmap(conf, annot=True, fmt='d', cmap="Blues", xticklabels=ODORS, yticklabels=ODORS)
            plt.title("Test Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.tight_layout()
            plt.savefig(os.path.join(run_name, "test_confusion_matrix.png"))
            plt.close()
    except Exception as e:
        print(f"Could not create classification report/confusion matrix: {e}")

    # ROC-AUC overall
    try:
        if multi_label:
            y_true_bin = test_labels
        else:
            y_true_bin = label_binarize(test_labels, classes=range(NUM_ODORS))

        overall_auc = roc_auc_score(y_true_bin, test_probs, average="macro")
        print(f"Test ROC-AUC (macro): {overall_auc:.4f}")
        # Save overall ROC plot (macro)
        if multi_label:
            test_preds_class = np.argmax(test_probs, axis=1)
            y_correct = (test_preds_class == test_labels).astype(int)
            y_score = np.max(test_probs, axis=1)
            fpr_overall, tpr_overall, _ = roc_curve(y_correct, y_score)
            roc_auc_overall = auc(fpr_overall, tpr_overall)
            plt.figure(figsize=(6,6))
            plt.plot(fpr_overall, tpr_overall, lw=2, label=f'Overall Accuracy ROC (AUC={roc_auc_overall:.2f})')
            plt.plot([0,1],[0,1],'k--')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Test Overall Accuracy ROC")
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(os.path.join(run_name, "test_roc_overall_accuracy.png"))
            plt.close()
    except Exception as e:
        print("Could not compute ROC-AUC for test set:", e)

    return test_probs, test_preds, test_labels


# ---------- Prepare Data ----------
# voxel_tensor: [N_or, D, H, W, C]
# ligand_fp_tensor: [N_ligand, F]
# ps6_df: full OR-ligand response table with DL_OR, cid, activation_zscore
# Cbc_res_coords, cid_smiles_dict: OR and ligand reference maps

or_ids = list(Cbc_res_coords.keys())
or_index_map = {or_id: i for i, or_id in enumerate(or_ids)}
voxel_tensor = torch.stack(voxels)


# ---------- Hyperparameter Random Sampling ----------
import random
from itertools import product
import argparse
import matplotlib.pyplot as plt
import json
import os


from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
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
        'augment_noise': 'an', 
        'augment_mask': 'am',
        'augment_num': 'anum', 
        'multi_label': 'ml', 
        'multi_valprob_threshold': 'vt'
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
    parser.add_argument("--num_trials", type=int, default=None)
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--min_epoch", type=int, default=40)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--param_dir", default=None,)
    args = parser.parse_args()

    voxel_tensor = torch.stack(voxels)  # your voxels
    or_ids = list(Cbc_res_coords.keys())
    
    # FOR DEBUGGING
    # or_ids = [f'Or1Ad1_{i}' for i in list(range(len(Cbc_res_coords)))] 
    
    or_index_map = {or_id: i for i, or_id in enumerate(or_ids)}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    voxel_shape = voxel_tensor.shape
    C, D, H, W = voxel_shape[-1], voxel_shape[1], voxel_shape[2], voxel_shape[3]

    # ---------------- Hyperparameter Sweep ----------------
    param_choices = {
            'lr': [1e-4, 3e-4, 1e-3, 3e-3],
            'batch_size': [8, 16, 32],
            'hidden_dim': [32, 64, 128],
            'kernel_size': [3, 5],
            'num_conv_layers': [2, 3, 4],
            'dropout': [0.0, 0.3, 0.5],
            'pooling': ['AVG'],
            # 'augment_noise': [0.05, 0.5],   # std for Gaussian noise
            # 'augment_mask': [0.05, 0.5],     # prob for masking
            # 'augment_num': [3],
            'augment_noise': [0],   # std for Gaussian noise
            'augment_mask': [0],     # prob for masking
            'augment_num': [0],
            'multi_valprob_threshold': [0.3, 0.5],
            'multi_label': [True]
            }
    
    if args.param_dir: 
        param_dir = str(args.param_dir)
        param_list = []
        for _param in os.listdir(param_dir): 
            with open(os.path.join(param_dir, _param), 'r') as file:
                data = json.load(file)
            param_list.append(data)
            
        num_trials = len(param_list)
        all_params = [_param.values() for _param in param_list] # convert to just list form 

    else: 
        all_params = list(product(*param_choices.values()))
        random.shuffle(all_params)

    sampled_params = all_params[:args.num_trials] if args.num_trials else all_params
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

        # ---------------- Train/Val/Test Split (by unique OR name, not individual predictions) ----------------
        # Extract unique OR names (base name without prediction index)
        unique_or_names = list(set([or_id.split('_')[0].lower() for or_id in or_ids]))
        
        # Binary stratification label: Activated OR (1) vs Non-activated (0)
        activated_set = set(activated_ps6_df.DL_OR.str.lower().values)
        unique_labels = []
        for or_name in unique_or_names:
            unique_labels.append(1 if or_name in activated_set else 0)
        unique_labels = np.array(unique_labels)

        # Split unique OR names with stratification
        train_unique_ors, remain_unique_ors = train_test_split(
            unique_or_names,
            test_size=0.3,  # 30% for val + test
            stratify=unique_labels,   # Important: keeps activation status balanced
            random_state=0
        )

        # Split remaining into val and test
        labels_remain = [unique_labels[unique_or_names.index(or_name)] for or_name in remain_unique_ors]
        val_unique_ors, test_unique_ors = train_test_split(
            remain_unique_ors,
            test_size=0.5,
            stratify=labels_remain,
            random_state=0
        )

        # Now expand back to all voxels: assign all predictions of an OR to the same fold
        train_ORs = [or_id for or_id in or_ids if or_id.split('_')[0].lower() in train_unique_ors]
        val_ORs = [or_id for or_id in or_ids if or_id.split('_')[0].lower() in val_unique_ors]
        test_ORs = [or_id for or_id in or_ids if or_id.split('_')[0].lower() in test_unique_ors]

        # Build index maps for train, val, and test ORs
        train_index_map = {or_id: or_index_map[or_id] for or_id in train_ORs}
        val_index_map = {or_id: or_index_map[or_id] for or_id in val_ORs}
        test_index_map = {or_id: or_index_map[or_id] for or_id in test_ORs}

        # Inflate training voxels only if augmentation is active
        if param_dict["augment_noise"] > 0 or param_dict["augment_mask"] > 0:
            voxel_tensor_train, train_index_map = inflate_voxels(
                voxel_tensor,
                train_index_map,
                ps6_df,
                n_augments=param_dict["augment_num"],
                noise_std=param_dict["augment_noise"],
                mask_prob=param_dict["augment_mask"]
            )
        else:
            voxel_tensor_train = voxel_tensor
            train_index_map = train_index_map

        # Validation set (no augmentation)
        voxel_tensor_val = voxel_tensor
        val_index_map = val_index_map
        
        voxel_tensor_test = voxel_tensor
        test_index_map = test_index_map

        # Build datasets
        train_dataset = ORVoxelDataset(voxel_tensor_train, train_index_map, ps6_df)
        val_dataset = ORVoxelDataset(voxel_tensor_val, val_index_map, ps6_df)
        test_dataset = ORVoxelDataset(voxel_tensor_test, test_index_map, ps6_df)

        
        train_loader = DataLoader(train_dataset, 
                                  batch_size=param_dict["batch_size"], 
                                  shuffle=True)
        val_loader   = DataLoader(val_dataset, 
                                  batch_size=param_dict["batch_size"])
        test_loader  = DataLoader(test_dataset, 
                                  batch_size=param_dict["batch_size"])

        # FOR DEBUGGING 
        # print(val_dataset.or_names)
        # print(val_dataset.targets)
        # break

        model = OROdorCNN(
            voxel_shape=(C, D, H, W),
            num_classes=NUM_ODORS,
            kernel_size=param_dict['kernel_size'],
            hidden_dim=param_dict['hidden_dim'],
            num_conv_layers=param_dict['num_conv_layers'],
            conv_channels=tuple([16 * 2**i for i in range(param_dict['num_conv_layers'])]),
            dropout=param_dict['dropout'],
            pool_type=param_dict['pooling']
        )

        best_model_path, train_losses, val_losses, val_accs = train_model(
            model, train_loader, val_loader, device, run_name,
            lr=param_dict['lr'],
            max_epochs=args.max_epoch,
            min_epoch=args.min_epoch,
            patience=args.patience, 
            threshold=param_dict['multi_valprob_threshold'],
            multi_label=param_dict['multi_label']
        )

        # Run final evaluation on test set using the saved best model
        try:
            evaluate_model(model, test_loader, device, run_name,
                           multi_label=param_dict['multi_label'],
                           threshold=param_dict['multi_valprob_threshold'],
                           model_path=best_model_path)
        except Exception as e:
            print(f"Test evaluation failed for run {run_name}: {e}")
        
if __name__ == "__main__":
    main()