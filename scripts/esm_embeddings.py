import os 
import numpy as np 
import pandas as pd 
import torch
import esm
from tqdm import tqdm 


# === PDB READING FUNCTIONS  ===
AA_THREE_TO_ONE = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"
}
        
def load_pdb_coordinates(pdb_file, single_aa_name=True, chain_id=None, keep_ligand=False):
    """
    Extracts atomic coordinates, separates ligand coordinates, and converts amino acid sequence to single-letter notation.

    :param pdb_file: Path to the PDB file.
    :param single_aa_name: Whether to return the sequence in single-letter notation.
    :param chain_id: Specific chain ID to extract (default: None, keeps all chains).
    :param keep_ligand: Whether to separate ligand coordinates from the main protein structure.
    :return: A tuple of (coordinates, backbone coordinates, sequence, ligand coordinates if keep_ligands=True).
    """
    atoms = read_pdb(pdb_file, chain_id=chain_id, include_ligands=keep_ligand)
    coords = []
    backbone = []
    sequence = []
    ligands = []

    for atom in atoms:
        x, y, z = atom['x'], atom['y'], atom['z']
        
        if atom['res_name'] in AA_THREE_TO_ONE.keys():  # Check if it's a standard amino acid
            coords.append([x, y, z])
            
            if atom['name'] == 'CA':  # Select alpha-carbon atoms
                backbone.append([x, y, z])
                res_name = atom['res_name']
                
                if single_aa_name:
                    single_letter = AA_THREE_TO_ONE.get(res_name, "X")  # Use "X" for unknown residues
                    sequence.append(single_letter)
                else:
                    sequence.append((res_name, atom['res_num']))  # Store aa and residue number
        else:
            ligands.append([x, y, z])  # Store ligand coordinates separately
    
    if single_aa_name:
        sequence = "".join(sequence)

    if keep_ligand:
        return np.array(coords), np.array(backbone), sequence, np.array(ligands)
    else:
        return np.array(coords), np.array(backbone), sequence

def read_pdb(pdb_file, chain_id=None, include_ligands=True):
    """
    Reads a PDB file and extracts atom information while ensuring strict PDB format adherence.
    Optionally filters atoms by a specific chain ID and includes/excludes ligands (HETATM lines).

    :param pdb_file: Path to the PDB file.
    :param chain_id: Specific chain ID to extract (default: None, keeps all chains).
    :param include_ligands: Boolean flag to include ligands (HETATM) or not (default: True).
    :return: List of atom dictionaries containing parsed PDB information.
    """
    atoms = []
    with open(pdb_file, 'r') as file:
        for line in file:
            if line.startswith("ATOM") or (include_ligands and line.startswith("HETATM")):
                try:
                    chain = line[21:22]
                    if chain_id is None or chain in chain_id:
                        atom = {
                            "serial": int(line[6:11]),
                            "name": line[12:16].strip(),
                            "alt_loc": line[16:17],
                            "res_name": line[17:20].strip(),
                            "chain": chain,
                            "res_num": int(line[22:26]),
                            "x": float(line[30:38]),
                            "y": float(line[38:46]),
                            "z": float(line[46:54]),
                            "occupancy": float(line[54:60]),
                            "temp_factor": float(line[60:66]),
                            "element": line[76:78].strip(),
                            "charge": line[78:80].strip()
                        }
                        atoms.append(atom)
                except ValueError:
                    print(f"Warning: Could not parse line: {line.strip()}")
    return atoms

# Extract sequence for all mouse ORs via previously queried AF structure. 
pdb_dir_path = '/mnt/data2/Justice/AF_files/AF_tmaligned_pdb/'
pdb_files = os.listdir(pdb_dir_path)
Olfr_DL = np.load('/mnt/data2/Justice/OR_learning/files/Olfr_DL.npy', allow_pickle=True).item()
OR_seq = []
for _pdb in pdb_files: 
    _pdb_path = os.path.join(pdb_dir_path, _pdb)
    _OR = Olfr_DL.get(_pdb.split('_')[0], None)
    if _OR: 
        OR_seq.append((_OR, load_pdb_coordinates(_pdb_path)[2]))
        
# Save for tracking OR indexing from ESM         
np.save('/mnt/data2/Justice/OR_learning/files/ESM/esm_OR_order.npy' , OR_seq)    


# === Load ESM2 model ===
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results


# === Process in batches ===
all_seq_embeddings = []
all_residue_embeddings = []

for i, _entry in tqdm(enumerate(OR_seq), total=len(OR_seq)):
    _OR, seq = _entry
    data = [(_OR, seq)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[6], return_contacts=False)
    
    token_embeddings = results["representations"][6]  # layer 6 for esm2_t6_8M
    sequence_embedding = token_embeddings[0, 1:len(seq)+1].mean(0).cpu().numpy()  # mean of residues
    residue_embeddings = token_embeddings[0, 1:len(seq)+1].cpu().numpy()  # per residue

    all_seq_embeddings.append(sequence_embedding)
    all_residue_embeddings.append(residue_embeddings)
    
  
# === Save sequence embeddings ===
np.save("/mnt/data2/Justice/OR_learning/files/ESM/sequence_embeddings.npy", np.array(all_seq_embeddings))  # shape: (N, D)
pd.DataFrame(all_seq_embeddings).to_csv("/mnt/data2/Justice/OR_learning/files/ESM/sequence_embeddings.csv", index=False)

# === Save residue embeddings ===
np.save("/mnt/data2/Justice/OR_learning/files/ESM/residue_embeddings.npy", np.array(all_residue_embeddings, dtype=object))
print("✅ Embeddings saved!")  
    
    
    
# batch_labels, batch_strs, batch_tokens = batch_converter(data)
# batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

# # Extract per-residue representations (on CPU)
# with torch.no_grad():
#     results = model(batch_tokens, repr_layers=[33], return_contacts=True)
# token_representations = results["representations"][33]

# # Generate per-sequence representations via averaging
# # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
# sequence_representations = []
# for i, tokens_len in enumerate(batch_lens):
#     sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))

# # Look at the unsupervised self-attention map contact predictions
# import matplotlib.pyplot as plt
# for (_, seq), tokens_len, attention_contacts in zip(data, batch_lens, results["contacts"]):
#     plt.matshow(attention_contacts[: tokens_len, : tokens_len])
#     plt.title(seq)
#     plt.show()
    
    
