import numpy as np
import os
import re 
import itertools
from scipy.spatial.distance import squareform, pdist
from sklearn.manifold import MDS
from collections import Counter
from tabulate import tabulate



AA_THREE_TO_ONE = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"
}

# Below functions for obtaining sequence alignment from structural alignment 
def structural_alignment_dp(ref_backbone, ref_sequence, tgt_backbone, tgt_sequence, gap_penalty):
    """
    Aligns two sequences based on their 3D backbone coordinates using dynamic programming, while considering structural distances 
    between residues. Gaps are introduced based on the specified gap penalty.

    Parameters:
        ref_backbone (list of numpy arrays): Backbone coordinates of the reference structure (shape: n x 3).
        ref_sequence (str): Amino acid sequence for the reference structure.
        tgt_backbone (list of numpy arrays): Backbone coordinates of the target structure (shape: n x 3).
        tgt_sequence (str): Amino acid sequence for the target structure.
        gap_penalty (float): Penalty for introducing gaps in the alignment. A higher value discourages gaps, 
                             while a lower value allows more gaps.

    Returns:
        tuple:
            - str: Aligned reference sequence with gaps.
            - str: Aligned target sequence with gaps.

    Alignment Strategy:
        - The function calculates the pairwise alignment cost based on the Euclidean distance between residues' 
          backbone coordinates.
        - A dynamic programming approach fills a table where each entry represents the minimum alignment cost 
          (considering both residue matches and gap penalties).
        - The gap_penalty controls the tradeoff between penalizing gaps and prioritizing matching residues.
            - Higher gap_penalty discourages gaps, forcing more residues to be aligned even if structurally distant.
            - Lower gap_penalty allows more flexibility to introduce gaps, resulting in more accurate alignments 
              for sequences with significant structural differences or insertions/deletions.

    Notes:
        - Choose an appropriate gap_penalty based on sequence similarity and structural distance.
        - Experiment with different values for gap_penalty to balance gap penalties and structural alignment.
    """
        
    n_ref = len(ref_backbone)
    n_tgt = len(tgt_backbone)

    # Initialize DP and traceback matrices
    dp = np.full((n_ref + 1, n_tgt + 1), float("inf"))
    traceback = np.empty((n_ref + 1, n_tgt + 1), dtype="U1")

    # Initialize base cases for gaps
    dp[0, 0] = 0  # No cost for starting
    for i in range(1, n_ref + 1):
        dp[i, 0] = i * gap_penalty  # Cumulative gap penalty for leading gaps in ref
        traceback[i, 0] = "U"
    for j in range(1, n_tgt + 1):
        dp[0, j] = j * gap_penalty  # Cumulative gap penalty for leading gaps in target
        traceback[0, j] = "L"

    # Fill the DP matrix
    for i in range(1, n_ref + 1):
        for j in range(1, n_tgt + 1):
            dist = np.linalg.norm(ref_backbone[i - 1] - tgt_backbone[j - 1])
            match_score = dp[i - 1, j - 1] + dist
            gap_ref_score = dp[i - 1, j] + gap_penalty
            gap_tgt_score = dp[i, j - 1] + gap_penalty

            dp[i, j] = min(match_score, gap_ref_score, gap_tgt_score)
            if dp[i, j] == match_score:
                traceback[i, j] = "D"  # Diagonal
            elif dp[i, j] == gap_ref_score:
                traceback[i, j] = "U"  # Up
            elif dp[i, j] == gap_tgt_score:
                traceback[i, j] = "L"  # Left

    # Traceback to generate aligned sequences
    aligned_ref, aligned_tgt = [], []
    i, j = n_ref, n_tgt
    while i > 0 or j > 0:
        if traceback[i, j] == "D":
            aligned_ref.append(ref_sequence[i - 1])
            aligned_tgt.append(tgt_sequence[j - 1])
            i -= 1
            j -= 1
        elif traceback[i, j] == "U":
            aligned_ref.append(ref_sequence[i - 1])
            aligned_tgt.append("-")
            i -= 1
        elif traceback[i, j] == "L":
            aligned_ref.append("-")
            aligned_tgt.append(tgt_sequence[j - 1])
            j -= 1

    # Reverse the sequences to get the final alignment
    aligned_ref = "".join(reversed(aligned_ref))
    aligned_tgt = "".join(reversed(aligned_tgt))

    return aligned_ref, aligned_tgt


# def generate_sequence_alignment_pairs_fromPDB(reference_pdb, 
#                                               target_pdbs, 
#                                               load_pdb_fn,
#                                               labels=None,  
#                                               gap_penalty=5.0):
#     """
#     Generates a Chimera-like sequence alignment based on structural alignment of proteins.

#     Parameters:
#         reference_pdb (str): File path to the reference PDB structure.
#         target_pdbs (list): List of file paths to the target PDB structures.
#         load_pdb_fn (function): Function to load PDB files. Should return (header, backbone_coords, sequence).
#         residue_cutoff (float): Cutoff distance used as the gap penalty during structural alignment.

#     Returns:
#         dict: A dictionary where keys are PDB filenames and values are the aligned sequences.
#     """
    
#     _, ref_backbone, ref_sequence = load_pdb_fn(reference_pdb)
#     # ref_basename = os.path.basename(reference_pdb).split('_')[0]
#     # alignment = {ref_basename: ref_sequence}
#     alignment = {}
    
#     for i, target_pdb in enumerate(target_pdbs):
#         # if target_pdb == reference_pdb:
#         #     continue
#         _, tgt_backbone, tgt_sequence = load_pdb_fn(target_pdb)
#         aligned_ref, aligned_tgt = structural_alignment_dp(
#             ref_backbone, ref_sequence, tgt_backbone, tgt_sequence, gap_penalty
#         )
#         if labels: 
#             tgt_basename = labels[i]
#         else: 
#             tgt_basename = os.path.basename(target_pdb).split('_')[0]
#         alignment[tgt_basename] = (aligned_ref, aligned_tgt)

#     return alignment


def generate_sequence_alignment_pairs(reference, targets, load_pdb_fn=None, labels=None, gap_penalty=5.0):
    """
    Generates a Chimera-like sequence alignment based on structural alignment of proteins.

    Parameters:
        reference (str or tuple): PDB file path or a tuple (atom_coords, backbone_coords, sequence).
        targets (list): List of PDB file paths or tuples [(atom_coords, backbone_coords, sequence), ...].
        load_pdb_fn (function, optional): Function to load PDB files. Should return (atom_coords, backbone_coords, sequence).
        labels (list, optional): Custom labels for target sequences.
        gap_penalty (float): Gap penalty used during structural alignment.

    Returns:
        dict: A dictionary where keys are structure labels and values are aligned sequences.
    """
    
    # Determine if reference is a file path or preloaded coordinates
    if isinstance(reference, str):
        if not load_pdb_fn:
            raise ValueError("load_pdb_fn is required when providing file paths.")
        _, ref_backbone, ref_sequence = load_pdb_fn(reference)
    else:
        _, ref_backbone, ref_sequence = reference
    
    alignment = {}
    
    for i, target in enumerate(targets):
        if isinstance(target, str):
            if not load_pdb_fn:
                raise ValueError("load_pdb_fn is required when providing file paths.")
            _, tgt_backbone, tgt_sequence = load_pdb_fn(target)
        else:
            _, tgt_backbone, tgt_sequence = target
        
        aligned_ref, aligned_tgt = structural_alignment_dp(
            ref_backbone, ref_sequence, tgt_backbone, tgt_sequence, gap_penalty
        )
        
        if labels:
            tgt_basename = labels[i]
        elif isinstance(target, str):
            tgt_basename = os.path.basename(target).split('_')[0]
        else:
            tgt_basename = f"target_{i}"
        
        alignment[tgt_basename] = (aligned_ref, aligned_tgt)
    
    return alignment


def union_gaps_with_consistency(ref_targ_pairs):
    """
    Generates a consensus reference sequence by unifying gaps across references and aligns target sequences to it.

    Parameters:
        ref_targ_pairs (list of tuples): Each tuple contains:
            - str: Reference sequence with possible gaps.
            - str: Target sequence corresponding to the reference.

    Returns:
        tuple:
            - str: Consensus reference sequence with unified gaps.
            - list of str: List of aligned target sequences to the consensus reference.
    """
    def merge_consensus_with_ref(consensus, ref):
        new_consensus = []
        ref_idx, cons_idx = 0, 0

        while ref_idx < len(ref) or cons_idx < len(consensus):
            if ref_idx < len(ref) and cons_idx < len(consensus):
                if ref[ref_idx] == consensus[cons_idx]:
                    new_consensus.append(consensus[cons_idx])
                    ref_idx += 1
                    cons_idx += 1
                elif ref[ref_idx] == "-":
                    new_consensus.append("-")
                    ref_idx += 1
                elif consensus[cons_idx] == "-":
                    new_consensus.append("-")
                    cons_idx += 1
                else:
                    new_consensus.append("-")
                    ref_idx += 1
            elif ref_idx < len(ref):
                new_consensus.append("-")
                ref_idx += 1
            elif cons_idx < len(consensus):
                new_consensus.append(consensus[cons_idx])
                cons_idx += 1

        return "".join(new_consensus)

    # def align_to_consensus(consensus, ref, targ):
    #     aligned_ref, aligned_targ = list(ref), list(targ)
    #     consensus_idx = 0

    #     for i, char in enumerate(ref):
    #         while consensus_idx < len(consensus) and consensus[consensus_idx] != char:
    #             aligned_ref.insert(i + consensus_idx, "-")
    #             aligned_targ.insert(i + consensus_idx, "-")
    #             consensus_idx += 1
    #         consensus_idx += 1

    #     return "".join(aligned_ref), "".join(aligned_targ)
    def align_to_consensus(consensus, ref, targ):
        """
        Aligns the given reference and target sequences to the consensus sequence,
        ensuring that gaps in the consensus are reflected in both reference and target.
        
        Parameters:
            consensus (str): The consensus reference sequence with unified gaps.
            ref (str): The original reference sequence.
            targ (str): The target sequence corresponding to the reference.

        Returns:
            tuple:
                - str: Aligned reference sequence.
                - str: Aligned target sequence.
        """
        aligned_ref = []
        aligned_targ = []
        ref_idx = 0  # Index for the reference sequence
        targ_idx = 0  # Index for the target sequence

        for cons_char in consensus:
            if ref_idx < len(ref) and ref[ref_idx] == cons_char:
                # Match: Add both reference and target characters
                aligned_ref.append(ref[ref_idx])
                aligned_targ.append(targ[targ_idx])
                ref_idx += 1
                targ_idx += 1
            elif ref_idx < len(ref) and ref[ref_idx] == '-':
                # Gap in the reference: Reflect the gap in both sequences
                aligned_ref.append('-')
                aligned_targ.append('-')
                ref_idx += 1
            elif cons_char == '-':
                # Gap in the consensus: Add gap to both sequences
                aligned_ref.append('-')
                aligned_targ.append('-')
            else:
                # Mismatch: Add gaps to both sequences (shouldn't frequently happen)
                aligned_ref.append('-')
                aligned_targ.append('-')

        return "".join(aligned_ref), "".join(aligned_targ)
    Seq_identity = list(ref_targ_pairs.keys())
    ref_targ_pairs = list(ref_targ_pairs.values())

    consensus_ref = ref_targ_pairs[0][0]
    for ref, _ in ref_targ_pairs[1:]:
        consensus_ref = merge_consensus_with_ref(consensus_ref, ref)
    # return consensus_ref 
    aligned_sequences = {}
    for i, (ref, targ) in enumerate(ref_targ_pairs):
        aligned_ref, aligned_targ = align_to_consensus(consensus_ref, ref, targ)
        # aligned_sequences.append(aligned_targ)
        aligned_sequences[Seq_identity[i]] =  aligned_targ

    return aligned_sequences


# Functions for making Weblogo
def trim_alignment(alignment_dict, gap_threshold=0.9):
    """
    Trims columns from an alignment dictionary based on the percentage of gaps.
    
    Parameters:
        alignment_dict (dict): Dictionary where keys are sequence identifiers and values are aligned sequences.
        gap_threshold (float): Fraction of gaps above which a column is removed. Default is 0.9.
        
    Returns:
        tuple: (filtered_alignment, frequency_matrix)
            - filtered_alignment (np.ndarray): The alignment with high-gap columns removed.
            - frequency_matrix (np.ndarray): The recalculated frequency matrix.
    """
    aligned_sequences = list(alignment_dict.values())
    
    # Convert aligned sequences into a NumPy array
    alignment_array = np.array([list(seq) for seq in aligned_sequences])
    
    # Calculate gap fractions
    gap_counts = np.sum(alignment_array == "-", axis=0)
    gap_fractions = gap_counts / len(aligned_sequences)
    
    # Identify columns to keep
    columns_to_keep = gap_fractions <= gap_threshold
    filtered_alignment = alignment_array[:, columns_to_keep]
    
    # Define alphabet and index mapping
    alphabet = list("ACDEFGHIKLMNPQRSTVWY-")
    alphabet_idx = {aa: i for i, aa in enumerate(alphabet)}
    
    # Initialize frequency matrix
    alignment_length = filtered_alignment.shape[1]
    frequency_matrix = np.zeros((len(alphabet), alignment_length))
    
    # Fill the frequency matrix
    for sequence in filtered_alignment:
        for pos, char in enumerate(sequence):
            frequency_matrix[alphabet_idx[char], pos] += 1
    
    # Normalize the frequency matrix to probabilities
    frequency_matrix /= len(filtered_alignment)
    
    return filtered_alignment, frequency_matrix


def map_aligned_residues(original_residues, aligned_seq, check_match=True):
    """
    Maps original residue indices to their corresponding positions in the aligned sequence.
    Optionally verifies if the amino acid matches at the new position.

    :param original_residues: List of residue labels (e.g., ['G71', 'A85']).
    :param aligned_seq: The gapped aligned sequence.
    :param check_match: Whether to verify if the amino acid matches (default: True).
    :return: Dictionary mapping original residue labels to aligned positions (and validation status if enabled).
    """
    # Convert original residue labels to numeric indices (1-based)
    original_indices = {int(res[1:]): res[0] for res in original_residues}

    count = 0  # Tracks residues in the original sequence (excluding '-')
    index_mapping = []  # Store mapped indices with optional validation

    for i, char in enumerate(aligned_seq):
        if char != '-':
            count += 1  # Increment for non-gap residues
            if count in original_indices:
                aligned_index = i + 1  # Convert to 1-based index
                
                # Check if the amino acid matches
                expected_aa = original_indices[count]
                actual_aa = char
                match_status = expected_aa == actual_aa

                if check_match:
                    if match_status:
                        # index_mapping += [f"{expected_aa}{aligned_index}"]
                        index_mapping += [aligned_index]
                else:                    
                    # index_mapping += [f"{expected_aa}{aligned_index}"] 
                    index_mapping += [aligned_index]

    return index_mapping


def map_aligned_indices(original_indices, aligned_seq):
    """
    Maps original residue indices to their new positions in the aligned sequence.

    :param original_indices: Array of original residue indices (1-based).
    :param original_seq: The ungapped original sequence.
    :param aligned_seq: The gapped aligned sequence.
    :return: Dictionary mapping original residue indices to aligned positions.
    """
    count = 0  # Tracks residues in the original sequence (excluding gaps)
    # index_mapping = {}  # Store mapped indices
    index_mapping = []
    for i, char in enumerate(aligned_seq):
        if char != '-':
            count += 1  # Increment for non-gap residues
            if count in original_indices:
                aligned_index = i + 1  # Convert to 1-based index
                # index_mapping[count] = aligned_index
                index_mapping += [aligned_index]
    return index_mapping



# def map_highlight_positions(primary_sequence, aligned_sequence, highlight_positions, verify_aa = True):
#     """
#     Maps residue positions from the primary sequence to the aligned sequence with verification.

#     Parameters:
#         primary_sequence (str): The primary sequence without gaps.
#         aligned_sequence (str): The alignment sequence with gaps.
#         highlight_positions (list): List of positions to highlight in the format "H104", "F155", etc.

#     Returns:
#         dict: Verified mapping of primary sequence indices to alignment indices for valid positions.

#     Raises:
#         ValueError: If the residue in the highlight position does not match the primary sequence.
#     """
#     # Parse the highlight positions into a list of tuples (aa, index)
#     parsed_positions = []
#     for pos in highlight_positions:
#         try:
#             aa = pos[0]  # Residue letter
#             index = int(pos[1:])  # Residue number
#             parsed_positions.append((aa, index))
#         except ValueError:
#             raise ValueError(f"Invalid format for highlight position: {pos}")

#     primary_index = 0  # Index for primary sequence
#     mapping = {}  # Mapping from primary sequence index to alignment sequence index

#     for aligned_index, char in enumerate(aligned_sequence):
#         if char != '-':  # Only map non-gap characters
#             # Map the primary sequence position to the alignment position
#             primary_index += 1

#             # Verify and add to the mapping
#             for aa, index in parsed_positions:
#                 if index == primary_index:
#                     # Verify the residue matches the primary sequence
#                     if verify_aa: 
#                         if primary_sequence[primary_index - 1] != aa:  # -1 for 0-based index
#                             raise ValueError(
#                                 f"Residue mismatch: expected {primary_sequence[primary_index - 1]} "
#                                 f"at position {index}, but got {aa}."
#                             )
#                     # Add the mapping
#                     mapping[f"{aa}{index}"] = aligned_index

#         # Stop if we've mapped all positions in the primary sequence
#         if primary_index >= len(primary_sequence):
#             break

#     return mapping

def map_residues_to_filtered(aligned_sequence, filtered_sequence, target_residues, verify_aa=True):
    """
    Maps residue positions from the original sequence to their corresponding positions in the filtered sequence.
    
    Parameters:
        # primary_sequence (str): The original unaligned sequence.
        aligned_sequence (str): The sequence after alignment (including gaps).
        filtered_sequence (str): The sequence after filtering (some residues may be removed).
        target_residues (list): List of residue positions to track, formatted as ["L72", "I103", ...].
    
    Returns:
        dict: Mapping of target residues to their positions in the filtered sequence.
    """
    
    # Step 1: Map filtered_sequence to aligned_sequence
    filtered_to_aligned = {}  # {filtered index (1-based): aligned index}
    filtered_index = 1  # 1-based index for filtered_sequence

    for align_index, char in enumerate(aligned_sequence):
        if filtered_index <= len(filtered_sequence) and filtered_sequence[filtered_index - 1] == char:
            filtered_to_aligned[filtered_index] = align_index + 1  # 1-based indexing
            filtered_index += 1

    # Step 2: Map original_sequence to aligned_sequence
    original_to_aligned = {}  # {original index (1-based): aligned index}
    orig_index = 1  # 1-based index for primary_sequence

    for align_index, char in enumerate(aligned_sequence):
        if char != '-':  # Skip gaps in aligned sequence
            original_to_aligned[orig_index] = align_index + 1  # 1-based indexing
            orig_index += 1

    # Step 3: Locate each target residue in filtered sequence
    residue_mapping = {}
    for res in target_residues:
        aa = res[0]  # Residue letter
        orig_pos = int(res[1:])  # Original sequence position

        # Find corresponding position in aligned sequence
        aligned_pos = original_to_aligned.get(orig_pos)
        if aligned_pos is None:
            print(f"⚠️ Warning: Residue {res} was removed in alignment.")
            continue

        # Find corresponding position in filtered sequence
        filtered_pos = None
        for filt_idx, align_idx in filtered_to_aligned.items():
            if align_idx == aligned_pos:
                filtered_pos = filt_idx
                break

        if filtered_pos is None:
            print(f"⚠️ Warning: Residue {res} was removed after filtering.")
            continue
        
        # Optional: Verify the amino acid identity
        if verify_aa and filtered_sequence[filtered_pos - 1] != aa:
            print(f"⚠️ Warning: Residue {res} was removed after filtering (AA mismatch).")
            continue

        residue_mapping[res] = filtered_pos

    return residue_mapping

# Complete Grantham distance matrix for all 20 amino acids
grantham_matrix = {
    ('A', 'A'): 0,   ('A', 'R'): 112, ('A', 'N'): 111, ('A', 'D'): 126, ('A', 'C'): 195, ('A', 'Q'): 91,  ('A', 'E'): 107, ('A', 'G'): 60,  ('A', 'H'): 86,  ('A', 'I'): 94,
    ('A', 'L'): 96,  ('A', 'K'): 106, ('A', 'M'): 84,  ('A', 'F'): 113, ('A', 'P'): 27,  ('A', 'S'): 99,  ('A', 'T'): 58,  ('A', 'W'): 148, ('A', 'Y'): 112, ('A', 'V'): 64,
    ('R', 'R'): 0,   ('R', 'N'): 86,  ('R', 'D'): 96,  ('R', 'C'): 180, ('R', 'Q'): 43,  ('R', 'E'): 54,  ('R', 'G'): 125, ('R', 'H'): 29,  ('R', 'I'): 97,  ('R', 'L'): 102,
    ('R', 'K'): 26,  ('R', 'M'): 91,  ('R', 'F'): 97,  ('R', 'P'): 103, ('R', 'S'): 110, ('R', 'T'): 71,  ('R', 'W'): 101, ('R', 'Y'): 77,  ('R', 'V'): 96,
    ('N', 'N'): 0,   ('N', 'D'): 23,  ('N', 'C'): 139, ('N', 'Q'): 46,  ('N', 'E'): 42,  ('N', 'G'): 80,  ('N', 'H'): 68,  ('N', 'I'): 149, ('N', 'L'): 153,
    ('N', 'K'): 94,  ('N', 'M'): 142, ('N', 'F'): 158, ('N', 'P'): 91,  ('N', 'S'): 46,  ('N', 'T'): 65,  ('N', 'W'): 174, ('N', 'Y'): 143, ('N', 'V'): 133,
    ('D', 'D'): 0,   ('D', 'C'): 154, ('D', 'Q'): 61,  ('D', 'E'): 45,  ('D', 'G'): 94,  ('D', 'H'): 81,  ('D', 'I'): 168, ('D', 'L'): 172, ('D', 'K'): 101,
    ('D', 'M'): 160, ('D', 'F'): 177, ('D', 'P'): 108, ('D', 'S'): 65,  ('D', 'T'): 85,  ('D', 'W'): 181, ('D', 'Y'): 160, ('D', 'V'): 152,
    ('C', 'C'): 0,   ('C', 'Q'): 154, ('C', 'E'): 170, ('C', 'G'): 159, ('C', 'H'): 174, ('C', 'I'): 198, ('C', 'L'): 198, ('C', 'K'): 202,
    ('C', 'M'): 196, ('C', 'F'): 205, ('C', 'P'): 169, ('C', 'S'): 112, ('C', 'T'): 149, ('C', 'W'): 215, ('C', 'Y'): 194, ('C', 'V'): 192,
    ('Q', 'Q'): 0,   ('Q', 'E'): 29,  ('Q', 'G'): 87,  ('Q', 'H'): 24,  ('Q', 'I'): 109, ('Q', 'L'): 113, ('Q', 'K'): 53,  ('Q', 'M'): 101,
    ('Q', 'F'): 116, ('Q', 'P'): 76,  ('Q', 'S'): 68,  ('Q', 'T'): 42,  ('Q', 'W'): 130, ('Q', 'Y'): 99,  ('Q', 'V'): 96,
    ('E', 'E'): 0,   ('E', 'G'): 98,  ('E', 'H'): 40,  ('E', 'I'): 134, ('E', 'L'): 138, ('E', 'K'): 56,  ('E', 'M'): 126, ('E', 'F'): 140,
    ('E', 'P'): 93,  ('E', 'S'): 80,  ('E', 'T'): 65,  ('E', 'W'): 152, ('E', 'Y'): 122, ('E', 'V'): 121,
    ('G', 'G'): 0,   ('G', 'H'): 98,  ('G', 'I'): 135, ('G', 'L'): 138, ('G', 'K'): 127, ('G', 'M'): 127, ('G', 'F'): 153, ('G', 'P'): 42,
    ('G', 'S'): 56,  ('G', 'T'): 59,  ('G', 'W'): 184, ('G', 'Y'): 147, ('G', 'V'): 109,
    ('H', 'H'): 0,   ('H', 'I'): 94,  ('H', 'L'): 99,  ('H', 'K'): 32,  ('H', 'M'): 87,  ('H', 'F'): 100, ('H', 'P'): 77,  ('H', 'S'): 89,
    ('H', 'T'): 47,  ('H', 'W'): 115, ('H', 'Y'): 83,  ('H', 'V'): 84,
    ('I', 'I'): 0,   ('I', 'L'): 5,   ('I', 'K'): 102, ('I', 'M'): 10,  ('I', 'F'): 21,  ('I', 'P'): 95,  ('I', 'S'): 142, ('I', 'T'): 89,
    ('I', 'W'): 61,  ('I', 'Y'): 33,  ('I', 'V'): 29,
    ('L', 'L'): 0,   ('L', 'K'): 107, ('L', 'M'): 10,  ('L', 'F'): 22,  ('L', 'P'): 98,  ('L', 'S'): 145, ('L', 'T'): 92,  ('L', 'W'): 61,
    ('L', 'Y'): 36,  ('L', 'V'): 32,
    ('K', 'K'): 0,   ('K', 'M'): 95,  ('K', 'F'): 102, ('K', 'P'): 103, ('K', 'S'): 121, ('K', 'T'): 78,  ('K', 'W'): 110, ('K', 'Y'): 85,
    ('K', 'V'): 97,
    ('M', 'M'): 0,   ('M', 'F'): 28,  ('M', 'P'): 87,  ('M', 'S'): 135, ('M', 'T'): 81,  ('M', 'W'): 67,  ('M', 'Y'): 36,  ('M', 'V'): 21,
    ('F', 'F'): 0,   ('F', 'P'): 114, ('F', 'S'): 155, ('F', 'T'): 103, ('F', 'W'): 40,  ('F', 'Y'): 22,  ('F', 'V'): 50,
    ('P', 'P'): 0,   ('P', 'S'): 74,  ('P', 'T'): 38,  ('P', 'W'): 147, ('P', 'Y'): 110, ('P', 'V'): 68,
    ('S', 'S'): 0,   ('S', 'T'): 58,  ('S', 'W'): 177, ('S', 'Y'): 144, ('S', 'V'): 124,
    ('T', 'T'): 0,   ('T', 'W'): 128, ('T', 'Y'): 92,  ('T', 'V'): 69,
    ('W', 'W'): 0,   ('W', 'Y'): 37,  ('W', 'V'): 88,
    ('Y', 'Y'): 0,   ('Y', 'V'): 55,
    ('V', 'V'): 0,
}

def get_grantham_distance(aa1, aa2):
    """Retrieve Grantham distance for an amino acid pair, considering symmetry."""
    if aa1 == aa2:
        return 0
    if not grantham_matrix.get((aa1, aa2), grantham_matrix.get((aa2, aa1), None)): 
        print(f'NONE FOUND FOR {aa1, aa2}')
        return None 
    return grantham_matrix.get((aa1, aa2), grantham_matrix.get((aa2, aa1), None))

def compute_sequence_grantham(seqA, seqB, aggregate='mean', gap_handling='ignore', gap_penalty=100):
    """Compute the Grantham distance between two aligned sequences."""
    if len(seqA) != len(seqB):
        raise ValueError("Sequences must be aligned and of equal length.")

    grantham_scores = []
    max_grantham_scores = []

    for aa1, aa2 in zip(seqA, seqB):
        if '-' in (aa1, aa2):
            if gap_handling == 'ignore':
                continue
            elif gap_handling == 'penalize':
                grantham_scores.append(gap_penalty)
                max_grantham_scores.append(215)
            elif gap_handling == 'impute':
                aa1 = 'A' if aa1 == '-' else aa1
                aa2 = 'A' if aa2 == '-' else aa2  

        dist = get_grantham_distance(aa1, aa2)
        if dist is not None:
            grantham_scores.append(dist)
            max_grantham_scores.append(215)

    if not grantham_scores:
        return None

    if aggregate == 'mean':
        return np.mean(grantham_scores)
    elif aggregate == 'sum':
        return np.sum(grantham_scores)
    elif aggregate == 'normalized':
        return np.sum(grantham_scores) / np.sum(max_grantham_scores)
    else:
        raise ValueError("Invalid aggregate method. Choose 'mean', 'sum', or 'normalized'.")

def pairwise_grantham_matrix(sequences, aggregate='mean', gap_handling='ignore'):
    """Compute pairwise Grantham distances for a list of sequences."""
    num_seqs = len(sequences)
    matrix = np.full((num_seqs, num_seqs), np.nan)  # Start with NaN

    for i, j in itertools.combinations(range(num_seqs), 2):
        dist = compute_sequence_grantham(sequences[i], sequences[j], aggregate, gap_handling)
        if dist is not None:
            matrix[i, j] = matrix[j, i] = dist

    # Fill diagonal with 0 (self-distance)
    np.fill_diagonal(matrix, 0)

    # Replace NaN values with max possible Grantham distance
    max_distance = 215
    matrix = np.nan_to_num(matrix, nan=max_distance)

    return matrix


def compute_consensus_sequence(sequences):
    """Generate a consensus sequence using majority rule at each position."""
    aligned_length = len(sequences[0])
    consensus = ''

    for i in range(aligned_length):
        column = [seq[i] for seq in sequences if seq[i] != '-']  # Ignore gaps
        if column:
            most_common_aa = Counter(column).most_common(1)[0][0]
            consensus += most_common_aa
        else:
            consensus += '-'

    return consensus

def population_grantham_scores(sequences, ref_sequence = None, aggregate='mean', gap_handling='ignore'):
    """
    Compute Grantham distance of each sequence from the reference sequence or consensus sequence if reference not provided.
    •	Computes Grantham distance from a reference sequence.
	•	Helps determine how much each sequence diverges from the population norm.

    """
    if ref_sequence: 
        ref_sequence = compute_consensus_sequence(sequences)
    scores = [compute_sequence_grantham(seq, ref_sequence, aggregate, gap_handling) for seq in sequences]
    return scores

def grantham_mds_projection(sequences, aggregate='mean', gap_handling='ignore', n_components=2):
    """
    Perform MDS on pairwise Grantham distances to project sequences in a lower-dimensional space.
    •	Uses Multidimensional Scaling (MDS) to embed sequences into low-dimensional space.
	•	Great for visualization and clustering.
    """
    dist_matrix = pairwise_grantham_matrix(sequences, aggregate, gap_handling)
    mds = MDS(n_components=n_components, dissimilarity="precomputed", random_state=42)
    return mds.fit_transform(dist_matrix), dist_matrix



def interpolate_bw(position, known_positions):
    """
    Infer the BW numbering of a given residue position based on known BW positions.

    Parameters:
    - position: Residue position in the sequence (integer).
    - known_positions: Dictionary of known BW positions {residue index (str): BW number (str)}.

    Returns:
    - Interpolated BW number (float).
    """
    # Convert residue positions to integers and BW numbers to floats
    # known_positions = {int(k[1:]): float(v) for k, v in known_positions.items()}

    # Get sorted residue indices
    sorted_positions = sorted(known_positions.keys())

    # Find the closest midpoint reference (X.50)
    closest_idx = min(sorted_positions, key=lambda x: abs(x - position))
    closest_bw = known_positions[closest_idx]

    # Calculate offset from the closest BW reference
    residue_offset = position - closest_idx
    inferred_bw = float(closest_bw) + (residue_offset * 0.01)

    return round(inferred_bw, 2)

# def map_residues_to_bw(query_residues, ref_seq_name, alignment, target_seq_name, bw_positions=None):
#     """
#     Maps query residues from the reference sequence to the target sequence and infers BW numbering.

#     Parameters:
#     - query_residues: List of residue positions (e.g., ['H104', 'F155']).
#     - ref_seq_name: The name of the reference sequence in the alignment.
#     - alignment: Dictionary of sequence alignments.
#     - target_seq_name: The target sequence name in the alignment.
#     - bw_positions: Dictionary of known BW numbers for the reference sequence.

#     Returns:
#     - Dictionary mapping target residues to inferred BW numbers.
#     """
    
#     if bw_positions is None: 
#         print(f'using default Or51E2 bw_position . . . on {ref_seq_name}')
#         bw_positions = {"N41":'1.50', "D69": '2.50', "R121":'3.50', 
#                        "V149":'4.50', "C178":'45.50', "D209":'5.50', 
#                        "P253":'6.50', "P288":'7.50'}    
    
#     ref_seq = alignment[ref_seq_name]
#     target_seq = alignment[target_seq_name]

#     # Step 1: Convert BW positions to residue indices
#     bw_residue_positions = {int(res[1:]): float(bw) for res, bw in bw_positions.items()}

#     # Step 2: Map OBR residues directly using the aligned sequence
#     target_bw_map = {}
#     ref_res_index    = 0  # Track reference sequence index ignoring gaps
#     target_res_index = 0
    
#     for ref_res in ref_seq:
#         if ref_res == "-":
#             continue  # Skip gaps in reference sequence
#         ref_res_index += 1  # Increment non-gap reference index
        
#         if target_seq[ref_res_index - 1] != "-": 
#             target_res_index += 1

#         for _res in query_residues:
#             # obr_res_num = int(obr_res[1:])  # Extract numeric residue position (e.g., 'H104' → 104)
#             res_num = int(re.sub( r"\D", "", _res))  # Extract numeric residue position (e.g., 'H104' → 104)
            
#             if ref_res_index == res_num:  # Find the corresponding aligned position
#                 target_res = target_seq[ref_res_index - 1]  # Get target residue at the same alignment position
                
#                 if target_res != "-":  # Ignore cases where target sequence has a gap
#                     inferred_bw = interpolate_bw(ref_res_index, bw_residue_positions)
#                     target_bw_map[f"{target_res}{target_res_index}"] = inferred_bw

#     return target_bw_map

def update_bw_positions(ref_seq_name, alignment, bw_positions):
    """
    Updates BW numbering to be applicable to all sequences based on the alignment.

    Parameters:
    - ref_seq_name: The reference sequence name in the alignment.
    - alignment: Dictionary of sequence alignments {seq_name: sequence}.
    - bw_positions: Dictionary of known BW numbers for the reference sequence.

    Returns:
    - Dictionary {aligned residue index: BW number}, valid for all sequences.
    """

    ref_seq = alignment[ref_seq_name]
    
    # Convert BW positions to numeric residue indices
    bw_residue_positions = {int(re.sub(r"\D", "", res)): float(bw) for res, bw in bw_positions.items()}

    # New mapping of aligned indices to BW numbers
    updated_bw_positions = {}

    ref_res_index = 0  # Tracks residue position in ref sequence (ignoring gaps)
    aligned_index = 0  # Tracks residue position in alignment

    # Iterate through the alignment
    for i, ref_res in enumerate(ref_seq):
        if ref_res != "-":  
            ref_res_index += 1  
        
        if any(seq[i] != "-" for seq in alignment.values()):  
            # If any sequence has a residue at this position, count it as an aligned position
            aligned_index += 1  

        # If this ref_res_index matches a known BW position, update it to aligned_index
        if ref_res_index in bw_residue_positions:
            updated_bw_positions[aligned_index] = bw_residue_positions[ref_res_index]  

    return updated_bw_positions

import re

def map_residues_to_bw(residue_list, alignment, target_seq_name, bw_positions):
    """
    Maps residue positions in the target sequence to their corresponding BW numbers, using interpolation when necessary.

    Parameters:
    - residue_list: List of residue positions based on target_seq (e.g., ['H104', 'F155']).
    - alignment: Dictionary containing aligned sequences {seq_name: sequence}.
    - target_seq_name: The sequence name for which we are mapping residues.
    - bw_positions: Dictionary {aligned residue index: BW number}, universal for all sequences.

    Returns:
    - Dictionary mapping target sequence residues to their BW numbers {residue: BW position}.
    """

    target_seq = alignment[target_seq_name]
    
    # **Preprocess**: Create a mapping of residue numbers to original labels
    residue_num_to_label = {int(re.sub(r"\D", "", res)): res for res in residue_list}

    target_res_index = 0  # Track actual residue index in target sequence (ignoring gaps)
    aligned_index = 0  # Track aligned position
    residue_to_bw = {}

    for i, target_res in enumerate(target_seq):
        if target_res != "-":
            target_res_index += 1  # Increment actual residue count
        
        if any(seq[i] != "-" for seq in alignment.values()):  
            aligned_index += 1  # Count as an aligned position if any sequence has a residue here

        # **Check if target_res_index is in the residue list set (O(1) lookup)**
        if target_res_index in residue_num_to_label:
            inferred_bw = interpolate_bw(aligned_index, bw_positions)
            residue_label = re.sub(r"\D", "", residue_num_to_label[target_res_index])
            # residue_to_bw[residue_label] = inferred_bw  
            residue_to_bw[f"{target_res}{residue_label}"] = f'{inferred_bw:.2f}'
    return residue_to_bw



def compare_bw_numbers(target_bw, query_bw):
    """
    Compare BW numbers between a target and a query, identifying overlaps and missing values.
    Displays output in a table format for better readability.

    Parameters:
    - target_bw: Dictionary of BW numbers mapped to a target sequence {residue: BW}.
    - query_bw: Dictionary of BW numbers from the query sequence {residue: BW}.
    """

    # Convert BW values to sets for easier comparison
    target_bw_values = {round(v, 2) for v in target_bw.values()}
    query_bw_values = {round(v, 2) for v in query_bw.values()}

    # Find overlapping BW numbers
    overlapping_bw_entries = {res: bw for res, bw in target_bw.items() if round(bw, 2) in query_bw_values}

    # Find missing BW entries (present in query but not in target)
    missing_query_bw_entries = {res: bw for res, bw in query_bw.items() if round(bw, 2) not in target_bw_values}

    # Prepare table data
    table = []
    max_len = max(len(target_bw), len(overlapping_bw_entries), len(missing_query_bw_entries))

    target_bw_list = list(target_bw.items())
    overlapping_bw_list = list(overlapping_bw_entries.items())
    missing_bw_list = list(missing_query_bw_entries.items())

    for i in range(max_len):
        target_entry = f"{target_bw_list[i][0]}: {target_bw_list[i][1]}" if i < len(target_bw_list) else ""
        overlap_entry = f"{overlapping_bw_list[i][0]}: {overlapping_bw_list[i][1]}" if i < len(overlapping_bw_list) else ""
        missing_entry = f"{missing_bw_list[i][0]}: {missing_bw_list[i][1]}" if i < len(missing_bw_list) else ""

        table.append([target_entry, overlap_entry, missing_entry])

    # Print table
    print(tabulate(table, headers=["Target BW", "Overlapping BW", "Missing Query BW"]))


def filter_get_conservation(df, filter_names=None):
    """
    Compute consensus sequence and conservation from a filtered alignment DataFrame.

    Args:
        df (pd.DataFrame): Filtered alignment (rows = sequences, cols = positions).
        filter_names (list, optional): List of sequence IDs (row labels) to include. 
                                       If None, use all rows.

    Returns:
        consensus_letters (list of str): Consensus amino acid at each position.
        conservation_percent (list of float): Percent conservation at each position.
        positions (list): Column labels corresponding to positions.
    """
    # Subset rows if filter is given
    if filter_names is not None:
        df = df.loc[df.index.intersection(filter_names)]
    
    if df.empty:
        raise ValueError("No sequences found after filtering. Check filter_names or df index.")
    
    consensus_letters = []
    conservation_percent = []

    for col in df.columns:
        col_data = df[col]
        # Exclude gaps
        # aa_series = col_data[col_data != '-']
        aa_series = col_data
        counts = Counter(aa_series)
        if counts:
            most_common_aa, count = counts.most_common(1)[0]
            consensus_letters.append(most_common_aa)
            conservation_percent.append(count / len(aa_series) * 100)
        else:
            consensus_letters.append(' ')
            conservation_percent.append(0.0)
    
    return df, consensus_letters, conservation_percent