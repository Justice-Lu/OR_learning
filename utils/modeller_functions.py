from pdb_functions import *

# Homology Model via modeller 
from modeller import *
from modeller.automodel import *

def run_modeller_homology(aln_file, 
                          template_pdb,
                          target_name,
                          atom_files_directory=['.'],
                          output_dir=None, 
                          num_models=3):
    """
    Runs homology modeling using Modeller and organizes the output files.

    :param aln_file: Path to the alignment file (.fa) containing sequence information.
    :param template_pdb: Name of the template structure (without file extension).
    :param target_name: Name of the target protein to be modeled.
    :param output_dir: Directory to store output files (default: target_name + "_modeller").
    :param num_models: Number of models to generate (default: 3).
    :return: Path to the output directory containing homology models.
    """

    class MyModel(AutoModel):
        pass  # No special assumptions for now

    env = Environ()
    env.io.atom_files_directory = atom_files_directory

    # Initialize homology modeling
    a = MyModel(env, alnfile=aln_file, knowns=template_pdb, sequence=target_name)
    a.starting_model = 1
    a.ending_model = num_models
    a.make()

    # Define output directory
    output_dir = output_dir or f"{target_name}_modeller"
    os.makedirs(output_dir, exist_ok=True)

    # Organize output files
    for _file in os.listdir():
        if os.path.isfile(_file) and _file.startswith(target_name):
            if _file.endswith('.pdb'):  # Move PDB files
                os.rename(_file, os.path.join(output_dir, f'HM_{_file}'))
            elif _file != aln_file:  # Remove unnecessary files
                os.remove(_file)

    print(f"Output files saved in: {output_dir}")

def write_modeller_fa(tgt_aligned_seq, ref_aligned_seq, 
                      ref_pdb, 
                      labels=None, 
                      output_file="alignment.fa"):
    """
    Writes a Modeller-compatible alignment file (.fa) for homology modeling.

    :param tgt_aligned_seq: Target sequence (string) in aligned format.
    :param ref_aligned_seq: Reference sequence (string) in aligned format.
    :param labels: List containing two labels [target_label, reference_label].
    :param output_file: Name of the output file (default: 'alignment.fa').
    :raises ValueError: If sequences are not the same length.
    """
    if len(tgt_aligned_seq) != len(ref_aligned_seq):
        raise ValueError("Aligned sequences must have the same length.")

    def get_residue_indices(sequence):
        """Get the first and last residue number excluding gaps ('-')."""
        residues = [i + 1 for i, res in enumerate(sequence) if res != '-']
        if residues:
            return residues[0], residues[-1]  # Start, End
        return " ", " "  # Handle case where sequence is all gaps

    # Extract labels
    target_label, ref_label = labels or ['target', 'template']

    # Compute start and end residue positions
    tgt_start, tgt_end = get_residue_indices(tgt_aligned_seq)
    
    # Structure pdb requires the position to line up with pdb. Read from pdb instead
    ref_atoms = read_pdb(ref_pdb)
    ref_start, ref_end = ref_atoms[0]['res_num'], ref_atoms[-1]['res_num']

    fa_content = f""">P1;{target_label}
sequence:{target_label}:{tgt_start}: :{tgt_end}: ::: 0.00: 0.00
{tgt_aligned_seq}*

>P1;{ref_label}.pdb
structureX:{ref_label}.pdb:{ref_start}:A:{ref_end}:A::: 0.00: 0.00
{ref_aligned_seq}*
"""

    with open(output_file, "w") as f:
        f.write(fa_content)

    print(f"Alignment file written to {output_file}")


