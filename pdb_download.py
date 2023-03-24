

"""
This py script is to download pdb from alphafold website 
"""

import requests
import os 

# Define the URL to access the AlphaFold server
ALPHAFOLD_URL = "https://alphafold.ebi.ac.uk/files/"

def pdb_download(uniprot_ids, DOWNLOAD_PATH):
    
    # Loop over each Uniprot ID and download the corresponding PDB file
    for uniprot_id in uniprot_ids:
    # Construct the URL for the PDB file
        pdb_url = f"{ALPHAFOLD_URL}{uniprot_id}.pdb"

        # Send a GET request to the URL to download the PDB file
        response = requests.get(pdb_url)

        # Create the directory if it does not exist
        os.makedirs(DOWNLOAD_PATH, exist_ok=True)

        # Save the PDB file to the directory
        with open(f"{DOWNLOAD_PATH}{uniprot_id}.pdb", "w") as f:
            f.write(response.text)
    
        print(f"Downloaded {uniprot_id}.pdb")
        