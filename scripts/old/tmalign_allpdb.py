

"""
Since tmalign takes a significantly longer time compared to kabasch. Use this script align and save files overnight 
"""


import numpy as np 
import pandas as pd 
import os 
import importlib
import copy

from tmtools.io import get_structure
from tmtools import tm_align
import rmsd_functions as rmsd 
importlib.reload(rmsd)
import pickle


PDB_PATH = '../AF_files/AF_pdb/'
pdb_files = os.listdir(PDB_PATH)


print("Loading pdb files")
data = {}
for file in pdb_files: 
    name_list = []
    name = file.split('.')[0]
#     If there is multiple Olfr names in pdb name, create seperate entry for them in data 
    for i in name.split('_'):
        if 'Olfr' in i: 
            name_list.append(i)
    for Or in name_list:
        try:
            atom, coord, resid, aa = rmsd.get_coordinates_pdb(PDB_PATH+file)   
            data[Or] = {}
            data[Or]["atom"] = atom
            data[Or]["coord"] = coord
            data[Or]["resid"] = resid
            data[Or]['amino_acid'] = aa
        except:
            print(file + ' NOT read')
print("{} Olfr loaded".format(len(data)))



backbone = copy.deepcopy(data)
backbone = rmsd.get_backbone(backbone)


print("Quality Control: Dropping small Olfr lesser than 285 resid")
small_Olfr = []
for Olfr in backbone:
    if backbone[Olfr]['resid'].max() < 285:
        small_Olfr.append(Olfr)
for Olfr in small_Olfr: 
    data.pop(Olfr)
    backbone.pop(Olfr)
print("{} considerred short \n{} Olfr left".format(len(small_Olfr), len(data)-len(small_Olfr)))


print("Begin tmalign rotation ")
aligned_dict = rmsd.tmalign_data(backbone, align="Olfr1377", rotate_data = data)
print("Rotation alignment done")
f=open('../AF_files/dict_tmaligned.pkl','wb')
pickle.dump(aligned_dict,f)
f.close
print("Files saved")





