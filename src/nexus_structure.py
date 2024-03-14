# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 14:33:12 2024

@author: Maksim Eremenko
"""

import h5py
import numpy as np
from nexusformat.nexus import *

class NeXusStructureFormat:
    def __init__(self, atom_data, metric_tensor, symmetry, unit_cells, atom_types, flags, metadata):
        self.atom_data = atom_data  # This should be a structured array or similar structure
        self.metric_tensor = metric_tensor
        self.symmetry = symmetry
        self.unit_cells = unit_cells
        self.atom_types = atom_types
        self.flags = flags  # Expected to be a dictionary
        self.metadata = metadata

    def save_to_nexus(self, file_path):
        with h5py.File(file_path, 'w') as file:
            file.attrs['default'] = 'entry'
            nx_entry = file.create_group('entry')
            nx_entry.attrs['NX_class'] = 'NXentry'
            nx_entry.attrs['default'] = 'data'
    
            nx_data = nx_entry.create_group('data')
            nx_data.attrs['NX_class'] = 'NXdata'
            
            nx_data.create_dataset('atom_data', data=self.atom_data)
            nx_data.create_dataset('metric_tensor', data=self.metric_tensor)
            nx_data.create_dataset('symmetry', data=self.symmetry)
            nx_data.create_dataset('unit_cells', data=self.unit_cells)
    
            # Option 2: Using variable-length strings for atom_types
            atom_types_vlen = np.array(self.atom_types, dtype=h5py.special_dtype(vlen=str))
            nx_data.create_dataset('atom_types', data=atom_types_vlen)
    
            if isinstance(self.flags, dict):
                for flag, value in self.flags.items():
                    nx_data.attrs[flag] = value