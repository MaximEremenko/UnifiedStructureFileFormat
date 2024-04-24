# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 14:33:12 2024

@author: Maksim Eremenko
"""

import h5py
import numpy as np

class NeXusStructureFormat:
    def __init__(self, metric_tensor,
                       symmetry,
                       unit_cells,
                       atom_types,
                       number_of_atoms,
                       atom_data,
                       flags=None,
                       metadata=None,
                       average_structure=None,
                       molecular_identity=None,
                       magnetic_spin=None,
                       property_flags=None):
        
        self.set_metric_tensor(metric_tensor)
        self.set_symmetry(symmetry)
        self.set_unit_cells(unit_cells)
        self.set_atom_types(atom_types)
        self.set_number_of_atoms(number_of_atoms)
        self.set_atom_data(atom_data) 
        self.flags = flags
        self.metadata = metadata
        self.average_structure = average_structure
        self.molecular_identity = molecular_identity
        self.magnetic_spin = magnetic_spin
        self.property_flags = property_flags

    def set_metric_tensor(self, metric_tensor):
        if not isinstance(metric_tensor, np.ndarray) or metric_tensor.shape != (3, 3) or metric_tensor.dtype != float:
            raise ValueError("Metric tensor must be a 3x3 matrix of type float.")
        self.metric_tensor = metric_tensor

    def set_symmetry(self, symmetry):
        if not isinstance(symmetry, str) or len(symmetry) > 32:
            raise ValueError("Symmetry must be a string of up to 32 characters.")
        self.symmetry = symmetry

    def set_unit_cells(self, unit_cells):
        if not isinstance(unit_cells, np.ndarray) or unit_cells.shape != (3,) or unit_cells.dtype != np.uint:
            raise ValueError("Unit cells must be an array of 3 unsigned integers.")
        self.unit_cells = unit_cells

    def set_number_of_atoms(self, number_of_atoms):
        if not isinstance(number_of_atoms, int) or number_of_atoms <= 0:
            raise ValueError("Number of atoms must be a positive integer.")
        self.number_of_atoms = number_of_atoms

    def set_atom_types(self, atom_types):
        # Check if atom_types is a list and each element is a tuple of the correct length and type
        if not all(isinstance(t, tuple) and len(t) == 4 for t in atom_types):
            raise ValueError("Each atom type entry must be a tuple with exactly 4 elements.")
    
        # Validate the content of each tuple
        for t in atom_types:
            if not (isinstance(t[0], str) and len(t[0]) <= 2 and
                    isinstance(t[1], int) and isinstance(t[2], int) and isinstance(t[3], int)):
                raise ValueError("Invalid atom type specification.")
        self.atom_types = atom_types
        
    def set_atom_data(self, atom_data):
        # Define the expected dtype for atom data
        atom_dtype = np.dtype([
            ('atom_id', np.int32),
            ('atom_type', np.int32),
            ('x', np.float64),
            ('y', np.float64),
            ('z', np.float64),
            ('uc_x', np.int32),
            ('uc_y', np.int32),
            ('uc_z', np.int32),
            ('uc_site', np.int32)
        ])
        
        # Validate the atom data
        if not isinstance(atom_data, np.ndarray):
            raise ValueError("Atom data must be a numpy array.")
        if atom_data.dtype != atom_dtype:
            raise ValueError("Atom data has incorrect data types or structure.")
        if len(atom_data.shape) != 1:
            raise ValueError("Atom data must be a 1D array of structured data.")

        self.atom_data = atom_data
    def save_to_nexus(self, file_path):
        try:
            with h5py.File(file_path, 'w') as file:
              
                nx_entry = file.create_group('entry')
                nx_entry.attrs['NX_class'] = 'NXentry'
                
                nx_data = nx_entry.create_group('data')
                nx_data.attrs['NX_class'] = 'NXdata'
                
                # Essential data elements
                nx_data.create_dataset('metric_tensor', data=self.metric_tensor)
                nx_data.create_dataset('symmetry', data=np.string_(self.symmetry))
                nx_data.create_dataset('unit_cells', data=self.unit_cells)
                nx_data.create_dataset('number_of_atoms', data=np.array([self.number_of_atoms], dtype=np.uint))
                # Atom types
                dtype = np.dtype([('type', 'S2'), ('atomic_number', np.uint), ('charge', int), ('isotope', np.uint)])
                atom_types_dataset = np.array(self.atom_types, dtype=dtype)
                nx_data.create_dataset('atom_types', data=atom_types_dataset)
                
                # Save the atom data as a dataset
                nx_data.create_dataset('atom_data', data=self.atom_data)
                # Optional and flags processing
                for key, value in self.flags.items():
                    nx_data.attrs['flag_' + key] = value
                #metadata  
                nx_metadata = nx_data.create_group('metadata')
                for key, value in self.metadata.items():
                    nx_metadata.attrs[key] = value
                
                # Optional components handling
                # Save average structure
                if self.average_structure is not None:
                    avg_dtype = np.dtype([
                        ('atom_type', np.int32),
                        ('x', np.float64), ('y', np.float64), ('z', np.float64),
                        ('avg_occupancy', np.float64),
                        ('Uiso', np.float64),
                        ('u_ij', np.float64, (6,))
                    ])
                    nx_data.create_dataset('average_structure', data=np.array(self.average_structure, dtype=avg_dtype))
                
                # Save molecular identity
                if self.molecular_identity is not None:
                    mol_id_dtype = np.dtype([
                        ('molecule_type', np.int32),
                        ('molecule_number', np.int32)
                    ])
                    nx_data.create_dataset('molecular_identity', data=np.array(self.molecular_identity, dtype=mol_id_dtype))
                
                # Save magnetic spin
                if self.magnetic_spin is not None:
                    mag_spin_dtype = np.dtype([
                        ('spin_x', np.float64),
                        ('spin_y', np.float64),
                        ('spin_z', np.float64)
                    ])
                    nx_data.create_dataset('magnetic_spin', data=np.array(self.magnetic_spin, dtype=mag_spin_dtype))
                
                # Save property flags
                if self.property_flags is not None:
                    nx_data.create_dataset('property_flags', data=np.array(self.property_flags, dtype=np.int32))


        except IOError:
            raise IOError("Failed to write to the specified file path.")
    @classmethod
    def load_from_nexus(cls, file_path):
        try:
            with h5py.File(file_path, 'r') as file:
                nx_data = file['entry/data']
                
                # Metric tensor validation
                metric_tensor = nx_data['metric_tensor'][:]
                if metric_tensor.shape != (3, 3) or metric_tensor.dtype != float:
                    raise ValueError("Metric tensor must be a 3x3 matrix of floats.")
                
                # Symmetry validation
                symmetry = nx_data['symmetry'][()].decode('utf-8')
                if not isinstance(symmetry, str) or len(symmetry) > 32:
                    raise ValueError("Symmetry must be a string of up to 32 characters.")
                
                # Unit cells validation
                unit_cells = nx_data['unit_cells'][:]
                if unit_cells.shape != (3,) or unit_cells.dtype != np.uint:
                    raise ValueError("Unit cells must be an array of 3 unsigned integers.")
                
                # Number of atoms validation
                # Load number of atoms, ensuring we extract the first element and convert appropriately
                number_of_atoms_array = nx_data['number_of_atoms'][:]  # Load as array
                if number_of_atoms_array.size == 0:
                    raise ValueError("Number of atoms array is empty.")
                number_of_atoms = int(number_of_atoms_array[0])  # Convert the first element to int
                
                if number_of_atoms <= 0:
                    raise ValueError("Number of atoms must be a positive integer.")

                atom_types_array = nx_data['atom_types'][:]
                atom_types = [(atype[0].decode('utf-8'), int(atype[1]), int(atype[2]), int(atype[3])) for atype in atom_types_array]
                
                # Load the atom data
                atom_data = nx_data['atom_data'][:]
                
                # Flags and metadata handling
                # Load flags that start with 'flag_' and strip the prefix
                flags = {key[5:]: nx_data.attrs[key] for key in nx_data.attrs if key.startswith('flag_')}
                nx_metadata = nx_data['metadata']  # Access the metadata group
                # Load metadata from attributes of the nx_metadata group
                metadata = {key: nx_metadata.attrs[key] for key in nx_metadata.attrs}
                
                # Optional parts loading with checks
                # Load average structure                
                if 'average_structure' in nx_data:
                    average_structure = nx_data['average_structure'][:]
                else:
                    average_structure = None
                
                # Load molecular identity
                if 'molecular_identity' in nx_data:
                    molecular_identity = nx_data['molecular_identity'][:]
                else:
                    molecular_identity = None
                
                # Load magnetic spin
                if 'magnetic_spin' in nx_data:
                    magnetic_spin = nx_data['magnetic_spin'][:]
                else:
                    magnetic_spin = None
                
                # Load property flags
                if 'property_flags' in nx_data:
                    property_flags = nx_data['property_flags'][:]
                else:
                    property_flags = None
                   
                return cls(metric_tensor=metric_tensor, 
                           symmetry=symmetry,
                           unit_cells=unit_cells, 
                           atom_types=atom_types,
                           number_of_atoms=number_of_atoms,
                           atom_data=atom_data,
                           flags=flags,
                           metadata=metadata,
                           average_structure=average_structure,
                           molecular_identity=molecular_identity, 
                           magnetic_spin=magnetic_spin,
                           property_flags=property_flags)
        
        except KeyError as e:
            raise KeyError(f"Expected dataset not found in the file: {e}")
        except Exception as e:
            raise Exception(f"An error occurred while loading from the NeXus file: {e}")

