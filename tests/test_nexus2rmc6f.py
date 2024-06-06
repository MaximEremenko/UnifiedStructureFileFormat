# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 13:28:28 2024

@author: Maksim Eremenko
"""
import sys
from pathlib import Path
import unittest
import numpy as np

src_path = Path(__file__).parent.parent / 'src'
sys.path.append(str(src_path))
src_path = Path(__file__).parent.parent / 'utilities'
sys.path.append(str(src_path))

from rmc6f_reader import RMC6fReader
from nexus_structure import NeXusStructureFormat

from periodictable import elements

class TestRMC6fToNeXus(unittest.TestCase):
    def test_rmc6f_to_nexus_integration(self):
        # Initialize the reader with the actual file path
        reader = RMC6fReader("./data/YSZ_rmc_01.rmc6f")

        # Extract data from the reader
        df = reader.get_data()  # DataFrame containing atom data
        B, B_, G, G_ = reader.vec2spacemat(reader.vectors)  # Assuming these are the lattice vectors
        metric_tensor = G
        atom_types_ = reader.get_atom_types()  # List of atom types
        unit_cells = np.array(reader.get_num_cells(), dtype=np.uint)  # Supercell dimensions
        symmetry = "P1"  
        number_of_atoms = df.shape[0]
        # Create a mapping from element to its index in atom_types
        atom_types = []
        for idx, element in enumerate(atom_types_):
            if (element.lower()!='Va'.lower()):
               atom_types.append((element, elements.symbol(element).number, 0, 0))
               
        else:
             atom_types.append(('Va', 0, 0, 0))
        
        element_to_id = {element: idx for idx, element in enumerate(atom_types_)}
        # Replace the 'element' column using the mapping
        df['element'] = df['element'].map(element_to_id)
        df['id'] = df['id'].apply(lambda x: int(x.strip('[]')) if isinstance(x, str) and x.startswith('[') and x.endswith(']') else x)
        #print("Before mapping, NaN counts in 'element':", df['element'].isna().sum())
        #df['element'].fillna(-1, inplace=True)
        atom_data_dtype = np.dtype([
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
        df_selected_colums = df[["id", "element", "x", "y", "z", "cellRefNumX", "cellRefNumY", "cellRefNumZ", "refNumber"]]
        atom_data = np.array(list(df_selected_colums.to_records(index=False)), dtype=atom_data_dtype)
        
        flags = {'super_structure': True}
        metadata = {'timestamp': '2024-01-25 14:33:12', 'software': 'NeXusTest v1.0', 'test': 'test_'}
        
        # Initialize NeXusStructureFormat with the data from RMC6fReader
        nexus_format = NeXusStructureFormat(
            metric_tensor=metric_tensor,
            symmetry=symmetry,
            unit_cells=unit_cells,
            atom_types=atom_types,
            number_of_atoms=number_of_atoms,
            atom_data=atom_data,
            flags = flags,
            metadata = metadata
        )

        # Save to a NeXus file (assuming method to save is correctly implemented)
        nexus_path = "YSZ_rmc_01_test_nexus_file.nxs"
        nexus_format.save_to_nexus(nexus_path)
 
        # Verify the file is saved (you could extend this by actually loading the file and checking contents)
        self.assertTrue(Path(nexus_path).exists())
        # Load from the NeXus file
        loaded_nexus = NeXusStructureFormat.load_from_nexus(nexus_path)
        # Assert the loaded data matches the saved data
        self.assertTrue(np.allclose(loaded_nexus.metric_tensor, metric_tensor))
        self.assertEqual(loaded_nexus.symmetry, symmetry)
        self.assertTrue(np.array_equal(loaded_nexus.unit_cells, unit_cells))
        self.assertEqual(loaded_nexus.number_of_atoms, number_of_atoms)
        self.assertEqual(loaded_nexus.flags, flags)
        self.assertEqual(loaded_nexus.metadata, metadata)
        
    def tearDown(self):
        # Clean up after tests
        try:
            Path("YSZ_rmc_01_test_nexus_file.nxs").unlink()
        except FileNotFoundError:
            pass
        
        
class TestNeXusToRMC6f(unittest.TestCase):
    def test_rmc6f_to_nexus_integration(self):
        # Initialize the reader with the actual file path
        reader = RMC6fReader("./data/YSZ_rmc_01.rmc6f")

        # Extract data from the reader
        df = reader.get_data()  # DataFrame containing atom data
        B, B_, G, G_ = reader.vec2spacemat(reader.vectors)  # Assuming these are the lattice vectors
        metric_tensor = G
        atom_types_ = reader.get_atom_types()  # List of atom types
        unit_cells = np.array(reader.get_num_cells(), dtype=np.uint)  # Supercell dimensions
        symmetry = "P1"  
        number_of_atoms = df.shape[0]
        # Create a mapping from element to its index in atom_types
        atom_types = []
        for idx, element in enumerate(atom_types_):
            if (element.lower()!='Va'.lower()):
               atom_types.append((element, elements.symbol(element).number, 0, 0))
               
        else:
             atom_types.append(('Va', 0, 0, 0))
        
        element_to_id = {element: idx for idx, element in enumerate(atom_types_)}
        # Replace the 'element' column using the mapping
        df['element'] = df['element'].map(element_to_id)
        df['id'] = df['id'].apply(lambda x: int(x.strip('[]')) if isinstance(x, str) and x.startswith('[') and x.endswith(']') else x)
        #print("Before mapping, NaN counts in 'element':", df['element'].isna().sum())
        #df['element'].fillna(-1, inplace=True)
        atom_data_dtype = np.dtype([
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
        df_selected_colums = df[["id", "element", "x", "y", "z", "cellRefNumX", "cellRefNumY", "cellRefNumZ", "refNumber"]]
        atom_data = np.array(list(df_selected_colums.to_records(index=False)), dtype=atom_data_dtype)
        
        flags = {'super_structure': True}
        metadata = {'timestamp': '2024-01-25 14:33:12', 'software': 'NeXusTest v1.0', 'test': 'test_'}
        
        # Initialize NeXusStructureFormat with the data from RMC6fReader
        nexus_format = NeXusStructureFormat(
            metric_tensor=metric_tensor,
            symmetry=symmetry,
            unit_cells=unit_cells,
            atom_types=atom_types,
            number_of_atoms=number_of_atoms,
            atom_data=atom_data,
            flags = flags,
            metadata = metadata
        )

        # Save to a NeXus file (assuming method to save is correctly implemented)
        nexus_path = "YSZ_rmc_01_test_nexus_file.nxs"
        nexus_format.save_to_nexus(nexus_path)
 
        # Verify the file is saved (you could extend this by actually loading the file and checking contents)
        self.assertTrue(Path(nexus_path).exists())
        # Load from the NeXus file
        loaded_nexus = NeXusStructureFormat.load_from_nexus(nexus_path)
        # Assert the loaded data matches the saved data
        self.assertTrue(np.allclose(loaded_nexus.metric_tensor, metric_tensor))
        self.assertEqual(loaded_nexus.symmetry, symmetry)
        self.assertTrue(np.array_equal(loaded_nexus.unit_cells, unit_cells))
        self.assertEqual(loaded_nexus.number_of_atoms, number_of_atoms)
        self.assertEqual(loaded_nexus.flags, flags)
        self.assertEqual(loaded_nexus.metadata, metadata)
        
    # def tearDown(self):
    #     # Clean up after tests
    #     try:
    #         Path("YSZ_rmc_01_test_nexus_file.nxs").unlink()
    #     except FileNotFoundError:
    #         pass

if __name__ == '__main__':
    unittest.main()