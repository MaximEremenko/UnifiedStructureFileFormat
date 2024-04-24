# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 14:06:56 2024

@author: Maksim Eremenko
"""
import sys

from pathlib import Path

src_path = Path(__file__).parent.parent / 'src'
sys.path.append(str(src_path))

import unittest
import numpy as np
import os
from nexus_structure import NeXusStructureFormat
from nexusformat.nexus import nxload



class TestNeXusStructureFormatNeXusLibRead(unittest.TestCase):

    def setUp(self):
        self.file_path = "test_nexus_file.hdf5"
        self.metric_tensor = np.array([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]], dtype=float)
        self.symmetry = "P2/m"
        self.unit_cells = np.array([10, 10, 10], dtype=np.uint)
        self.atom_types = [
            ('H', 1, 1, 0),
            ('C', 6, 0, 0)
        ]
        self.number_of_atoms = 2
        
        # Define atom_data with the required structure
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
        self.atom_data = np.array([
            (1, 1, 0.1, 0.2, 0.3, 0, 0, 0, 1),
            (2, 2, 0.4, 0.5, 0.6, 1, 1, 1, 2)
        ], dtype=atom_data_dtype)

        self.flags = {'super_structure': True}
        self.metadata = {'timestamp': '2024-01-25 14:33:12', 'software': 'NeXusTest v1.0', 'test': 'test_good'}
        # Setup the NeXusStructureFormat instance
        self.nexus = NeXusStructureFormat(
            metric_tensor=self.metric_tensor,
            symmetry=self.symmetry,
            unit_cells=self.unit_cells,
            atom_types=self.atom_types,
            number_of_atoms=self.number_of_atoms,
            atom_data=self.atom_data,
            flags=self.flags,
            metadata=self.metadata
        )

    def test_save_and_load(self):
        self.nexus.save_to_nexus(self.file_path)
        loaded_nexus = nxload(self.file_path)
        print(loaded_nexus.tree)
    def tearDown(self):
        # Clean up the file after tests to avoid clutter
        if os.path.exists(self.file_path):
            os.remove(self.file_path) 
class TestNeXusStructureFormat(unittest.TestCase):

    def setUp(self):
        self.file_path = "test_nexus_file.hdf5"
        self.metric_tensor = np.array([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]], dtype=float)
        self.symmetry = "P2/m"
        self.unit_cells = np.array([10, 10, 10], dtype=np.uint)
        self.atom_types = [
            ('H', 1, 1, 0),
            ('C', 6, 0, 0)
        ]
        self.number_of_atoms = 2
        
        # Define atom_data with the required structure
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
        self.atom_data = np.array([
            (1, 1, 0.1, 0.2, 0.3, 0, 0, 0, 1),
            (2, 2, 0.4, 0.5, 0.6, 1, 1, 1, 2)
        ], dtype=atom_data_dtype)

        self.flags = {'super_structure': True}
        self.metadata = {'timestamp': '2024-01-25 14:33:12', 'software': 'NeXusTest v1.0', 'test': 'test_good'}
        # Setup the NeXusStructureFormat instance
        self.nexus = NeXusStructureFormat(
            metric_tensor=self.metric_tensor,
            symmetry=self.symmetry,
            unit_cells=self.unit_cells,
            atom_types=self.atom_types,
            number_of_atoms=self.number_of_atoms,
            atom_data=self.atom_data,
            flags=self.flags,
            metadata=self.metadata
        )

    def test_save_and_load(self):
        # Save the NeXus file
        self.nexus.save_to_nexus(self.file_path)
        
        # Load from the NeXus file
        loaded_nexus = NeXusStructureFormat.load_from_nexus(self.file_path)

        # Assert the loaded data matches the saved data
        self.assertTrue(np.allclose(loaded_nexus.metric_tensor, self.metric_tensor))
        self.assertEqual(loaded_nexus.symmetry, self.symmetry)
        self.assertTrue(np.array_equal(loaded_nexus.unit_cells, self.unit_cells))
        self.assertEqual(loaded_nexus.number_of_atoms, self.number_of_atoms)
        self.assertEqual(loaded_nexus.flags, self.flags)
        self.assertEqual(loaded_nexus.metadata, self.metadata)
    def tearDown(self):
        # Clean up the file after tests to avoid clutter
        if os.path.exists(self.file_path):
            os.remove(self.file_path)


class TestNeXusStructureFormatAllParameters(unittest.TestCase):

    def setUp(self):
        self.file_path = "test_nexus_file.hdf5"
        self.metric_tensor = np.array([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]], dtype=float)
        self.symmetry = "P2/m"
        self.unit_cells = np.array([10, 10, 10], dtype=np.uint)
        self.atom_types = [
            ('H', 1, 1, 0),
            ('C', 6, 0, 0)
        ]
        self.number_of_atoms = 2
        
        # Define atom_data with the required structure
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
        self.atom_data = np.array([
            (1, 1, 0.1, 0.2, 0.3, 0, 0, 0, 1),
            (2, 2, 0.4, 0.5, 0.6, 1, 1, 1, 2)
        ], dtype=atom_data_dtype)

        self.flags = {'super_structure': True}
        self.metadata = {'timestamp': '2024-01-25 14:33:12', 'software': 'NeXusTest v1.0', 'test': 'test_good'}
        # Setup the NeXusStructureFormat instance
        # Define average structure
        avg_structure_dtype = np.dtype([
            ('atom_type', np.int32),
            ('x', np.float64), ('y', np.float64), ('z', np.float64),
            ('avg_occupancy', np.float64),
            ('Uiso', np.float64),
            ('u_ij', np.float64, (6,))
        ])
        self.average_structure = np.array([
            (1, 0.1, 0.2, 0.3, 0.99, 0.05, (0.01, 0.02, 0.01, 0.02, 0.03, 0.04)),
            (2, 0.4, 0.5, 0.6, 0.95, 0.10, (0.02, 0.01, 0.03, 0.04, 0.05, 0.06))
        ], dtype=avg_structure_dtype)
    
        # Define molecular identity
        mol_identity_dtype = np.dtype([
            ('molecule_type', np.int32),
            ('molecule_number', np.int32)
        ])
        self.molecular_identity = np.array([
            (1, 100),
            (2, 200)
        ], dtype=mol_identity_dtype)
    
        # Define magnetic spin
        mag_spin_dtype = np.dtype([
            ('spin_x', np.float64),
            ('spin_y', np.float64),
            ('spin_z', np.float64)
        ])
        self.magnetic_spin = np.array([
            (0.5, -0.5, 1.0),
            (-0.5, 0.5, -1.0)
        ], dtype=mag_spin_dtype)
    
        # Define property flags
        self.property_flags = np.array([1, 2], dtype=np.int32)
    
        # Reinitialize instance with new fields
        self.nexus = NeXusStructureFormat(
            metric_tensor=self.metric_tensor,
            symmetry=self.symmetry,
            unit_cells=self.unit_cells,
            atom_types=self.atom_types,
            number_of_atoms=self.number_of_atoms,
            atom_data=self.atom_data,
            flags=self.flags,
            metadata=self.metadata,
            average_structure=self.average_structure,
            molecular_identity=self.molecular_identity,
            magnetic_spin=self.magnetic_spin,
            property_flags=self.property_flags
        )

    def test_save_and_load(self):
        # Save the NeXus file
        self.nexus.save_to_nexus(self.file_path)
        
        # Load from the NeXus file
        loaded_nexus = NeXusStructureFormat.load_from_nexus(self.file_path)

        # Assert the loaded data matches the saved data
        self.assertTrue(np.allclose(loaded_nexus.metric_tensor, self.metric_tensor))
        self.assertEqual(loaded_nexus.symmetry, self.symmetry)
        self.assertTrue(np.array_equal(loaded_nexus.unit_cells, self.unit_cells))
        self.assertEqual(loaded_nexus.number_of_atoms, self.number_of_atoms)
        self.assertEqual(loaded_nexus.flags, self.flags)
        self.assertEqual(loaded_nexus.metadata, self.metadata)
        
        # Asserts for new structured data
        loaded_avg_structure = loaded_nexus.average_structure
        np.testing.assert_allclose(loaded_avg_structure['avg_occupancy'], self.average_structure['avg_occupancy'])
        np.testing.assert_allclose(loaded_avg_structure['Uiso'], self.average_structure['Uiso'])
        np.testing.assert_allclose(loaded_avg_structure['u_ij'], self.average_structure['u_ij'])
    
        loaded_molecular_identity = loaded_nexus.molecular_identity
        np.testing.assert_array_equal(loaded_molecular_identity['molecule_type'], self.molecular_identity['molecule_type'])
        np.testing.assert_array_equal(loaded_molecular_identity['molecule_number'], self.molecular_identity['molecule_number'])
    
        loaded_magnetic_spin = loaded_nexus.magnetic_spin
        np.testing.assert_allclose(loaded_magnetic_spin['spin_x'], self.magnetic_spin['spin_x'])
        np.testing.assert_allclose(loaded_magnetic_spin['spin_y'], self.magnetic_spin['spin_y'])
        np.testing.assert_allclose(loaded_magnetic_spin['spin_z'], self.magnetic_spin['spin_z'])
    
        loaded_property_flags = loaded_nexus.property_flags
        np.testing.assert_array_equal(loaded_property_flags, self.property_flags)
    def tearDown(self):
        # Clean up the file after tests to avoid clutter
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

if __name__ == '__main__':
    unittest.main()



