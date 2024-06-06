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
        B, B_, G, G_ = reader.vec2spacemat(reader.vectors)
        metric_tensor = G
        atom_types_ = reader.get_atom_types()  # List of atom types
        unit_cells = np.array(reader.get_num_cells(), dtype=np.uint)  # Supercell dimensions
        symmetry = "P1"  
        number_of_atoms = df.shape[0]

        atom_types = []
        for idx, element in enumerate(atom_types_):
            if (element.lower()!='Va'.lower()):
               atom_types.append((element, elements.symbol(element).number, 0, 0))
               
        else:
             atom_types.append(('Va', 0, 0, 0))
        
        element_to_id = {element: idx for idx, element in enumerate(atom_types_)}

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

        # Save to a NeXus file 
        nexus_path = "YSZ_rmc_01_test_nexus_file.nxs"
        nexus_format.save_to_nexus(nexus_path)
 
        # Verify the file is saved
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
    
    def B_to_parameters(self, B):
        # Extract vectors from the matrix B
        av = B[:, 0]
        bv = B[:, 1]
        cv = B[:, 2]
    
        # Compute the lengths of the vectors
        a = np.linalg.norm(av)
        b = np.linalg.norm(bv)
        c = np.linalg.norm(cv)
    
        # Compute the angles in radians
        alpha = np.arccos(np.dot(bv, cv) / (b * c))
        beta = np.arccos(np.dot(av, cv) / (a * c))
        gamma = np.arccos(np.dot(av, bv) / (a * b))
    
        # Convert angles to degrees
        alpha = np.degrees(alpha)
        beta = np.degrees(beta)
        gamma = np.degrees(gamma)

        return a, b, c, alpha, beta, gamma
    
    def calculate_number_density(self, vectors, number_of_atoms):
        # Calculate the volume of the unit cell
        av = vectors[:, 0]
        bv = vectors[:, 1]
        cv = vectors[:, 2]
        volume = np.abs(np.dot(av, np.cross(bv, cv)))
        
        # Calculate the number density
        number_density = number_of_atoms / volume
        return number_density
    
    def test_rmc6f_to_nexus_integration(self):
        reader = RMC6fReader("./data/YSZ_rmc_01.rmc6f")

        # Extract data from the reader
        df = reader.get_data()  # DataFrame containing atom data
        B, B_, G, G_ = reader.vec2spacemat(reader.vectors) 
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

        # Save to a NeXus file
        nexus_path = "YSZ_rmc_01_test_nexus_file.nxs"
        nexus_format.save_to_nexus(nexus_path)
 
        # Verify the file is saved 
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
        

        output_filename = "./data/YSZ_rmc_01_out.rmc6f"
        # Write the headers and metadata
        with open(output_filename, 'w') as f:
            # Write lattice vectors
            metric_tensor = np.sqrt(loaded_nexus.metric_tensor)
            
            a, b, c, alpha, beta, gamma = self.B_to_parameters(metric_tensor)
            

            f.write("(Version 6f format configuration file)\n")
            f.write(f"Metadata owner:     {loaded_nexus.metadata.get('owner', 'NEXUS')}\n")
            f.write(f"Metadata date:     {loaded_nexus.metadata.get('date', '202x-0x-0x')}\n")
            f.write(f"Metadata material: {loaded_nexus.metadata.get('material', 'Unknown Material')}\n")
            f.write(f"Metadata comment:  {loaded_nexus.metadata.get('comment', 'Test')}\n")
            f.write(f"Metadata source:   {loaded_nexus.metadata.get('source', 'Unknown Source')}\n")

       
            # Write atom types and counts
            atom_types = loaded_nexus.atom_types
            atom_counts = {atom_type[0]: 0 for atom_type in atom_types}
            for atom in loaded_nexus.atom_data:
                atom_counts[atom_types[atom['atom_type']][0]] += 1
            total_atoms = 0
            total_atoms_excluding_va = 0
            for atom in loaded_nexus.atom_data:
                atom_type = atom_types[atom['atom_type']][0]
                atom_counts[atom_type] += 1
                total_atoms += 1
                if atom_type.lower() != 'va':
                    total_atoms_excluding_va += 1
                    
            
            number_density = self.calculate_number_density(metric_tensor, total_atoms_excluding_va)           
            
            f.write(f"Number of types of atoms:           {len(atom_types)}\n")
            f.write(f"Atom types present:                 {'  '.join([atom[0] for atom in atom_types])}\n")
            f.write(f"Number of each atom type:           {'  '.join([str(atom_counts[atom[0]]) for atom in atom_types])}\n")
            
            # Additional metadata
            f.write(f"Number of moves generated:           {loaded_nexus.metadata.get('moves_generated', 0)}\n")
            f.write(f"Number of moves tried:               {loaded_nexus.metadata.get('moves_tried', 0)}\n")
            f.write(f"Number of moves accepted:            {loaded_nexus.metadata.get('moves_accepted', 0)}\n")
            f.write(f"Accumulated time (s) in running loop: {loaded_nexus.metadata.get('accumulated_time', 0.00)}\n")
            f.write(f"Number of prior configuration saves: {loaded_nexus.metadata.get('prior_saves', 0)}\n")
            f.write(f"Number of atoms:                     {loaded_nexus.number_of_atoms}\n")
            
            f.write(f"Number density (Ang^-3):             {number_density:.6f}\n")
       
            # Write supercell dimensions
            supercell_dims = loaded_nexus.unit_cells
            f.write(f"Supercell dimensions:                {'  '.join(map(str, supercell_dims))}\n")
            

            # Write the cell parameters
            f.write(f"Cell (Ang/deg):    {a:.6f}   {b:.6f}   {c:.6f}   {alpha:.6f}   {beta:.6f}   {gamma:.6f}\n")
           
            f.write("Lattice vectors (Ang):\n")
            for vec in metric_tensor:
                f.write(f"   {'   '.join(map(str, vec))}\n")
       
            # Write atom data
            f.write("Atoms:\n")
            for idx, atom in enumerate(loaded_nexus.atom_data):
                atom_type_str = atom_types[atom['atom_type']][0]
                f.write(f"     {idx + 1}   {atom_type_str} [{atom['atom_id']}] {atom['x']: .12f} {atom['y']: .12f} {atom['z']: .12f}  {atom['uc_site']}   {atom['uc_x']}   {atom['uc_y']}   {atom['uc_z']}\n")

       
       
        
       
if __name__ == '__main__':
    unittest.main()