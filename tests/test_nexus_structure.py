# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 14:06:56 2024

@author: Maksim Eremenko
"""
import sys
import pytest
from pathlib import Path

src_path = Path(__file__).parent.parent / 'src'
sys.path.append(str(src_path))

from nexus_structure import NeXusStructureFormat
from nexusformat.nexus import nxload
import numpy as np

@pytest.fixture(scope="module")
def create_nexus_file(tmp_path_factory):
    file_path = tmp_path_factory.mktemp("data") / "structure.nxs"
    nexus_structure = NeXusStructureFormat(
        atom_data=np.array([[0,0,0], [0.5,0.5,0.5]]),
        metric_tensor=np.array([[1,0,0], [0,1,0], [0,0,1]]),
        symmetry="m-3m",
        unit_cells=np.array([1,1,1]),
        atom_types=np.array(["H"]),
        flags={"example_flag": True},
        metadata=""
    )
    
    nexus_structure.save_to_nexus(str(file_path))
    return file_path

def test_read_nexus(create_nexus_file):
    file_path = create_nexus_file
    nxfile = nxload(file_path)
    
    assert '/entry/data/atom_data' in nxfile, "Atom data dataset not found in Nexus file"
    assert nxfile['/entry/data/atom_data'].shape == (2, 3), "Atom data shape is incorrect"

