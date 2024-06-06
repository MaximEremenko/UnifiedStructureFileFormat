# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 14:15:28 2023

@author: Maksim Eremenko
"""

import pandas as pd
import numpy as np
import warnings 

class RMC6fReader:
    def __init__(self, filename):
        self.skiprows = 0
        self.notfound = True
        self.filename = filename
        
        if self.notfound:
            self.skiprows = self.find_header_line()           
        while self.notfound:
            try:
                df = pd.read_table(filename, skiprows=self.skiprows, nrows=151,
                                   header=None, delim_whitespace=True, engine='python')
                self.notfound = False
            except:
                self.skiprows = self.skiprows + 1
                if (self.skiprows > 151):
                    break
                continue

        self.df = pd.read_table(filename, skiprows=self.skiprows,
                           header=None, delim_whitespace=True, engine='python')
        if (self.df.shape[1] == 10):
            self.df = self.df.set_axis(['atomNumber', 'element', 'id', 'x', 'y', 'z', 'refNumber', 'cellRefNumX', 'cellRefNumY','cellRefNumZ' ], axis=1)
        elif (self.df.shape[1] == 9):
            self.df = self.df.set_axis(['atomNumber', 'element',  'x', 'y', 'z', 'refNumber', 'cellRefNumX', 'cellRefNumY','cellRefNumZ' ], axis=1)
        else:
            warnings.warn("Unsupported rmc6f format") 

        self.get_header()
        self.get_num_cells()
        a, b, c, alpha, beta, gamma = self.get_vectors()
        self.vectors = self.cell2vec(a, b, c, alpha, beta, gamma)
        self.vectors_pc = self.cell2vec(a/self.supercell[0], b/self.supercell[1], c/self.supercell[2], alpha, beta, gamma)
        
    def find_header_line(self):
        with open(self.filename, 'r') as file:
            for i, line in enumerate(file):
                if "Atoms:" in line:
                    return i
                if i >= 150:  # Only read the first 151 lines
                    break
        return None  # Return None if "Atoms:" not found within the first 151 lines

    
    def cell2vec(self, a, b, c, alpha, beta, gamma):
        vectors = np.zeros((3, 3))
        vectors[2, 0] = a*np.cos(beta)
        vectors[0, 1] = 0.0
        vectors[1, 1] = b*np.sin(alpha)
        vectors[2, 1] = b*np.cos(alpha)
        vectors[0, 2] = 0.0
        vectors[1, 2] = 0.0
        vectors[2, 2] = c
        vectors[1, 0] = (a*b*np.cos(gamma) - vectors[2, 0]
                         * vectors[2, 1])/vectors[1, 1]
        vectors[0, 0] = np.sqrt(a**2 - vectors[1, 0]**2 - vectors[2, 0]**2)
        vectors = np.round(vectors.T,10)
        return vectors
    
    def vec2spacemat(self, vectors):
        av = vectors[:, 0]
        bv = vectors[:, 1]
        cv = vectors[:, 2]

        # basis vectors of reciprocal space
        a_ = np.cross(bv, cv)/(np.dot(av, np.cross(bv, cv)))
        b_ = np.cross(cv, av)/(np.dot(av, np.cross(bv, cv)))
        c_ = np.cross(av, bv)/(np.dot(av, np.cross(bv, cv)))

        B_ = np.array([a_, b_, c_])
        B = np.linalg.inv(B_)
        G_ = np.matmul(B_.T, B_)
        G = np.matmul(B.T, B)
        return B, B_, G, G_
    
    def is_float(self, string):
        try:
            float(string)
            return True
        except ValueError:
            return False    
    
    def get_header(self):
        self.rmc_header = {}
        with open(self.filename) as f:
            for i, line in enumerate(f):
               self.rmc_header[i] = line
               if i >= ( self.skiprows-1):
                   break
            f.close()
            
    def get_num_cells(self):
        search_key  = "Supercell"
        res = [val for key, val in self.rmc_header.items() if search_key in self.rmc_header[key]]
        self.supercell = np.array([int(i) for i in res[0].split() if i.isdigit()])
        return self.supercell
      


    def get_vectors(self):
        search_key  = "Cell"
        res = [val for key, val in self.rmc_header.items() if search_key in self.rmc_header[key]]
        self.cell_param = np.array([i for i in res[0].split() if self.is_float(i)], dtype='f')
        try:
            a, b, c, alpha, beta, gamma = self.cell_param
            alpha = alpha * np.pi/180
            beta = beta * np.pi/180
            gamma = gamma * np.pi/180
            
            return a, b, c, alpha, beta, gamma
        except ValueError:
            print("Oops! cell_param contains !6 numbers")
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        
    
    def get_metric(self, vectors):
        return self.vec2spacemat(vectors)
    
    def get_metadata_owner(self):
        search_key  = "Metadata owner:"
        res = [val for key, val in self.rmc_header.items() if search_key in self.rmc_header[key]]
        return res[0].replace(search_key, '').replace('\x00', '').replace('\n', '')
        
    def get_metadata_date(self):
        search_key  = "Metadata date:"
        res = [val for key, val in self.rmc_header.items() if search_key in self.rmc_header[key]]
        return res[0].replace(search_key, '').replace('\x00', '').replace('\n', '')
    
    def get_data(self): 
        return self.df
    
    def get_atom_types (self):
        search_key  = "Atom types present:"
        res = [val for key, val in self.rmc_header.items() if search_key in self.rmc_header[key]]
        res = res[0].replace(search_key, '').replace('\x00', '').replace('\n', '')
        self.atom_types  = res.split()
        return  self.atom_types
        
    def make_average_structure(self):
        df_average = self.df.copy(deep=True)
        elements = df_average['element'].unique()
        elements_refn = df_average['refNumber'].unique()

        for element in elements:
            for element_refn in elements_refn:
                element_indices = (df_average['element'] == element) & (df_average['refNumber'] == element_refn)

                if not df_average.loc[element_indices, ['x', 'y', 'z']].empty:
                    divided_values = df_average.loc[element_indices, ['cellRefNumX', 'cellRefNumY', 'cellRefNumZ']] / self.get_num_cells()[0:3]
                    delta = df_average.loc[element_indices, ['x', 'y', 'z']] - divided_values.to_numpy()
                    delta = delta.apply(lambda col: col.map(lambda x: x + 1 if x < -0.5 else x - 1 if x > 0.5 else x))
                    
                    avg_values = np.average(delta, axis=0)
                    
                    df_average.loc[element_indices, ['x', 'y', 'z']] = divided_values.to_numpy() + avg_values

                    df_average.loc[element_indices, ['x', 'y', 'z']] = df_average.loc[element_indices, ['x', 'y', 'z']].apply(
                        lambda col: col.map(lambda x: x + 1 if x < 0 else x - 1 if x > 1 else x))

        self.df_average = df_average
        return self.df_average
