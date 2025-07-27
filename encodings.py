# -*- coding: utf-8 -*-
"""
Created on Sun Jul 27 14:19:06 2025

@author: heba_
"""

import numpy as np
import pandas as pd
from typing import List, Union
import logging
import time

logger = logging.getLogger(__name__)

class BLOSUM62Encoder:
    """BLOSUM62 matrix encoder for protein sequences."""
    
    def __init__(self):
        self.blosum62_matrix = {
            'A': [4,  -1, -2, -2, 0,  -1, -1, 0, -2,  -1, -1, -1, -1, -2, -1, 1,  0,  -3, -2, 0],
            'R': [-1, 5,  0,  -2, -3, 1,  0,  -2, 0,  -3, -2, 2,  -1, -3, -2, -1, -1, -3, -2, -3],
            'N': [-2, 0,  6,  1,  -3, 0,  0,  0,  1,  -3, -3, 0,  -2, -3, -2, 1,  0,  -4, -2, -3],
            'D': [-2, -2, 1,  6,  -3, 0,  2,  -1, -1, -3, -4, -1, -3, -3, -1, 0,  -1, -4, -3, -3],
            'C': [0,  -3, -3, -3, 9,  -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],
            'Q': [-1, 1,  0,  0,  -3, 5,  2,  -2, 0,  -3, -2, 1,  0,  -3, -1, 0,  -1, -2, -1, -2],
            'E': [-1, 0,  0,  2,  -4, 2,  5,  -2, 0,  -3, -3, 1,  -2, -3, -1, 0,  -1, -3, -2, -2],
            'G': [0,  -2, 0,  -1, -3, -2, -2, 6,  -2, -4, -4, -2, -3, -3, -2, 0,  -2, -2, -3, -3],
            'H': [-2, 0,  1,  -1, -3, 0,  0,  -2, 8,  -3, -3, -1, -2, -1, -2, -1, -2, -2, 2,  -3],
            'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4,  2,  -3, 1,  0,  -3, -2, -1, -3, -1, 3],
            'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2,  4,  -2, 2,  0,  -3, -2, -1, -2, -1, 1],
            'K': [-1, 2,  0,  -1, -3, 1,  1,  -2, -1, -3, -2, 5,  -1, -3, -1, 0,  -1, -3, -2, -2],
            'M': [-1, -1, -2, -3, -1, 0,  -2, -3, -2, 1,  2,  -1, 5,  0,  -2, -1, -1, -1, -1, 1],
            'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0,  0,  -3, 0,  6,  -4, -2, -2, 1,  3,  -1],
            'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7,  -1, -1, -4, -3, -2],
            'S': [1,  -1, 1,  0,  -1, 0,  0,  0,  -1, -2, -2, 0,  -1, -2, -1, 4,  1,  -3, -2, -2],
            'T': [0,  -1, 0,  -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1,  5,  -2, -2, 0],
            'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1,  -4, -3, -2, 11, 2,  -3],
            'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2,  -1, -1, -2, -1, 3,  -3, -2, -2, 2,  7,  -1],
            'V': [0,  -3, -3, -3, -1, -2, -2, -3, -3, 3,  1,  -2, 1,  -1, -2, -2, 0,  -3, -1, 4],
            'X': [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            '-': [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        }
        self.encoding_dim = 20

    def encode_sequence(self, sequence: str) -> np.ndarray:
        """Encode a single protein sequence using BLOSUM62."""
        encoding = []
        for aa in sequence:
            if aa.upper() in self.blosum62_matrix:
                encoding.extend(self.blosum62_matrix[aa.upper()])
            else:
                encoding.extend([0] * self.encoding_dim)
                logger.warning(f"Unknown amino acid '{aa}' found, using zero padding")
        
        return np.array(encoding)

    def encode_sequences(self, sequences: List[str]) -> np.ndarray:
        """Encode multiple protein sequences."""
        logger.info(f"Encoding {len(sequences)} sequences using BLOSUM62...")
        start_time = time.time()
        
        encoded = np.array([self.encode_sequence(seq) for seq in sequences])
        
        end_time = time.time()
        logger.info(f"Encoding completed in {end_time - start_time:.2f} seconds")
        
        return encoded

    def encode_dataframe(self, df: pd.DataFrame, sequence_col: str = 'sequence') -> pd.DataFrame:
        """Encode sequences in a DataFrame."""
        result_df = df.copy()
        encoded_sequences = self.encode_sequences(df[sequence_col].tolist())
        
        # Add encoded features as separate columns
        for i in range(encoded_sequences.shape[1]):
            result_df[f'blosum_feature_{i}'] = encoded_sequences[:, i]
        
        # Remove original sequence column
        result_df = result_df.drop(columns=[sequence_col])
        
        return result_df