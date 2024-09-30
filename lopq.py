# MIT License
# Copyright (c) 2024 Wang Li, Peking University

import numpy as np
from .opq import OPQ
from scipy.cluster.vq import kmeans2

def get_Ks(n_row, n_col, M, mem_in_byte, local, dtype):
    """
    Calculate the optimal number of codewords (Ks) for LOPQ, considering memory constraints.

    Args:
        n_row (int): Number of data vectors.
        n_col (int): Dimensionality of each vector.
        M (int): Number of subspaces.
        mem_in_byte (int): Available memory in bytes.
        local (int): Number of local clusters.
        dtype (str): Data type of the vectors, e.g., 'float32' or 'float16'.

    Returns:
        int: Optimal number of codewords (Ks) or 0 if memory constraints are not satisfied.
    """
    
    cell_bytes = np.dtype(dtype).itemsize

    if mem_in_byte <= 0:
        return 0
    cell_bytes = np.dtype(dtype).itemsize

    if local < 1:
        print("Error: Number of local clusters must be at least 1.")
        return 0

    low, high = 0, n_row*2
    max_Ks = 0

    while low < high - 1:
        mid = (low + high) // 2
        # Memory calculation in bits for each component
        split_imap_mem = n_row * (local - 1).bit_length() if local > 1 else 0
        R_matrix_mem = n_col * n_col * cell_bytes * 8 * local  # Rotation matrix memory usage
        codewords_mem = mid * n_col * cell_bytes * 8 * local  # Codewords memory usage
        codes_mem =  n_row * M * (mid - 1).bit_length()  # Encoded vectors memory usage
        # Total memory usage in bytes
        mem_usage = (split_imap_mem + R_matrix_mem + codewords_mem + codes_mem + 7) // 8
        if mem_usage <= mem_in_byte:
            low = mid
        else:
            high = mid
    max_Ks = low
    return max_Ks


class LOPQ(object):
    """
    Python implementation of Local Optimized Product Quantization (LOPQ).

    Args:
        M (int): Number of subspaces.
        MemorySize (int): Memory size constraint in bytes.
        Ks (int): Number of codewords for each subspace.
        local (int): Number of local clusters.
        metric (str): Distance metric for k-means ('l2' by default).
        verbose (bool): Flag to control verbosity.

    Attributes:
        opq (list): List of OPQ instances.
        split (np.ndarray): Array indicating cluster assignments for vectors.
    """

    def __init__(self, M=8, MemorySize=1, Ks=None, local=2, metric="l2", verbose=False):
        self.M = M
        self.MemorySize = MemorySize
        self.local = local
        self.Ks = Ks
        self.metric = metric
        self.verbose = verbose
        
        self.opq = []
        self.split = None
        self.kmeans = None

    def __eq__(self, other):
        if isinstance(other, LOPQ):
            return (
                self.M == other.M and
                self.Ks == other.Ks and
                self.metric == other.metric and
                self.verbose == other.verbose and
                self.local == other.local and
                self.opq == other.opq
            )
        else:
            return False


    @property
    def codewords(self):
        return self.opq[0].codewords

    
    @property
    def Ds(self):
        return self.opq[0].Ds

    def fit(self, vecs, parametric_init=False, pq_iter=20, rotation_iter=10, seed=123, minit="points", memory=128):
        """
        Train the LOPQ model.

        Args:
            vecs (np.ndarray): Training vectors of shape (N, D).
            parametric_init (bool): Whether to use parametric initialization for rotation.
            pq_iter (int): Number of k-means iterations for PQ learning.
            rotation_iter (int): Number of iterations for learning the rotation matrix.
            seed (int): Random seed for reproducibility.
            minit (str): Method for k-means initialization ('points' by default).
            memory (int): Available memory in kilobytes.

        Returns:
            LOPQ: Trained LOPQ instance.
        """
        assert vecs.ndim == 2
        N, D = vecs.shape
        InputType = vecs.dtype

        if self.Ks is None:
            self.Ks = get_Ks(N, D, self.M, self.MemorySize, self.local, InputType)

        self.opq = [OPQ(M=self.M, Ks=self.Ks, metric=self.metric, verbose=self.verbose) for _ in range(self.local)]
        self.kmeans, self.split = kmeans2(vecs, self.local, iter=pq_iter, minit=minit)
       
        for i in range(self.local):
            self.opq[i] = OPQ(M=self.M, Ks=min(self.Ks, np.sum(self.split == i)), verbose=False)
            self.opq[i].fit(vecs[self.split == i, :], parametric_init, pq_iter, rotation_iter, seed, minit)
        return self

    def encode(self, vecs):
        """
        Encode vectors using the trained LOPQ model.

        Args:
            vecs (np.ndarray): Input vectors of shape (N, D).

        Returns:
            np.ndarray: Encoded PQ codes of shape (N, M).
        """
        assert vecs.dtype == self.codewords.dtype
        N, D = vecs.shape
        codes = np.empty((N, self.M), dtype=np.int32)
        for i in range(self.local):
            local_codes = self.opq[i].encode(vecs[self.split == i, :])
            codes[self.split == i, :] = local_codes
        return codes

    def decode(self, codes):
        """
        Decode PQ codes back to the original vector space.

        Args:
            codes (np.ndarray): PQ codes of shape (N, M).

        Returns:
            np.ndarray: Reconstructed vectors of shape (N, D).
        """
        N, M = codes.shape
        vecs = np.empty((N, self.Ds * self.M), dtype=self.codewords.dtype)
        for i in range(self.local):
            vecs[self.split == i, :] = self.opq[i].decode(codes[self.split == i, :])
        return vecs
