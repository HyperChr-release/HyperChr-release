
# The following code is adapted from nanopq (https://github.com/matsui528/nanopq)
# Original author: Yusuke Matsui

import numpy as np
from scipy.cluster.vq import kmeans2, vq


def dist_l2(q, x):
    """Compute the squared Euclidean (L2) distance between a query vector and codewords.

    Args:
        q (np.ndarray): Query vector with shape (Ds,).
        x (np.ndarray): Codewords with shape (Ks, Ds).

    Returns:
        np.ndarray: Squared L2 distances with shape (Ks,).
    """
    return np.linalg.norm(q - x, ord=2, axis=1) ** 2


def dist_ip(q, x):
    """Compute the inner product between a query vector and codewords.

    Args:
        q (np.ndarray): Query vector with shape (Ds,).
        x (np.ndarray): Codewords with shape (Ks, Ds).

    Returns:
        np.ndarray: Inner products with shape (Ks,).
    """
    return q @ x.T


def get_Ks(n_row, n_col, M, mem_in_byte, dtype):
    """Compute the maximum number of codewords (Ks) that can fit into the given memory size.

    Args:
        n_row (int): Number of data vectors.
        n_col (int): Dimension of the data vectors (D).
        M (int): Number of subspaces.
        mem_in_byte (int): Available memory size in bytes.
        dtype (numpy.dtype or str): Data type of the codewords (e.g., np.float32 or 'float32').

    Returns:
        int: Maximum number of codewords (Ks) that can fit into the memory size.
    """
    if mem_in_byte <= 0:
        return 0

    cell_bytes = np.dtype(dtype).itemsize

    low, high = 1, n_row*2
    max_Ks = 0
    while low < high - 1:
        mid = (low + high) // 2
        bits_per_code = (mid - 1).bit_length()  # Bits needed to represent Ks codeword
        codewords_memory = mid * n_col * cell_bytes * 8  # Memory for codewords
        codes_memory = n_row * M * bits_per_code  # Memory for codes
        total_memory = (codewords_memory + codes_memory + 7 ) // 8
        if total_memory <= mem_in_byte:
            low = mid
        else:
            high = mid
    max_Ks = low
    return max_Ks


metric_function_map = {"l2": dist_l2, "dot": dist_ip}


class PQ:
    """Pure Python implementation of Product Quantization (PQ) [Jegou11]_.

    In the indexing phase, each input vector of dimension D is divided into M sub-vectors of dimension Ds = D / M.
    Each sub-vector is quantized into one of Ks codewords (centroids) using k-means clustering.
    In the querying phase, given a new query vector, the distance between the query and database PQ-codes
    can be efficiently approximated via Asymmetric Distance Computation.

    All vectors must be np.ndarray with consistent data types (e.g., np.float32).

    .. [Jegou11] H. Jegou et al., "Product Quantization for Nearest Neighbor Search", IEEE TPAMI 2011

    Args:
        M (int): The number of subspaces.
        MemorySize (int): The memory budget in bytes for storing codewords and codes.
        metric (str): The distance metric to use ("l2" or "dot").
            Note that even for 'dot', k-means and encoding are performed in the Euclidean space.
        verbose (bool): Verbose flag for logging.

    Attributes:
        M (int): The number of subspaces.
        Ks (int): The number of codewords for each subspace (computed during fitting).
        MemorySize (int): The memory budget in bytes.
        metric (str): The distance metric used.
        verbose (bool): Verbose flag.
        Ds (int): The dimension of each sub-vector, i.e., Ds = D / M.
        codewords (np.ndarray): Codewords with shape (M, Ks, Ds) and dtype matching the input data.
            codewords[m][ks] is the ks-th codeword for the m-th subspace.
    """

    def __init__(self, M, MemorySize = None, Ks=None, metric="l2", verbose=False):
        assert metric in ["l2", "dot"], "Metric must be 'l2' or 'dot'."
        self.M = M
        self.MemorySize = MemorySize
        self.metric = metric
        self.verbose = verbose

        self.Ks = Ks
        self.codewords = None
        self.Ds = None

        if verbose:
            print(f"M: {M}, MemorySize: {MemorySize}, metric: {metric}")

    def __eq__(self, other):
        if isinstance(other, PQ):
            return (
                self.M == other.M
                and self.Ks == other.Ks
                and self.metric == other.metric
                and self.verbose == other.verbose
                and self.Ds == other.Ds
                and np.array_equal(self.codewords, other.codewords)
            )
        else:
            return False

    def fit(self, vecs, iter=20,seed=123, minit="points"):
        """Run k-means clustering for each subspace to create codewords.

        This function should be run once before encoding vectors.
        It computes the maximum number of codewords (Ks) that can fit into the given memory size.

        Args:
            vecs (np.ndarray): Training vectors with shape (N, D).
            iter (int): The number of iterations for k-means.
            seed (int): The seed for random processes.
            minit (str): Initialization method for k-means centroids ('random', '++', 'points', 'matrix').

        Returns:
            self
        """
        assert vecs.ndim == 2, "Input vectors should be a 2D array."
        N, D = vecs.shape
        InputType = vecs.dtype
        self.Ds = D // self.M
        assert D % self.M == 0, "Input dimension must be divisible by M."

        if(self.Ks==None):
            self.Ks = get_Ks(N, D, self.M, self.MemorySize, InputType)

        assert self.Ks <= N, "The number of training vectors should be more than Ks."
        assert minit in ["random", "++", "points", "matrix"], "Invalid minit method."

        np.random.seed(seed)
        if self.verbose:
            print(f"iter: {iter}, seed: {seed}, Ks: {self.Ks}")

        # Initialize codewords array
        self.codewords = np.zeros((self.M, self.Ks, self.Ds), dtype=InputType)

        # Run k-means for each subspace
        for m in range(self.M):
            if self.verbose:
                print(f"Training subspace: {m + 1} / {self.M}")
            vecs_sub = vecs[:, m * self.Ds : (m + 1) * self.Ds]
            self.codewords[m], _ = kmeans2(
                vecs_sub, self.Ks, iter=iter, minit=minit, seed=seed
            )
        return self

    def encode(self, vecs):
        """Encode input vectors into PQ codes.

        Args:
            vecs (np.ndarray): Input vectors with shape (N, D) and dtype matching the codewords.

        Returns:
            np.ndarray: PQ codes with shape (N, M) 
        """
        assert vecs.dtype == self.codewords.dtype, "Input vectors must have the same dtype as codewords."
        assert vecs.ndim == 2, "Input vectors should be a 2D array."
        N, D = vecs.shape
        assert D == self.Ds * self.M, f"Input dimension must be Ds * M = {self.Ds * self.M}."

        codes = np.empty((N, self.M), dtype=np.int32)
        for m in range(self.M):
            if self.verbose:
                print(f"Encoding subspace: {m + 1} / {self.M}")
            vecs_sub = vecs[:, m * self.Ds : (m + 1) * self.Ds]
            # Assign each sub-vector to the nearest codeword
            codes[:, m], _ = vq(vecs_sub, self.codewords[m])

        return codes

    def decode(self, codes):
        """Reconstruct the original vectors approximately from PQ codes.

        Args:
            codes (np.ndarray): PQ codes with shape (N, M) 

        Returns:
            np.ndarray: Reconstructed vectors with shape (N, D) and dtype matching the codewords.
        """
        assert codes.ndim == 2, "Codes should be a 2D array."
        N, M = codes.shape
        assert M == self.M, f"Codes dimension M should be {self.M}."


        vecs = np.empty((N, self.Ds * self.M), dtype=self.codewords.dtype)
        for m in range(self.M):
            # Retrieve codewords for each code
            vecs[:, m * self.Ds : (m + 1) * self.Ds] = self.codewords[m][codes[:, m], :]

        return vecs
