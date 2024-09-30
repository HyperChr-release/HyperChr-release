

import numpy as np
from scipy.cluster.vq import kmeans2, vq
import math
from rtn import rtn_quant,rtn_invquant

class HPQ_Q_A:
    def __init__(self, num_groups, compression_rate, n_sub, pq_iter):
        self.num_groups = num_groups
        self.compression_rate = compression_rate
        self.n_sub = n_sub
        self.pq_iter = pq_iter
        self.is_fitted = False

    def transform_data(self, data):
        return data.reshape(-1, self.n_sub)

    def restore_data(self, data):
        return data.reshape(-1, self.D)

    def compute_quantiles(self, data):
        cols = data.shape[1]
        quantiles = [np.percentile(data[:, col], [100 * i / self.num_groups for i in range(1, self.num_groups)]) for col
                     in range(cols)]
        return np.array(quantiles)

    def determine_group(self, value, quantiles):
        groups = np.array(
            [np.searchsorted(quantiles[col], value[:, col], side='right') for col in range(len(value[0]))]).T
        weights = np.array([self.num_groups ** i for i in range(self.n_sub)])
        return np.sum(groups * weights, axis=1)

    def getK_pq(self,n_row, n_col, M, mem_in_byte, dtype):
        if (mem_in_byte <= 0): return 0

        if dtype == "float32":
            cell_bytes = 4
        elif dtype == "float16":
            cell_bytes = 2

        low, high = 1, n_row
        max_Ks = 0

        while low < high - 1:
            mid = (low + high) // 2
            if (mid * n_col * cell_bytes + (n_row * M* math.log2(mid - 1))/8) <= mem_in_byte:
                low = mid
            else:
                high = mid

        max_Ks = low
        return max_Ks
    def fit(self, data):
        N, D = data.shape
        self.D=D




        data_1 = self.transform_data(data)
        n0=data_1.shape[0]
        self.quantiles = self.compute_quantiles(data_1)
        self.group = self.determine_group(data_1, self.quantiles)

        counts = np.zeros(self.num_groups ** self.n_sub, dtype=int)
        unique_groups, unique_counts = np.unique(self.group, return_counts=True)
        counts[unique_groups] = unique_counts

        total_memory = self.compression_rate * N * D *4
        ks=self.getK_pq(n0,self.n_sub,1,total_memory,dtype="float32")
        print(n0,ks)
        centroids_per_group = (counts / np.sum(counts)) * ks
        centroids_per_group = np.round(centroids_per_group).astype(int)
        centroids_per_group=np.where(centroids_per_group==0,1,centroids_per_group)
        self.all_centroids = []
        self.indices = []

        for g in range(self.num_groups ** self.n_sub):
            vectors = data_1[self.group == g]
            if  vectors.size > 0:
                centroids, _ = kmeans2(vectors, centroids_per_group[g], iter=self.pq_iter, minit='points')
                start_index = len(self.all_centroids)
                self.all_centroids.extend(centroids)
                end_index = len(self.all_centroids)
                self.indices.append((start_index, end_index))
            else:
                self.indices.append((0, 0))

        self.is_fitted = True

    def encode(self, data):
        if not self.is_fitted:
            raise Exception("Model is not fitted yet.")

        data_1 = self.transform_data(data )
        encode_data = np.zeros(data_1.shape[0], dtype=int)

        for idx in range(self.num_groups ** self.n_sub):
            if self.indices[idx][1] > 0:  # Check if there are centroids in this group
                vectors = data_1[self.group == idx]
                start, end = self.indices[idx]
                closest_idx, _ = vq(vectors, self.all_centroids[start:end])
                encode_data[self.group == idx] = start + closest_idx
        self.all_centroids = np.array(self.all_centroids)

        return encode_data

    def decode(self, encode_data):
        if not self.is_fitted:
            raise Exception("Model is not fitted yet.")
        encode_data=encode_data.astype(int)
        encode_data = encode_data.reshape(-1)



        decode_data = self.all_centroids[encode_data]
        original_shape_data = self.restore_data(decode_data)



        return original_shape_data