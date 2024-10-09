# HyperChr: QUANTIZATION OF HETEROGENEOUSLY DISTRIBUTED MATRICES THROUGH DISTRIBUTION-AWARE SUBSPACE PARTITIONIN

## Introduction

This project provides Python implementations of four quantization methods for efficient similarity search and data compression:

- **HyperChr**: Our Method
- **LOPQ**: Locally Optimized Product Quantization
- **OPQ**: Optimized Product Quantization
- **PQ**: Product Quantization

Our primary contribution is the **HyperChr** method, which is designed to improve upon existing quantization techniques by introducing a hierarchical approach for better accuracy and efficiency.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Algorithms](#algorithms)
  - [HyperChr (Our Method)](#hpq-our-method)
  - [LOPQ](#lopq)
  - [OPQ](#opq)
  - [PQ](#pq)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [HPQ Example](#hpq-example)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features

- **High Efficiency**: Implementations are optimized for performance and scalability.
- **Flexible Configurations**: Support for customizing parameters like the number of subspaces, codewords, and compression rates.
- **Easy Integration**: Modular codebase allows for easy integration into existing projects.
- **Educational Value**: Clear implementations suitable for learning and extending the algorithms.

## Algorithms

### HyperChr (Our Method)

**HyperChr** is our novel approach that introduces a hierarchical structure to product quantization, allowing for more efficient encoding and decoding of high-dimensional data. HPQ adaptively quantizes data based on its distribution and employs a hierarchical grouping mechanism to improve both compression and retrieval performance under memory constraints.

**Key Features of HyperChr**:

- **Adaptive Quantization**: Automatically computes quantiles based on data distribution for finer grouping.
- **Memory Optimization**: Dynamically calculates the number of centroids per group according to memory constraints.
- **Hierarchical Structure**: Utilizes a multi-level quantization approach to capture data nuances.

### LOPQ

**Locally Optimized Product Quantization (LOPQ)** enhances standard PQ by optimizing quantization locally within clusters, leading to improved accuracy in nearest neighbor search tasks.

### OPQ

**Optimized Product Quantization (OPQ)** introduces a rotation matrix to minimize quantization errors, making it highly effective for approximate nearest neighbor searches in high-dimensional spaces.

### PQ

**Product Quantization (PQ)** is a fundamental method for compressing high-dimensional vectors. It divides vectors into subspaces and quantizes each subspace separately, enabling fast and memory-efficient approximate nearest neighbor searches.

## Requirements

- Python 3.x
- NumPy
- SciPy

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/HyperChr-release/HyperChr-release.git
   ```

2. **Navigate to the project directory**

   ```bash
   cd HyperChr-release
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   Ensure that `requirements.txt` contains:

   ```
   numpy
   scipy
   ```

## Usage

### HyperChr Example

Here's how to use our **HyperChr** implementation:

```python
import numpy as np
from hpq_Q_A(3) import HPQ_Q_A

# Generate random data
data = np.random.rand(10000, 128)  # 10000 samples, each with 128 dimensions

# Initialize the HPQ model
hpq = HPQ_Q_A(
    num_groups=4,          # Number of quantization groups per dimension
    compression_rate=0.5,  # Compression rate (e.g., 0.5 for 50% compression)
    n_sub=4,               # Number of subspaces
    pq_iter=20             # Number of iterations for k-means clustering
)

# Fit the model to the data
hpq.fit(data)

# Encode the data
encoded_data = hpq.encode(data)

# Decode the data
decoded_data = hpq.decode(encoded_data)

# Compute reconstruction error
error = np.linalg.norm(data - decoded_data)
print(f"Reconstruction Error: {error}")
```

## Project Structure

```plaintext

├── HPQ_Q_A.py              # HPQ implementation (Our Method)
├── lopq.py             # LOPQ implementation
├── opq.py              # OPQ implementation
├── pq.py               # PQ implementation
├── requirements.txt    # List of dependencies
├── README.md           # Project description
```

## Contributing

Contributions are welcome! You can contribute in the following ways:

- **Reporting Issues**: Submit issues for bugs or feature requests.
- **Pull Requests**: Fork the repository and submit pull requests for improvements.
- **Documentation**: Help improve the documentation.

Please ensure your contributions adhere to the project's coding standards and styles.

## License

This project is licensed under the [MIT License](LICENSE).

- **Note**: Some code in this project is adapted from external sources. Specifically:
  - `pq.py` and portions of `opq.py` are adapted from the [nanopq](https://github.com/matsui528/nanopq) library by **Yusuke Matsui**.
  - If you use these portions of the code, please adhere to the licensing terms specified in the original repository.

## Acknowledgments

- **Yusuke Matsui** for the nanopq library, which inspired parts of this project.
- The community for continuous support and contributions.

---

If you have any questions or need further assistance, feel free to open an issue or contact the maintainers.
