import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class NonlinearTransformEmbedding(nn.Module):
    def __init__(self, input_dim=1, output_dim=16):
        super(NonlinearTransformEmbedding, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, weight):
        weight = torch.tensor([[weight]], dtype=torch.float32)
        embedding = F.relu(self.linear(weight))
        return embedding.flatten()


def gaussian_embedding(weight, dim=16, sigma=0.1):
    embeddings = np.random.normal(loc=weight, scale=sigma, size=dim)
    return embeddings


def nonlinear_transform_embedding(weight, dim=16):
    # np.random.seed(42)
    transform_matrix = np.random.rand(dim, 1)
    linear_output = np.dot(transform_matrix, np.array([[weight]]))
    embedding = np.maximum(linear_output, 0)
    embeddings = embedding.flatten()
    return embeddings


def transformer_positional_encoding(seq_len, d_model):
    position = np.arange(seq_len).reshape(-1, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe


def fixed_positional_encoding(flags, num_positions=3, num_feature=16):
    pos_embedding = nn.Embedding(num_positions, num_feature)
    positional_encoding = pos_embedding(flags)
    return positional_encoding


def generate_positional_encoding(seq_len=3, d_model=16):
    position = np.arange(seq_len).reshape(-1, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe


# Example usage
# positional_encoding = learnable_positional_encoding(seq_len = 2, d_model = 16)
# print(positional_encoding)
# positional_encoding = generate_positional_encoding(seq_len = 2, d_model = 16)
# print(positional_encoding)

# # Visualization
# plt.figure(figsize=(8, 4))
# for i in range(d_model):
#     plt.plot(positional_encoding[:, i], label=f"Dim {i}")
# plt.xlabel("Position")
# plt.ylabel("Encoding Value")
# plt.title("Positional Encoding Visualization")
# plt.legend()
# plt.show()

# --- New Embeddings for count_ratio (0-1 continuous value) ---


def _continuous_sinusoidal_base_embedding(value_0_to_1, even_dim, base_value=100.0):
    """Helper for sinusoidal embedding, expects even_dim.
    Implements Transformer-style positional encoding adapted for a continuous value in [0,1].
    Low dimensions have higher frequencies (shorter periods).
    High dimensions have lower frequencies (longer periods).
    The input 'value_0_to_1' is scaled by 2*pi before applying to sin/cos arguments.
    The base value for period calculation is parameterizable.
    """
    if even_dim == 0:
        return np.array([], dtype=np.float32)

    position_scalar = float(value_0_to_1) * 2.0 * math.pi

    dim_indices = np.arange(0, even_dim, 2, dtype=np.float64)

    period_term = np.power(float(base_value), dim_indices / float(even_dim))

    period_term[period_term == 0] = 1e-6

    embedding = np.zeros(even_dim, dtype=np.float32)
    embedding[0::2] = np.sin(position_scalar / period_term)
    embedding[1::2] = np.cos(position_scalar / period_term)
    return embedding.flatten()


def get_sinusoidal_embedding_for_continuous_value(
    value_0_to_1, dim=16, base_value=100.0
):
    """
    Generates a sinusoidal-based embedding for a continuous value in [0, 1].
    If dim is odd, the last dimension will be the value itself.
    Args:
        value_0_to_1 (float): Input value, expected to be in [0, 1].
        dim (int): Desired output embedding dimension.
        base_value (float): The base value used in period calculation (default 100.0 for [0,1] scaled inputs).
    Returns:
        np.ndarray: Embedding vector of shape (dim,).
    """
    value_0_to_1 = np.clip(float(value_0_to_1), 0.0, 1.0)

    if not isinstance(dim, int) or dim <= 0:
        return np.array([], dtype=np.float32)

    if dim == 1:
        return np.array([value_0_to_1], dtype=np.float32)

    if dim % 2 == 0:
        return _continuous_sinusoidal_base_embedding(value_0_to_1, dim, base_value)
    else:
        sin_cos_embedding = _continuous_sinusoidal_base_embedding(
            value_0_to_1, dim - 1, base_value
        )
        return np.append(sin_cos_embedding, value_0_to_1).astype(np.float32)


def get_rbf_embedding_for_continuous_value(value_0_to_1, dim=16, sigma=None):
    """
    Generates an RBF-based embedding for a continuous value in [0, 1].
    Args:
        value_0_to_1 (float): Input value, expected to be in [0, 1].
        dim (int): Desired output embedding dimension (number of RBF centers).
        sigma (float, optional): Width of the Gaussian RBF kernels.
                                If None, defaults to a heuristic based on dim.
    Returns:
        np.ndarray: Embedding vector of shape (dim,).
    """
    value_0_to_1 = np.clip(float(value_0_to_1), 0.0, 1.0)

    if not isinstance(dim, int) or dim <= 0:
        # print(f"Warning: Invalid dimension {dim} for RBF. Returning empty array.")
        return np.array([], dtype=np.float32)

    centers = np.linspace(0, 1, dim, dtype=np.float32)

    if sigma is None:
        if dim == 1:
            sigma = 0.5  # For a single RBF, a wider sigma might be reasonable
        else:
            # Heuristic: make sigma such that RBFs overlap moderately
            # This value means the std dev is about 1/dim of the range.
            sigma = 1.0 / float(dim)
    else:
        sigma = float(sigma)

    diffs = value_0_to_1 - centers
    embedding = np.exp(-(diffs**2) / (2 * sigma**2))
    return embedding.astype(np.float32)


# --- End of New Embeddings ---
