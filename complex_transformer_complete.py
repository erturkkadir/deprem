"""
Complete Complex-Valued Transformer Implementation
===================================================
A comprehensive implementation of a transformer using complex numbers for:
- Token embeddings (complex-valued)
- Rotary Position Embeddings (RoPE)
- Multi-head attention with complex Q, K, V
- Complex feedforward networks
- Complex layer normalization
- Complex activations (CReLU, modReLU, complex GELU)
- Custom optimizer for complex parameters

Author: Claude (Anthropic)
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings


# =============================================================================
# CONFIGURATION
# =============================================================================

class ActivationType(Enum):
    """Supported activation functions for complex networks."""
    CRELU = "crelu"
    MODRELU = "modrelu"
    CGELU = "cgelu"
    ZRELU = "zrelu"
    CARDIOID = "cardioid"


class AttentionType(Enum):
    """Attention computation variants."""
    HERMITIAN = "hermitian"      # Uses conjugate: Re(Q @ K*)
    REAL_PART = "real_part"      # Uses Re(Q) @ Re(K) + Im(Q) @ Im(K)
    MAGNITUDE = "magnitude"      # Uses |Q @ K*|


class PoolingType(Enum):
    """Pooling strategies for classification."""
    MEAN = "mean"
    CLS = "cls"
    MAX = "max"


@dataclass
class ComplexTransformerConfig:
    """
    Configuration for Complex Transformer model.
    
    Attributes:
        vocab_size: Size of vocabulary
        embed_dim: Dimension of embeddings (must be divisible by num_heads)
        num_heads: Number of attention heads
        num_layers: Number of transformer blocks
        ff_dim: Hidden dimension of feedforward network
        max_seq_len: Maximum sequence length for positional encoding
        dropout: Dropout probability
        attention_dropout: Dropout for attention weights (separate from main dropout)
        layer_norm_eps: Epsilon for layer normalization stability
        use_rope: Whether to use Rotary Position Embeddings
        rope_theta: Base frequency for RoPE
        activation: Activation function type for FFN
        attention_type: Method for computing attention scores
        pre_norm: Use pre-norm (True) or post-norm (False) architecture
        tie_embeddings: Tie input and output embeddings
        init_std: Standard deviation for weight initialization
        gradient_checkpointing: Enable gradient checkpointing for memory efficiency
    """
    vocab_size: int = 32000
    embed_dim: int = 512
    num_heads: int = 8
    num_layers: int = 6
    ff_dim: int = 2048
    max_seq_len: int = 2048
    dropout: float = 0.1
    attention_dropout: float = 0.1
    layer_norm_eps: float = 1e-6
    use_rope: bool = True
    rope_theta: float = 10000.0
    activation: ActivationType = ActivationType.MODRELU
    attention_type: AttentionType = AttentionType.REAL_PART
    pre_norm: bool = True
    tie_embeddings: bool = False
    init_std: float = 0.02
    gradient_checkpointing: bool = False
    
    # Additional options
    use_bias: bool = True
    ff_expansion_factor: float = 4.0
    qkv_bias: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.embed_dim % self.num_heads == 0, \
            f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})"
        
        if self.ff_dim is None:
            self.ff_dim = int(self.embed_dim * self.ff_expansion_factor)


# =============================================================================
# COMPLEX NUMBER UTILITIES
# =============================================================================

def ensure_complex(x: torch.Tensor) -> torch.Tensor:
    """Convert tensor to complex if not already."""
    if not x.is_complex():
        return torch.complex(x, torch.zeros_like(x))
    return x


def complex_to_real_2d(x: torch.Tensor) -> torch.Tensor:
    """
    Convert complex tensor to real by concatenating real and imaginary parts.
    [..., D] complex -> [..., 2*D] real
    """
    return torch.cat([x.real, x.imag], dim=-1)


def real_2d_to_complex(x: torch.Tensor) -> torch.Tensor:
    """
    Convert real tensor back to complex by splitting.
    [..., 2*D] real -> [..., D] complex
    """
    d = x.shape[-1] // 2
    return torch.complex(x[..., :d], x[..., d:])


# =============================================================================
# COMPLEX ACTIVATION FUNCTIONS
# =============================================================================

class ComplexActivation(nn.Module):
    """Base class for complex activation functions."""
    pass


class CReLU(ComplexActivation):
    """
    Complex ReLU: applies ReLU separately to real and imaginary parts.
    CReLU(z) = ReLU(Re(z)) + i * ReLU(Im(z))
    
    Simple but may cause phase discontinuities.
    """
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.complex(F.relu(z.real), F.relu(z.imag))


class ModReLU(ComplexActivation):
    """
    Modulus ReLU: applies ReLU to magnitude, preserves phase.
    modReLU(z) = ReLU(|z| + b) * e^(i * angle(z))
    
    Better gradient flow and phase preservation than CReLU.
    The learnable bias b allows the network to learn appropriate thresholds.
    """
    def __init__(self, num_features: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        magnitude = z.abs()
        phase = z.angle()
        
        # ReLU on (magnitude + bias)
        activated_magnitude = F.relu(magnitude + self.bias)
        
        # Reconstruct with original phase
        return activated_magnitude * torch.exp(1j * phase)


class ComplexGELU(ComplexActivation):
    """
    Complex GELU: applies GELU separately to real and imaginary parts.
    Smoother than CReLU, often works better in practice.
    """
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.complex(F.gelu(z.real), F.gelu(z.imag))


class ZReLU(ComplexActivation):
    """
    Z-ReLU: keeps values only in the first quadrant (positive real and imag).
    ZReLU(z) = z if Re(z) > 0 and Im(z) > 0, else 0
    
    More restrictive but maintains holomorphic-like properties in active region.
    """
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mask = (z.real > 0) & (z.imag > 0)
        return z * mask


class Cardioid(ComplexActivation):
    """
    Cardioid activation: smooth, bounded activation inspired by cardioid shape.
    cardioid(z) = 0.5 * (1 + cos(angle(z))) * z
    
    Provides smooth phase-dependent gating.
    """
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        phase = z.angle()
        scale = 0.5 * (1 + torch.cos(phase))
        return scale * z


class ComplexSiLU(ComplexActivation):
    """
    Complex SiLU (Swish): z * sigmoid(|z|)
    Uses magnitude for gating while preserving phase.
    """
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(z.abs())
        return z * gate


def get_complex_activation(activation_type: ActivationType, num_features: int = None) -> ComplexActivation:
    """Factory function for complex activations."""
    if activation_type == ActivationType.CRELU:
        return CReLU()
    elif activation_type == ActivationType.MODRELU:
        assert num_features is not None, "ModReLU requires num_features"
        return ModReLU(num_features)
    elif activation_type == ActivationType.CGELU:
        return ComplexGELU()
    elif activation_type == ActivationType.ZRELU:
        return ZReLU()
    elif activation_type == ActivationType.CARDIOID:
        return Cardioid()
    else:
        raise ValueError(f"Unknown activation type: {activation_type}")


# =============================================================================
# COMPLEX DROPOUT
# =============================================================================

class ComplexDropout(nn.Module):
    """
    Dropout for complex tensors.
    Drops both real and imaginary parts together to maintain consistency.
    """
    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0:
            return z
        
        # Create real-valued mask
        mask = torch.bernoulli(
            torch.full(z.shape, 1 - self.p, device=z.device, dtype=z.real.dtype)
        )
        mask = mask / (1 - self.p)  # Scale for expected value preservation
        
        return z * mask


# =============================================================================
# COMPLEX LINEAR LAYER
# =============================================================================

class ComplexLinear(nn.Module):
    """
    Complex-valued linear transformation.
    
    For complex weight W = W_r + i*W_i and input x = x_r + i*x_i:
    Wx = (W_r @ x_r - W_i @ x_i) + i*(W_r @ x_i + W_i @ x_r)
    
    This implements the standard complex matrix multiplication.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init_std: float = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Separate real and imaginary weight matrices
        self.weight_real = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_imag = nn.Parameter(torch.empty(out_features, in_features))
        
        if bias:
            self.bias_real = nn.Parameter(torch.empty(out_features))
            self.bias_imag = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_imag', None)
        
        self._init_weights(init_std)
    
    def _init_weights(self, init_std: float = None):
        """
        Initialize weights using complex Glorot initialization.
        
        For complex weights, variance should be split between real and imag
        to maintain the same overall variance as real-valued layers.
        """
        if init_std is None:
            # Complex Glorot: variance = 1 / (fan_in + fan_out)
            # Split between real and imag: each gets variance / 2
            std = 1.0 / math.sqrt(2 * (self.in_features + self.out_features))
        else:
            std = init_std / math.sqrt(2)  # Split variance
        
        nn.init.normal_(self.weight_real, std=std)
        nn.init.normal_(self.weight_imag, std=std)
        
        if self.bias_real is not None:
            nn.init.zeros_(self.bias_real)
            nn.init.zeros_(self.bias_imag)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Complex tensor of shape [..., in_features]
        Returns:
            Complex tensor of shape [..., out_features]
        """
        # Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        real_part = F.linear(x.real, self.weight_real) - F.linear(x.imag, self.weight_imag)
        imag_part = F.linear(x.real, self.weight_imag) + F.linear(x.imag, self.weight_real)
        
        if self.bias_real is not None:
            real_part = real_part + self.bias_real
            imag_part = imag_part + self.bias_imag
        
        return torch.complex(real_part, imag_part)
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias_real is not None}'


# =============================================================================
# COMPLEX LAYER NORMALIZATION
# =============================================================================

class ComplexLayerNorm(nn.Module):
    """
    Layer normalization for complex tensors.
    
    Several approaches exist:
    1. Normalize based on magnitude (|z|)
    2. Normalize real and imaginary separately
    3. Normalize based on complex variance
    
    This implementation uses approach 3 (complex variance) which is most
    theoretically grounded for complex-valued networks.
    
    Complex variance: Var(z) = E[|z - μ|²] where μ = E[z] is complex mean
    """
    
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True
    ):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if elementwise_affine:
            # Complex scale and shift parameters
            self.gamma_real = nn.Parameter(torch.ones(normalized_shape))
            self.gamma_imag = nn.Parameter(torch.zeros(normalized_shape))
            self.beta_real = nn.Parameter(torch.zeros(normalized_shape))
            self.beta_imag = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter('gamma_real', None)
            self.register_parameter('gamma_imag', None)
            self.register_parameter('beta_real', None)
            self.register_parameter('beta_imag', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Complex tensor of shape [..., normalized_shape]
        Returns:
            Normalized complex tensor of same shape
        """
        # Compute complex mean
        mean_real = x.real.mean(dim=-1, keepdim=True)
        mean_imag = x.imag.mean(dim=-1, keepdim=True)
        
        # Center the data
        x_centered_real = x.real - mean_real
        x_centered_imag = x.imag - mean_imag
        
        # Compute variance: E[|x - μ|²] = E[(x_r - μ_r)² + (x_i - μ_i)²]
        variance = (x_centered_real ** 2 + x_centered_imag ** 2).mean(dim=-1, keepdim=True)
        std = torch.sqrt(variance + self.eps)
        
        # Normalize
        norm_real = x_centered_real / std
        norm_imag = x_centered_imag / std
        
        if self.elementwise_affine:
            # Apply complex affine: (norm_r + i*norm_i) * (γ_r + i*γ_i) + (β_r + i*β_i)
            out_real = norm_real * self.gamma_real - norm_imag * self.gamma_imag + self.beta_real
            out_imag = norm_real * self.gamma_imag + norm_imag * self.gamma_real + self.beta_imag
            return torch.complex(out_real, out_imag)
        
        return torch.complex(norm_real, norm_imag)


class ComplexRMSNorm(nn.Module):
    """
    RMS Normalization for complex tensors.
    
    Simpler than LayerNorm - only normalizes by RMS, no centering.
    RMS(z) = sqrt(mean(|z|²))
    """
    
    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        # Complex scale parameter
        self.gamma_real = nn.Parameter(torch.ones(normalized_shape))
        self.gamma_imag = nn.Parameter(torch.zeros(normalized_shape))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS of complex magnitude
        rms = torch.sqrt((x.real ** 2 + x.imag ** 2).mean(dim=-1, keepdim=True) + self.eps)
        
        # Normalize
        norm_real = x.real / rms
        norm_imag = x.imag / rms
        
        # Apply complex scale
        out_real = norm_real * self.gamma_real - norm_imag * self.gamma_imag
        out_imag = norm_real * self.gamma_imag + norm_imag * self.gamma_real
        
        return torch.complex(out_real, out_imag)


# =============================================================================
# COMPLEX EMBEDDINGS
# =============================================================================

class ComplexEmbedding(nn.Module):
    """
    Complex-valued token embedding.
    
    Each token is represented as a complex vector, enabling richer
    representation through both magnitude and phase components.
    
    The complex representation can encode:
    - Magnitude: "importance" or "strength" of features
    - Phase: "type" or "category" of features
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        init_std: float = 0.02
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        
        # Separate embeddings for real and imaginary parts
        self.embed_real = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.embed_imag = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        
        self._init_weights(init_std)
    
    def _init_weights(self, init_std: float):
        """Initialize with small random values, split variance."""
        std = init_std / math.sqrt(2)
        nn.init.normal_(self.embed_real.weight, std=std)
        nn.init.normal_(self.embed_imag.weight, std=std)
        
        if self.padding_idx is not None:
            nn.init.zeros_(self.embed_real.weight[self.padding_idx])
            nn.init.zeros_(self.embed_imag.weight[self.padding_idx])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token indices of shape [batch, seq_len]
        Returns:
            Complex embeddings of shape [batch, seq_len, embedding_dim]
        """
        return torch.complex(self.embed_real(x), self.embed_imag(x))


# =============================================================================
# ROTARY POSITION EMBEDDING (RoPE)
# =============================================================================

class ComplexRotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for complex-valued attention.
    
    RoPE encodes positions as rotations in 2D subspaces. For complex numbers,
    this is naturally expressed as multiplication by e^(i*θ).
    
    Key properties:
    - Relative position information emerges from dot products
    - No explicit position embedding vectors needed
    - Extrapolates well to longer sequences
    
    Reference: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    """
    
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        theta: float = 10000.0,
        scaling_factor: float = 1.0
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.scaling_factor = scaling_factor
        
        # Compute inverse frequencies
        # θ_i = theta^(-2i/dim) for i = 0, 1, ..., dim/2 - 1
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Build cache
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        """Precompute cos and sin for all positions."""
        positions = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        positions = positions / self.scaling_factor
        
        # Outer product: [seq_len, dim/2]
        freqs = torch.outer(positions, self.inv_freq)
        
        # Cache cos and sin
        self.register_buffer('cos_cached', freqs.cos(), persistent=False)
        self.register_buffer('sin_cached', freqs.sin(), persistent=False)
    
    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get rotation matrices for given sequence length.
        
        Args:
            seq_len: Current sequence length
            device: Device to put tensors on
        Returns:
            (cos, sin) tensors of shape [seq_len, dim/2]
        """
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            self.max_seq_len = seq_len
        
        return (
            self.cos_cached[:seq_len].to(device),
            self.sin_cached[:seq_len].to(device)
        )


def apply_complex_rotary_embedding(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to complex Q and K tensors.
    
    For complex tensors, we apply rotation to consecutive pairs of dimensions.
    This is equivalent to multiplying by e^(i*θ) in each 2D subspace.
    
    Args:
        q: Complex query tensor [batch, heads, seq_len, head_dim]
        k: Complex key tensor [batch, heads, seq_len, head_dim]
        cos: Cosine values [seq_len, head_dim/2]
        sin: Sine values [seq_len, head_dim/2]
        position_ids: Optional position indices [batch, seq_len]
    Returns:
        Rotated (q, k) tensors
    """
    batch_size, num_heads, seq_len, head_dim = q.shape
    
    # Handle position_ids for variable-length sequences
    if position_ids is not None:
        cos = cos[position_ids]  # [batch, seq_len, head_dim/2]
        sin = sin[position_ids]
        cos = cos.unsqueeze(1)   # [batch, 1, seq_len, head_dim/2]
        sin = sin.unsqueeze(1)
    else:
        cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim/2]
        sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)
    
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of x."""
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)
    
    def apply_rotation(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply rotation to both real and imaginary parts."""
        # Expand cos/sin to full head_dim
        cos_full = torch.cat([cos, cos], dim=-1)
        sin_full = torch.cat([sin, sin], dim=-1)
        
        # Rotate real part
        x_real_rot = x.real * cos_full + rotate_half(x.real) * sin_full
        # Rotate imaginary part
        x_imag_rot = x.imag * cos_full + rotate_half(x.imag) * sin_full
        
        return torch.complex(x_real_rot, x_imag_rot)
    
    q_rot = apply_rotation(q, cos, sin)
    k_rot = apply_rotation(k, cos, sin)
    
    return q_rot, k_rot


# =============================================================================
# COMPLEX MULTI-HEAD ATTENTION
# =============================================================================

class ComplexMultiHeadAttention(nn.Module):
    """
    Multi-head attention with complex-valued queries, keys, and values.
    
    Three methods for computing attention scores from complex vectors:
    
    1. HERMITIAN: Uses conjugate transpose (standard complex inner product)
       score = Re(Q @ K^H) = Re(q) @ Re(k) + Im(q) @ Im(k)
       
    2. REAL_PART: Uses real part of regular product
       score = Re(Q @ K^T) = Re(q) @ Re(k) - Im(q) @ Im(k)
       
    3. MAGNITUDE: Uses magnitude of complex product
       score = |Q @ K^H|
    
    The HERMITIAN/REAL_PART methods are most common as they're efficient
    and provide meaningful similarity measures.
    """
    
    def __init__(self, config: ComplexTransformerConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads
        self.scale = self.head_dim ** -0.5
        self.attention_type = config.attention_type
        
        # Q, K, V projections
        self.q_proj = ComplexLinear(config.embed_dim, config.embed_dim, bias=config.qkv_bias)
        self.k_proj = ComplexLinear(config.embed_dim, config.embed_dim, bias=config.qkv_bias)
        self.v_proj = ComplexLinear(config.embed_dim, config.embed_dim, bias=config.qkv_bias)
        self.out_proj = ComplexLinear(config.embed_dim, config.embed_dim, bias=config.use_bias)
        
        # Rotary embeddings
        if config.use_rope:
            self.rotary_emb = ComplexRotaryEmbedding(
                self.head_dim,
                config.max_seq_len,
                config.rope_theta
            )
        else:
            self.rotary_emb = None
        
        self.attention_dropout = ComplexDropout(config.attention_dropout)
        self.resid_dropout = ComplexDropout(config.dropout)
    
    def _compute_attention_scores(
        self,
        Q: torch.Tensor,
        K: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute attention scores based on configured method.
        
        Args:
            Q: [batch, heads, seq_len, head_dim] complex
            K: [batch, heads, seq_len, head_dim] complex
        Returns:
            Real-valued attention scores [batch, heads, seq_len, seq_len]
        """
        if self.attention_type == AttentionType.HERMITIAN:
            # Hermitian inner product: <q, k> = sum(q * conj(k))
            # Re(<q,k>) = Re(q)@Re(k)^T + Im(q)@Im(k)^T
            scores = (
                torch.matmul(Q.real, K.real.transpose(-2, -1)) +
                torch.matmul(Q.imag, K.imag.transpose(-2, -1))
            )
        
        elif self.attention_type == AttentionType.REAL_PART:
            # Real part of regular product
            # Re(q @ k^T) = Re(q)@Re(k)^T - Im(q)@Im(k)^T
            scores = (
                torch.matmul(Q.real, K.real.transpose(-2, -1)) -
                torch.matmul(Q.imag, K.imag.transpose(-2, -1))
            )
        
        elif self.attention_type == AttentionType.MAGNITUDE:
            # Magnitude of complex product
            real_part = (
                torch.matmul(Q.real, K.real.transpose(-2, -1)) +
                torch.matmul(Q.imag, K.imag.transpose(-2, -1))
            )
            imag_part = (
                torch.matmul(Q.imag, K.real.transpose(-2, -1)) -
                torch.matmul(Q.real, K.imag.transpose(-2, -1))
            )
            scores = torch.sqrt(real_part ** 2 + imag_part ** 2 + 1e-8)
        
        else:
            raise ValueError(f"Unknown attention type: {self.attention_type}")
        
        return scores * self.scale
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: Complex input tensor [batch, seq_len, embed_dim]
            attention_mask: Optional mask [batch, seq_len] or [batch, 1, seq_len, seq_len]
            is_causal: Whether to apply causal masking
            position_ids: Optional position indices for RoPE
            past_key_value: Cached K, V for inference
            use_cache: Whether to return updated cache
        Returns:
            (output, cache) where output is [batch, seq_len, embed_dim]
        """
        B, L, D = x.shape
        
        # Project to Q, K, V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Reshape for multi-head attention: [B, L, D] -> [B, H, L, D_h]
        Q = Q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Handle KV cache for inference
        if past_key_value is not None:
            K = torch.cat([past_key_value[0], K], dim=2)
            V = torch.cat([past_key_value[1], V], dim=2)
        
        cache = (K, V) if use_cache else None
        
        # Apply rotary embeddings
        if self.rotary_emb is not None:
            cos, sin = self.rotary_emb(K.shape[2], Q.device)
            Q, K = apply_complex_rotary_embedding(Q, K, cos, sin, position_ids)
        
        # Compute attention scores
        attn_scores = self._compute_attention_scores(Q, K)
        
        # Apply causal mask
        if is_causal:
            causal_mask = torch.triu(
                torch.ones(L, K.shape[2], device=x.device, dtype=torch.bool),
                diagonal=K.shape[2] - L + 1
            )
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        
        # Apply attention mask
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # [B, L] -> [B, 1, 1, L]
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Softmax (real-valued)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.config.attention_dropout, training=self.training)
        
        # Apply attention to complex values
        out_real = torch.matmul(attn_weights, V.real)
        out_imag = torch.matmul(attn_weights, V.imag)
        out = torch.complex(out_real, out_imag)
        
        # Reshape back: [B, H, L, D_h] -> [B, L, D]
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        
        # Output projection
        out = self.out_proj(out)
        out = self.resid_dropout(out)
        
        return out, cache


# =============================================================================
# COMPLEX FEED-FORWARD NETWORK
# =============================================================================

class ComplexFeedForward(nn.Module):
    """
    Position-wise feed-forward network for complex transformers.
    
    Standard FFN: FFN(x) = activation(x @ W1 + b1) @ W2 + b2
    
    For complex networks, we use complex linear layers and complex activations.
    The default activation is modReLU which preserves phase information.
    """
    
    def __init__(self, config: ComplexTransformerConfig):
        super().__init__()
        
        self.fc1 = ComplexLinear(
            config.embed_dim,
            config.ff_dim,
            bias=config.use_bias,
            init_std=config.init_std
        )
        self.fc2 = ComplexLinear(
            config.ff_dim,
            config.embed_dim,
            bias=config.use_bias,
            init_std=config.init_std
        )
        
        self.activation = get_complex_activation(config.activation, config.ff_dim)
        self.dropout = ComplexDropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Complex tensor [batch, seq_len, embed_dim]
        Returns:
            Complex tensor [batch, seq_len, embed_dim]
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class ComplexGatedFeedForward(nn.Module):
    """
    Gated feed-forward network (SwiGLU-style) for complex transformers.
    
    GLU(x) = (x @ W) * σ(x @ V) @ W2
    
    Uses complex gating for potentially better expressivity.
    """
    
    def __init__(self, config: ComplexTransformerConfig):
        super().__init__()
        
        # Gated linear unit style
        self.gate_proj = ComplexLinear(config.embed_dim, config.ff_dim, bias=False)
        self.up_proj = ComplexLinear(config.embed_dim, config.ff_dim, bias=False)
        self.down_proj = ComplexLinear(config.ff_dim, config.embed_dim, bias=False)
        
        self.dropout = ComplexDropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Complex SiLU gating
        gate = self.gate_proj(x)
        gate_activated = gate * torch.sigmoid(gate.abs())  # Complex SiLU
        
        up = self.up_proj(x)
        
        # Element-wise complex multiplication
        hidden = torch.complex(
            gate_activated.real * up.real - gate_activated.imag * up.imag,
            gate_activated.real * up.imag + gate_activated.imag * up.real
        )
        
        output = self.down_proj(hidden)
        return self.dropout(output)


# =============================================================================
# COMPLEX TRANSFORMER BLOCK
# =============================================================================

class ComplexTransformerBlock(nn.Module):
    """
    A single transformer block with complex-valued components.
    
    Supports both pre-norm and post-norm architectures.
    Pre-norm (default): More stable training, used in modern LLMs
    Post-norm: Original transformer architecture
    """
    
    def __init__(self, config: ComplexTransformerConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.pre_norm = config.pre_norm
        
        # Attention and FFN
        self.attention = ComplexMultiHeadAttention(config)
        self.feed_forward = ComplexFeedForward(config)
        
        # Normalization layers
        self.norm1 = ComplexLayerNorm(config.embed_dim, config.layer_norm_eps)
        self.norm2 = ComplexLayerNorm(config.embed_dim, config.layer_norm_eps)
        
        # Residual dropout
        self.dropout = ComplexDropout(config.dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: Complex input [batch, seq_len, embed_dim]
            attention_mask: Optional attention mask
            is_causal: Whether to use causal attention
            position_ids: Optional position indices
            past_key_value: Cached K, V
            use_cache: Whether to cache K, V
        Returns:
            (output, cache)
        """
        if self.pre_norm:
            # Pre-norm: Norm -> Sublayer -> Residual
            normed = self.norm1(x)
            attn_out, cache = self.attention(
                normed, attention_mask, is_causal, position_ids, past_key_value, use_cache
            )
            x = x + attn_out
            
            normed = self.norm2(x)
            ff_out = self.feed_forward(normed)
            x = x + self.dropout(ff_out)
        else:
            # Post-norm: Sublayer -> Residual -> Norm
            attn_out, cache = self.attention(
                x, attention_mask, is_causal, position_ids, past_key_value, use_cache
            )
            x = self.norm1(x + attn_out)
            
            ff_out = self.feed_forward(x)
            x = self.norm2(x + self.dropout(ff_out))
        
        return x, cache


# =============================================================================
# COMPLETE COMPLEX TRANSFORMER MODEL
# =============================================================================

class ComplexTransformerModel(nn.Module):
    """
    Complete complex-valued transformer encoder.
    
    This is the base model without task-specific heads.
    Can be used as backbone for various downstream tasks.
    """
    
    def __init__(self, config: ComplexTransformerConfig):
        super().__init__()
        self.config = config
        
        # Token embedding
        self.embedding = ComplexEmbedding(
            config.vocab_size,
            config.embed_dim,
            init_std=config.init_std
        )
        
        # Embedding dropout
        self.embed_dropout = ComplexDropout(config.dropout)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            ComplexTransformerBlock(config, layer_idx=i)
            for i in range(config.num_layers)
        ])
        
        # Final normalization (for pre-norm architecture)
        if config.pre_norm:
            self.final_norm = ComplexLayerNorm(config.embed_dim, config.layer_norm_eps)
        else:
            self.final_norm = nn.Identity()
        
        # Gradient checkpointing
        self.gradient_checkpointing = config.gradient_checkpointing
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Args:
            input_ids: Token indices [batch, seq_len]
            attention_mask: Optional mask [batch, seq_len]
            is_causal: Whether to use causal attention
            position_ids: Optional position indices
            past_key_values: List of cached K, V per layer
            use_cache: Whether to return caches
        Returns:
            (hidden_states, caches)
        """
        # Get embeddings
        hidden_states = self.embedding(input_ids)
        hidden_states = self.embed_dropout(hidden_states)
        
        # Process through transformer layers
        new_caches = [] if use_cache else None
        
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            
            if self.gradient_checkpointing and self.training:
                hidden_states, cache = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    is_causal,
                    position_ids,
                    past_kv,
                    use_cache,
                    use_reentrant=False
                )
            else:
                hidden_states, cache = layer(
                    hidden_states,
                    attention_mask,
                    is_causal,
                    position_ids,
                    past_kv,
                    use_cache
                )
            
            if use_cache:
                new_caches.append(cache)
        
        # Final normalization
        hidden_states = self.final_norm(hidden_states)
        
        return hidden_states, new_caches


# =============================================================================
# LANGUAGE MODEL HEAD
# =============================================================================

class ComplexTransformerLMHead(nn.Module):
    """
    Language model head for complex transformers.
    
    Converts complex hidden states to real-valued vocabulary logits.
    """
    
    def __init__(self, config: ComplexTransformerConfig):
        super().__init__()
        
        # Project complex to real by concatenating real and imaginary parts
        # Then project to vocabulary
        self.projection = nn.Linear(config.embed_dim * 2, config.vocab_size, bias=False)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: Complex tensor [batch, seq_len, embed_dim]
        Returns:
            Logits [batch, seq_len, vocab_size]
        """
        # Concatenate real and imaginary parts
        combined = complex_to_real_2d(hidden_states)
        return self.projection(combined)


class ComplexTransformerForCausalLM(nn.Module):
    """
    Complex transformer for causal language modeling.
    
    Includes the base transformer and an LM head for next-token prediction.
    """
    
    def __init__(self, config: ComplexTransformerConfig):
        super().__init__()
        self.config = config
        
        self.transformer = ComplexTransformerModel(config)
        self.lm_head = ComplexTransformerLMHead(config)
        
        # Optionally tie embeddings
        if config.tie_embeddings:
            # This is complex - we'd need to tie both real and imag
            # For simplicity, we don't implement full tying here
            pass
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: Token indices [batch, seq_len]
            attention_mask: Optional mask
            labels: Target tokens for loss computation [batch, seq_len]
            position_ids: Optional position indices
            past_key_values: Cached K, V
            use_cache: Whether to return cache
        Returns:
            Dict with 'logits', optionally 'loss' and 'past_key_values'
        """
        # Forward through transformer
        hidden_states, caches = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            is_causal=True,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache
        )
        
        # Get logits
        logits = self.lm_head(hidden_states)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        output = {'logits': logits}
        if loss is not None:
            output['loss'] = loss
        if use_cache:
            output['past_key_values'] = caches
        
        return output
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.
        
        Args:
            input_ids: Starting tokens [batch, seq_len]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (1.0 = no change)
            top_k: Keep only top-k tokens
            top_p: Keep tokens with cumulative prob < top_p (nucleus sampling)
            do_sample: Whether to sample (False = greedy)
            eos_token_id: Stop generation at this token
            pad_token_id: Padding token
        Returns:
            Generated tokens [batch, seq_len + max_new_tokens]
        """
        self.eval()
        batch_size = input_ids.shape[0]
        
        # Use KV caching for efficiency
        past_key_values = None
        
        for _ in range(max_new_tokens):
            # Only process new tokens if we have cache
            if past_key_values is not None:
                curr_input = input_ids[:, -1:]
            else:
                curr_input = input_ids
                if curr_input.size(1) > self.config.max_seq_len:
                    curr_input = curr_input[:, -self.config.max_seq_len:]
            
            # Forward pass
            outputs = self(
                curr_input,
                past_key_values=past_key_values,
                use_cache=True
            )
            logits = outputs['logits'][:, -1, :]
            past_key_values = outputs.get('past_key_values')
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Apply top-k filtering
            if top_k is not None and top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            # Sample or greedy
            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Check for EOS
            if eos_token_id is not None:
                if (next_token == eos_token_id).all():
                    break
        
        return input_ids


# =============================================================================
# SEQUENCE CLASSIFICATION HEAD
# =============================================================================

class ComplexTransformerForSequenceClassification(nn.Module):
    """
    Complex transformer for sequence classification tasks.
    
    Pools the sequence representation and projects to class logits.
    """
    
    def __init__(
        self,
        config: ComplexTransformerConfig,
        num_classes: int,
        pooling: PoolingType = PoolingType.MEAN
    ):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        self.pooling = pooling
        
        self.transformer = ComplexTransformerModel(config)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.embed_dim * 2, config.embed_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embed_dim, num_classes)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def _pool(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Pool sequence to single vector."""
        if self.pooling == PoolingType.CLS:
            # Use first token
            pooled = hidden_states[:, 0, :]
        
        elif self.pooling == PoolingType.MEAN:
            # Mean pooling
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                pooled_real = (hidden_states.real * mask).sum(dim=1) / mask.sum(dim=1)
                pooled_imag = (hidden_states.imag * mask).sum(dim=1) / mask.sum(dim=1)
                pooled = torch.complex(pooled_real, pooled_imag)
            else:
                pooled = hidden_states.mean(dim=1)
        
        elif self.pooling == PoolingType.MAX:
            # Max pooling on magnitude, then get corresponding complex values
            magnitudes = hidden_states.abs()
            if attention_mask is not None:
                magnitudes = magnitudes.masked_fill(
                    attention_mask.unsqueeze(-1) == 0, float('-inf')
                )
            max_indices = magnitudes.argmax(dim=1, keepdim=True)
            max_indices = max_indices.expand(-1, -1, hidden_states.shape[-1])
            pooled = hidden_states.gather(1, max_indices).squeeze(1)
        
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling}")
        
        return pooled
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: Token indices [batch, seq_len]
            attention_mask: Optional mask
            labels: Class labels for loss computation [batch]
        Returns:
            Dict with 'logits' and optionally 'loss'
        """
        hidden_states, _ = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            is_causal=False
        )
        
        # Pool sequence
        pooled = self._pool(hidden_states, attention_mask)
        
        # Convert to real and classify
        pooled_real = complex_to_real_2d(pooled)
        logits = self.classifier(pooled_real)
        
        # Compute loss
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        
        output = {'logits': logits}
        if loss is not None:
            output['loss'] = loss
        
        return output


# =============================================================================
# TOKEN CLASSIFICATION HEAD
# =============================================================================

class ComplexTransformerForTokenClassification(nn.Module):
    """
    Complex transformer for token classification (e.g., NER, POS tagging).
    """
    
    def __init__(self, config: ComplexTransformerConfig, num_labels: int):
        super().__init__()
        self.config = config
        self.num_labels = num_labels
        
        self.transformer = ComplexTransformerModel(config)
        
        # Classification head per token
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.embed_dim * 2, num_labels)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: Token indices [batch, seq_len]
            attention_mask: Optional mask
            labels: Token labels [batch, seq_len]
        Returns:
            Dict with 'logits' and optionally 'loss'
        """
        hidden_states, _ = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            is_causal=False
        )
        
        # Convert to real and classify
        hidden_real = complex_to_real_2d(hidden_states)
        logits = self.classifier(hidden_real)
        
        # Compute loss
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.num_labels),
                labels.view(-1),
                ignore_index=-100
            )
        
        output = {'logits': logits}
        if loss is not None:
            output['loss'] = loss
        
        return output


# =============================================================================
# CUSTOM OPTIMIZER FOR COMPLEX PARAMETERS
# =============================================================================

class ComplexAdamW(torch.optim.Optimizer):
    """
    AdamW optimizer adapted for complex parameters.
    
    Handles complex gradients by treating them as 2D real vectors,
    which is the standard approach in complex optimization.
    
    Key insight: For complex z = a + bi, we can view optimization
    as jointly optimizing (a, b) in R².
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        amsgrad: bool = False
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta[0]: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta[1]: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        
        defaults = dict(
            lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay, amsgrad=amsgrad
        )
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                if p.is_complex():
                    # Convert to real view for optimization
                    p_real = torch.view_as_real(p)
                    grad_real = torch.view_as_real(grad)
                    
                    state = self.state[p]
                    
                    # Initialize state
                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p_real)
                        state['exp_avg_sq'] = torch.zeros_like(p_real)
                        if group['amsgrad']:
                            state['max_exp_avg_sq'] = torch.zeros_like(p_real)
                    
                    exp_avg = state['exp_avg']
                    exp_avg_sq = state['exp_avg_sq']
                    
                    state['step'] += 1
                    
                    # Decoupled weight decay
                    if group['weight_decay'] != 0:
                        p_real.mul_(1 - group['lr'] * group['weight_decay'])
                    
                    # Update biased first moment estimate
                    exp_avg.mul_(beta1).add_(grad_real, alpha=1 - beta1)
                    
                    # Update biased second raw moment estimate
                    exp_avg_sq.mul_(beta2).addcmul_(grad_real, grad_real, value=1 - beta2)
                    
                    # Bias correction
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    
                    step_size = group['lr'] / bias_correction1
                    
                    if group['amsgrad']:
                        max_exp_avg_sq = state['max_exp_avg_sq']
                        torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                        denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                    else:
                        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                    
                    # Update parameters
                    p_real.addcdiv_(exp_avg, denom, value=-step_size)
                
                else:
                    # Standard AdamW for real parameters
                    state = self.state[p]
                    
                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)
                        if group['amsgrad']:
                            state['max_exp_avg_sq'] = torch.zeros_like(p)
                    
                    exp_avg = state['exp_avg']
                    exp_avg_sq = state['exp_avg_sq']
                    
                    state['step'] += 1
                    
                    if group['weight_decay'] != 0:
                        p.mul_(1 - group['lr'] * group['weight_decay'])
                    
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    
                    step_size = group['lr'] / bias_correction1
                    
                    if group['amsgrad']:
                        max_exp_avg_sq = state['max_exp_avg_sq']
                        torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                        denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                    else:
                        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                    
                    p.addcdiv_(exp_avg, denom, value=-step_size)
        
        return loss


# =============================================================================
# LEARNING RATE SCHEDULER
# =============================================================================

class CosineWarmupScheduler:
    """
    Learning rate scheduler with linear warmup and cosine decay.
    
    Common schedule for transformer training.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_step = 0
    
    def step(self):
        """Update learning rate."""
        self.current_step += 1
        lr = self._compute_lr()
        
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = lr * base_lr / self.base_lrs[0]
    
    def _compute_lr(self) -> float:
        if self.current_step < self.warmup_steps:
            # Linear warmup
            return self.base_lrs[0] * self.current_step / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(1.0, progress)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.min_lr + (self.base_lrs[0] - self.min_lr) * cosine_decay


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # For complex parameters, count both real and imag
    complex_params = sum(
        p.numel() * 2 if p.is_complex() else p.numel()
        for p in model.parameters()
    )
    
    return {
        'total': total,
        'trainable': trainable,
        'complex_equivalent': complex_params
    }


def get_parameter_groups(
    model: nn.Module,
    weight_decay: float = 0.01,
    no_decay_patterns: List[str] = None
) -> List[Dict]:
    """
    Create parameter groups with proper weight decay.
    
    Biases, LayerNorm parameters, and embeddings typically shouldn't have weight decay.
    """
    if no_decay_patterns is None:
        no_decay_patterns = ['bias', 'norm', 'embedding']
    
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if any(pattern in name.lower() for pattern in no_decay_patterns):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    return [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]


class GradientClipper:
    """Gradient clipping for complex parameters."""
    
    def __init__(self, max_norm: float = 1.0):
        self.max_norm = max_norm
    
    def __call__(self, parameters):
        """Clip gradients by global norm."""
        parameters = list(filter(lambda p: p.grad is not None, parameters))
        
        total_norm = 0.0
        for p in parameters:
            if p.is_complex():
                # Use magnitude of complex gradient
                param_norm = torch.view_as_real(p.grad).norm() ** 2
            else:
                param_norm = p.grad.norm() ** 2
            total_norm += param_norm
        
        total_norm = total_norm ** 0.5
        
        clip_coef = self.max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for p in parameters:
                p.grad.mul_(clip_coef)
        
        return total_norm


# =============================================================================
# EXAMPLE TRAINING LOOP
# =============================================================================

def train_step(
    model: ComplexTransformerForCausalLM,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scheduler: CosineWarmupScheduler,
    gradient_clipper: GradientClipper,
    gradient_accumulation_steps: int = 1,
    step: int = 0
) -> Dict[str, float]:
    """
    Single training step.
    
    Args:
        model: The model to train
        batch: Dict with 'input_ids' and optionally 'attention_mask', 'labels'
        optimizer: Optimizer
        scheduler: LR scheduler
        gradient_clipper: Gradient clipper
        gradient_accumulation_steps: Number of steps to accumulate gradients
        step: Current step number
    Returns:
        Dict with metrics
    """
    model.train()
    
    # Forward pass
    outputs = model(
        input_ids=batch['input_ids'],
        attention_mask=batch.get('attention_mask'),
        labels=batch.get('labels', batch['input_ids'])
    )
    
    loss = outputs['loss'] / gradient_accumulation_steps
    
    # Backward pass
    loss.backward()
    
    metrics = {'loss': loss.item() * gradient_accumulation_steps}
    
    # Update weights
    if (step + 1) % gradient_accumulation_steps == 0:
        grad_norm = gradient_clipper(model.parameters())
        metrics['grad_norm'] = grad_norm.item()
        
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        metrics['lr'] = optimizer.param_groups[0]['lr']
    
    return metrics


@torch.no_grad()
def evaluate(
    model: ComplexTransformerForCausalLM,
    dataloader,
    max_batches: Optional[int] = None
) -> Dict[str, float]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader with evaluation data
        max_batches: Maximum batches to evaluate (None = all)
    Returns:
        Dict with evaluation metrics
    """
    model.eval()
    
    total_loss = 0.0
    total_tokens = 0
    
    for i, batch in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break
        
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch.get('attention_mask'),
            labels=batch.get('labels', batch['input_ids'])
        )
        
        # Count non-padding tokens
        if 'attention_mask' in batch:
            num_tokens = batch['attention_mask'].sum().item()
        else:
            num_tokens = batch['input_ids'].numel()
        
        total_loss += outputs['loss'].item() * num_tokens
        total_tokens += num_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return {
        'loss': avg_loss,
        'perplexity': perplexity
    }


# =============================================================================
# MAIN / DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Complex-Valued Transformer - Complete Implementation")
    print("=" * 70)
    
    # Create configuration
    config = ComplexTransformerConfig(
        vocab_size=10000,
        embed_dim=256,
        num_heads=8,
        num_layers=4,
        ff_dim=1024,
        max_seq_len=512,
        dropout=0.1,
        use_rope=True,
        activation=ActivationType.MODRELU,
        attention_type=AttentionType.HERMITIAN,
        pre_norm=True
    )
    
    print(f"\nConfiguration:")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Embedding dim: {config.embed_dim}")
    print(f"  Num heads: {config.num_heads}")
    print(f"  Num layers: {config.num_layers}")
    print(f"  FF dim: {config.ff_dim}")
    print(f"  Max seq len: {config.max_seq_len}")
    print(f"  Use RoPE: {config.use_rope}")
    print(f"  Activation: {config.activation.value}")
    print(f"  Attention type: {config.attention_type.value}")
    
    # Create model
    model = ComplexTransformerForCausalLM(config)
    
    # Count parameters
    param_counts = count_parameters(model)
    print(f"\nParameters:")
    print(f"  Total: {param_counts['total']:,}")
    print(f"  Trainable: {param_counts['trainable']:,}")
    print(f"  Complex equivalent: {param_counts['complex_equivalent']:,}")
    
    # Create sample input
    batch_size = 4
    seq_len = 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"\nSample input shape: {input_ids.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    
    print(f"Output logits shape: {outputs['logits'].shape}")
    print(f"Loss: {outputs['loss'].item():.4f}")
    
    # Test generation
    print("\nTesting generation...")
    prompt = torch.randint(0, config.vocab_size, (1, 5))
    generated = model.generate(
        prompt,
        max_new_tokens=20,
        temperature=0.8,
        top_k=50,
        do_sample=True
    )
    print(f"Generated sequence shape: {generated.shape}")
    
    # Test classifier
    print("\nTesting sequence classifier...")
    classifier = ComplexTransformerForSequenceClassification(config, num_classes=5)
    class_outputs = classifier(input_ids)
    print(f"Classification logits shape: {class_outputs['logits'].shape}")
    
    # Test optimizer
    print("\nTesting optimizer...")
    model.train()
    optimizer = ComplexAdamW(
        get_parameter_groups(model, weight_decay=0.01),
        lr=1e-4
    )
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps=100, total_steps=1000)
    gradient_clipper = GradientClipper(max_norm=1.0)
    
    # Training step
    outputs = model(input_ids, labels=input_ids)
    loss = outputs['loss']
    loss.backward()
    
    grad_norm = gradient_clipper(model.parameters())
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    
    print(f"Training step completed!")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Gradient norm: {grad_norm:.4f}")
    print(f"  Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
    
    print("\n" + "=" * 70)
    print("All tests passed successfully!")
    print("=" * 70)
