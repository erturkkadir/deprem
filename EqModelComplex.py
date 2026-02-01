"""
EqModelComplex - Advanced Complex-Valued Transformer for Earthquake Prediction
===============================================================================
Based on complex_transformer_complete.py architecture with:
- Rotary Position Embeddings (RoPE)
- PhaseGatedSiLU activation (smooth gradients + phase-aware gating)
- Hermitian attention scores
- Pre-norm architecture
- Complex dropout

Author: Generated from complex_transformer_complete.py
"""

import torch
import torch.nn as nn
from torch.nn import functional as FN
import math
from typing import Optional, Tuple, List, Dict
from enum import Enum


# =============================================================================
# Configuration Enums
# =============================================================================

class AttentionType(Enum):
    """Attention computation variants for complex-valued attention."""
    HERMITIAN = "hermitian"      # Uses conjugate: Re(Q @ K*) = Re(Q)@Re(K)^T + Im(Q)@Im(K)^T
    REAL_PART = "real_part"      # Uses Re(Q) @ Re(K)^T - Im(Q) @ Im(K)^T
    MAGNITUDE = "magnitude"      # Uses |Q @ K*|


# =============================================================================
# Complex-valued Neural Network Components
# =============================================================================

class ComplexLinear(nn.Module):
    """
    Complex-valued linear layer with magnitude-phase parameterization.

    Key insight: Complex weights W = |W| * exp(i*θ) can be parameterized as:
    - magnitude: |W| (always positive, learned in log-space)
    - phase: θ (learnable rotation angle in [-π, π])

    Why this is better than (real, imag) parameterization:
    1. Rotations (phase changes) are learned DIRECTLY - natural for complex space
    2. Magnitude scaling is DECOUPLED from rotation
    3. Better gradient flow for phase-based patterns
    4. Initialization can use uniform random phases for diverse rotations

    Complex multiplication is TRUE complex: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    This keeps real/imaginary parts naturally coupled.
    """
    def __init__(self, in_features, out_features, bias=True, init_std=None, phase_init='uniform'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Magnitude: log-magnitude for unconstrained optimization
        # Initialized around 1.0 (log(1) ≈ 0) with small variance
        # Using Glorot-like scaling: std = 1/sqrt(fan_in + fan_out)
        mag_std = 0.1  # Small variance around magnitude=1
        mag_init = torch.randn(out_features, in_features) * mag_std
        self.log_magnitude = nn.Parameter(mag_init)

        # Phase: learnable rotation angles
        # Uniform initialization gives diverse rotations at start
        if phase_init == 'uniform':
            phase_init_val = torch.rand(out_features, in_features) * 2 * math.pi - math.pi
        elif phase_init == 'zero':
            phase_init_val = torch.zeros(out_features, in_features)
        else:  # 'small'
            phase_init_val = torch.randn(out_features, in_features) * 0.1
        self.phase = nn.Parameter(phase_init_val)

        if bias:
            # Bias as magnitude + phase
            self.bias_mag = nn.Parameter(torch.zeros(out_features))
            self.bias_phase = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias_mag', None)
            self.register_parameter('bias_phase', None)

    def get_complex_weight(self):
        """Convert magnitude-phase to (real, imag) weight matrices."""
        # W = |W| * exp(i*θ) = |W| * (cos(θ) + i*sin(θ))
        magnitude = torch.exp(self.log_magnitude)
        real_weight = magnitude * torch.cos(self.phase)
        imag_weight = magnitude * torch.sin(self.phase)
        return real_weight, imag_weight

    def forward(self, x):
        """Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i"""
        real_weight, imag_weight = self.get_complex_weight()

        # TRUE complex matrix multiplication
        real = x.real @ real_weight.T - x.imag @ imag_weight.T
        imag = x.real @ imag_weight.T + x.imag @ real_weight.T

        if self.bias_mag is not None:
            # Complex bias: b = |b| * exp(i*θ_b)
            bias_real = self.bias_mag * torch.cos(self.bias_phase)
            bias_imag = self.bias_mag * torch.sin(self.bias_phase)
            real = real + bias_real
            imag = imag + bias_imag

        return torch.complex(real, imag)

    def orthogonality_loss(self):
        """
        Compute soft unitary regularization loss: ||W^H W - I||_F

        Encourages weight matrix to be close to unitary (preserves norms).
        Complex unitary matrices preserve the geometry of complex space,
        which helps with gradient flow and representation learning.

        Returns scalar loss to add to total loss.
        """
        real_weight, imag_weight = self.get_complex_weight()

        # W^H W where W^H is conjugate transpose
        # (W^H W)_ij = sum_k (W_ki^* W_kj) = sum_k (W_ki_r - i*W_ki_i)(W_kj_r + i*W_kj_i)
        # Real part: sum_k (W_ki_r * W_kj_r + W_ki_i * W_kj_i)
        # Imag part: sum_k (W_ki_r * W_kj_i - W_ki_i * W_kj_r)

        WH_W_real = real_weight.T @ real_weight + imag_weight.T @ imag_weight
        WH_W_imag = real_weight.T @ imag_weight - imag_weight.T @ real_weight

        # Want WH_W ≈ I (identity)
        # ||WH_W - I||_F^2 = ||WH_W_real - I||_F^2 + ||WH_W_imag||_F^2
        n = min(self.in_features, self.out_features)
        identity = torch.eye(self.in_features, device=real_weight.device)

        loss_real = ((WH_W_real - identity) ** 2).sum()
        loss_imag = (WH_W_imag ** 2).sum()

        return (loss_real + loss_imag) / (self.in_features * self.out_features)


class ComplexEmbedding(nn.Module):
    """
    Complex-valued embedding layer with magnitude-phase parameterization.

    Each embedding vector is: e = |e| * exp(i*θ)

    This is natural for features like:
    - Month (θ can represent position in yearly cycle)
    - Geographic coordinates (θ encodes angular position)
    - Time intervals (magnitude = intensity, phase = periodicity)

    Magnitude is learned in log-space (always positive).
    Phase is learned directly (full rotation freedom).
    """
    def __init__(self, num_embeddings, embedding_dim, init_std=0.02):
        super().__init__()

        # Magnitude embedding (log-scale)
        self.log_magnitude = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.normal_(self.log_magnitude.weight, mean=0.0, std=0.1)

        # Phase embedding (full circle [-π, π])
        self.phase = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.phase.weight, -math.pi, math.pi)

    def forward(self, x):
        magnitude = torch.exp(self.log_magnitude(x))
        phase = self.phase(x)
        return torch.complex(magnitude * torch.cos(phase), magnitude * torch.sin(phase))


class ComplexLayerNorm(nn.Module):
    """
    Layer normalization for complex tensors using complex variance.
    Var(z) = E[|z - μ|²] where μ = E[z] is complex mean
    """
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma_real = nn.Parameter(torch.ones(features))
        self.gamma_imag = nn.Parameter(torch.zeros(features))
        self.beta_real = nn.Parameter(torch.zeros(features))
        self.beta_imag = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        # Compute complex mean
        mean_real = x.real.mean(-1, keepdim=True)
        mean_imag = x.imag.mean(-1, keepdim=True)

        # Center the data
        x_centered_real = x.real - mean_real
        x_centered_imag = x.imag - mean_imag

        # Compute variance: E[|x - μ|²]
        variance = (x_centered_real ** 2 + x_centered_imag ** 2).mean(-1, keepdim=True)
        std = torch.sqrt(variance + self.eps)

        # Normalize
        norm_real = x_centered_real / std
        norm_imag = x_centered_imag / std

        # Apply complex affine transformation
        out_real = norm_real * self.gamma_real - norm_imag * self.gamma_imag + self.beta_real
        out_imag = norm_real * self.gamma_imag + norm_imag * self.gamma_real + self.beta_imag

        return torch.complex(out_real, out_imag)


class ComplexRMSNorm(nn.Module):
    """
    RMS Normalization for complex tensors.

    Simpler than LayerNorm - only normalizes by RMS, no centering.
    RMS(z) = sqrt(mean(|z|²))

    Full Complex: Uses magnitude for normalization while applying
    complex affine transformation that preserves the coupling.
    """
    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps

        # Complex scale parameter (magnitude-phase style)
        self.gamma_real = nn.Parameter(torch.ones(normalized_shape))
        self.gamma_imag = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS of complex magnitude: sqrt(mean(|z|²))
        rms = torch.sqrt((x.real ** 2 + x.imag ** 2).mean(dim=-1, keepdim=True) + self.eps)

        # Normalize by RMS (no centering)
        norm_real = x.real / rms
        norm_imag = x.imag / rms

        # Apply complex scale: (norm_r + i*norm_i) * (γ_r + i*γ_i)
        # = (norm_r*γ_r - norm_i*γ_i) + i*(norm_r*γ_i + norm_i*γ_r)
        out_real = norm_real * self.gamma_real - norm_imag * self.gamma_imag
        out_imag = norm_real * self.gamma_imag + norm_imag * self.gamma_real

        return torch.complex(out_real, out_imag)


class ComplexSiLU(nn.Module):
    """
    Complex SiLU (Swish): z * sigmoid(|z|)

    The KEY insight: operates on z as a whole, not splitting real/imag.
    Uses magnitude for gating while preserving the full complex number.
    This maintains the coupling between real and imaginary parts.
    """
    def forward(self, z):
        gate = torch.sigmoid(z.abs())
        return z * gate


class Cardioid(nn.Module):
    """
    Cardioid activation: smooth, bounded, phase-dependent gating.
    cardioid(z) = 0.5 * (1 + cos(angle(z))) * z

    This keeps z intact as a complex unit while smoothly gating based on phase.
    Values with phase near 0 (positive real) are preserved,
    values with phase near ±π (negative real) are suppressed.

    Add learnable phase offset for flexibility.
    """
    def __init__(self, num_features):
        super().__init__()
        self.phase_offset = nn.Parameter(torch.zeros(num_features))

    def forward(self, z):
        phase = z.angle()
        # Smooth phase-dependent scaling (keeps z as a unit)
        scale = 0.5 * (1 + torch.cos(phase - self.phase_offset))
        return scale * z


class ComplexGatedActivation(nn.Module):
    """
    Combined activation: Cardioid + learnable magnitude bias.

    This is the best of both worlds:
    - Cardioid provides smooth phase-dependent gating (z stays coupled)
    - Magnitude bias allows learning activation thresholds
    - No dead neurons (unlike ModReLU)

    Formula: scale(phase) * z * sigmoid(|z| + bias)
    """
    def __init__(self, num_features):
        super().__init__()
        self.mag_bias = nn.Parameter(torch.zeros(num_features))
        self.phase_offset = nn.Parameter(torch.zeros(num_features))

    def forward(self, z):
        # Magnitude-based gating (smooth, no dead neurons)
        mag_gate = torch.sigmoid(z.abs() + self.mag_bias)

        # Phase-based gating (cardioid-style, keeps z coupled)
        phase = z.angle()
        phase_gate = 0.5 * (1 + torch.cos(phase - self.phase_offset))

        # Combined: both gates applied to z as a whole
        return z * mag_gate * phase_gate


class ComplexGELU(nn.Module):
    """Complex GELU: applies GELU separately to real and imaginary parts."""
    def forward(self, z):
        return torch.complex(FN.gelu(z.real), FN.gelu(z.imag))


class ModReLU(nn.Module):
    """
    Modulus ReLU: applies ReLU to magnitude, preserves phase.
    modReLU(z) = ReLU(|z| + b) * e^(i * angle(z))

    KEY for Full Complex: This is phase-preserving - the direction in complex plane
    is maintained while only the magnitude is gated. The learnable bias b allows
    the network to learn appropriate activation thresholds.

    Better gradient flow than CReLU because:
    1. Phase information flows through unchanged
    2. Magnitude gets smooth ReLU treatment
    3. No discontinuities in phase
    """
    def __init__(self, num_features: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        magnitude = z.abs()
        phase = z.angle()

        # ReLU on (magnitude + bias) - allows network to learn threshold
        activated_magnitude = FN.relu(magnitude + self.bias)

        # Reconstruct with ORIGINAL phase - this is the key!
        # Using exp(1j * phase) is true complex reconstruction
        return activated_magnitude * torch.exp(1j * phase)


class ZReLU(nn.Module):
    """
    Z-ReLU: keeps values only in the first quadrant (positive real and imag).
    ZReLU(z) = z if Re(z) > 0 and Im(z) > 0, else 0

    More restrictive but maintains holomorphic-like properties in active region.
    The complex number is kept as a WHOLE - either passed through or zeroed.
    """
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mask = (z.real > 0) & (z.imag > 0)
        return z * mask


class CReLU(nn.Module):
    """
    Complex ReLU: applies ReLU separately to real and imaginary parts.
    CReLU(z) = ReLU(Re(z)) + i * ReLU(Im(z))

    WARNING: This DOES split real/imag - use ModReLU for true complex behavior.
    Included for completeness but ModReLU is preferred for Full Complex.
    """
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.complex(FN.relu(z.real), FN.relu(z.imag))


class ComplexDropout(nn.Module):
    """Dropout for complex tensors - drops both real and imag together."""
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, z):
        if not self.training or self.p == 0:
            return z
        mask = torch.bernoulli(
            torch.full(z.shape, 1 - self.p, device=z.device, dtype=z.real.dtype)
        )
        mask = mask / (1 - self.p)
        return z * mask


# =============================================================================
# Rotary Position Embedding (RoPE)
# =============================================================================

class ComplexRotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for complex-valued attention.
    Encodes positions as rotations - naturally expressed as multiplication by e^(i*θ).
    """
    def __init__(self, dim, max_seq_len=2048, theta=10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        positions = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq)
        self.register_buffer('cos_cached', freqs.cos(), persistent=False)
        self.register_buffer('sin_cached', freqs.sin(), persistent=False)

    def forward(self, seq_len, device):
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            self.max_seq_len = seq_len
        return (
            self.cos_cached[:seq_len].to(device),
            self.sin_cached[:seq_len].to(device)
        )


def apply_complex_rotary_embedding(q, k, cos, sin):
    """Apply rotary embeddings to complex Q and K tensors.

    Handles odd head dimensions by only applying RoPE to even portion.
    """
    batch_size, num_heads, seq_len, head_dim = q.shape

    # RoPE only applies to pairs of dimensions
    # For odd head_dim, we apply to head_dim-1 dims and leave last dim unchanged
    rope_dim = (head_dim // 2) * 2  # Largest even number <= head_dim

    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim/2]
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)

    # Truncate cos/sin to match rope_dim/2
    half_rope_dim = rope_dim // 2
    cos = cos[..., :half_rope_dim]
    sin = sin[..., :half_rope_dim]

    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)

    def apply_rotation(x, cos, sin):
        # Split into rotated and pass-through portions
        x_rope = x[..., :rope_dim]
        x_pass = x[..., rope_dim:] if rope_dim < head_dim else None

        cos_full = torch.cat([cos, cos], dim=-1)
        sin_full = torch.cat([sin, sin], dim=-1)

        x_rope_real_rot = x_rope.real * cos_full + rotate_half(x_rope.real) * sin_full
        x_rope_imag_rot = x_rope.imag * cos_full + rotate_half(x_rope.imag) * sin_full
        x_rope_rot = torch.complex(x_rope_real_rot, x_rope_imag_rot)

        if x_pass is not None:
            return torch.cat([x_rope_rot, x_pass], dim=-1)
        return x_rope_rot

    return apply_rotation(q, cos, sin), apply_rotation(k, cos, sin)


# =============================================================================
# Complex Multi-Head Attention with RoPE
# =============================================================================

class SpatialAttentionBias(nn.Module):
    """
    Learnable spatial attention bias based on geographic distance between earthquakes.

    Nearby earthquakes should attend more strongly to each other (aftershock sequences,
    fault propagation). This adds a distance-dependent bias to attention scores,
    making every attention head spatially aware.

    Uses log-distance with learnable scaling per head - captures both local (km-scale)
    and regional (1000km-scale) spatial relationships.
    """
    def __init__(self, num_heads, max_distance_deg=180.0):
        super().__init__()
        self.num_heads = num_heads
        self.max_distance_deg = max_distance_deg

        # Per-head learnable parameters:
        # distance_scale: how strongly distance affects attention (negative = nearby preferred)
        # distance_offset: baseline bias
        self.distance_scale = nn.Parameter(torch.randn(num_heads) * 0.1 - 0.5)  # Init slightly negative
        self.distance_offset = nn.Parameter(torch.zeros(num_heads))

    def forward(self, lat, lon, seq_len):
        """
        Compute pairwise spatial attention bias for the sequence.

        Args:
            lat: [B, T] latitude values (0-180 encoded)
            lon: [B, T] longitude values (0-360 encoded)
            seq_len: actual sequence length

        Returns:
            bias: [B, num_heads, T, T] attention bias to add to scores
        """
        B = lat.shape[0]
        T = seq_len

        # Convert to radians for great-circle distance
        lat_rad = ((lat[:, :T].float() - 90) / 180.0) * math.pi
        lon_rad = (lon[:, :T].float() / 360.0) * 2 * math.pi

        # Compute pairwise great-circle distance (haversine)
        # [B, T, 1] vs [B, 1, T] broadcasting
        lat1 = lat_rad.unsqueeze(2)  # [B, T, 1]
        lat2 = lat_rad.unsqueeze(1)  # [B, 1, T]
        lon1 = lon_rad.unsqueeze(2)
        lon2 = lon_rad.unsqueeze(1)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = torch.sin(dlat/2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon/2)**2
        # Angular distance in degrees (0 to 180)
        distance_deg = 2 * torch.asin(torch.sqrt(torch.clamp(a, 0, 1))) * (180.0 / math.pi)

        # Log-distance features (smooth, handles zero distance)
        log_dist = torch.log1p(distance_deg)  # [B, T, T]

        # Per-head bias: scale * log_distance + offset
        # [B, T, T] -> [B, 1, T, T] * [num_heads] -> [B, num_heads, T, T]
        log_dist = log_dist.unsqueeze(1)  # [B, 1, T, T]
        scale = self.distance_scale.view(1, self.num_heads, 1, 1)
        offset = self.distance_offset.view(1, self.num_heads, 1, 1)

        bias = scale * log_dist + offset

        return bias


class ComplexMultiHeadAttention(nn.Module):
    """
    Multi-head attention with complex Q, K, V, Rotary Position Embeddings,
    and Spatial Attention Bias for geographic distance awareness.

    Supports three attention score computation methods:
    - HERMITIAN: Re(Q @ K*) = Re(Q)@Re(K)^T + Im(Q)@Im(K)^T (standard complex inner product)
    - REAL_PART: Re(Q @ K^T) = Re(Q)@Re(K)^T - Im(Q)@Im(K)^T
    - MAGNITUDE: |Q @ K*| (magnitude of complex product)

    HERMITIAN is recommended as it's the proper complex inner product that
    respects the complex structure (preserves coupling between real/imag).

    The Spatial Attention Bias adds a learnable distance-dependent term to
    attention scores, making nearby earthquakes attend more strongly to each other.
    """
    def __init__(self, embed_dim, num_heads, block_size, dropout=0.1, use_rope=True,
                 use_spatial_bias=True, attention_type: AttentionType = AttentionType.HERMITIAN):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.embed_dim = embed_dim
        self.use_rope = use_rope
        self.use_spatial_bias = use_spatial_bias
        self.attention_type = attention_type

        self.q_proj = ComplexLinear(embed_dim, embed_dim, bias=False)
        self.k_proj = ComplexLinear(embed_dim, embed_dim, bias=False)
        self.v_proj = ComplexLinear(embed_dim, embed_dim, bias=False)
        self.out_proj = ComplexLinear(embed_dim, embed_dim)

        if use_rope:
            self.rotary_emb = ComplexRotaryEmbedding(self.head_dim, block_size)
        else:
            self.rotary_emb = None

        # Spatial attention bias
        if use_spatial_bias:
            self.spatial_bias = SpatialAttentionBias(num_heads)
        else:
            self.spatial_bias = None

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.resid_dropout = ComplexDropout(dropout)

    def forward(self, x, lat=None, lon=None):
        B, L, D = x.shape

        Q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE if enabled
        if self.rotary_emb is not None:
            cos, sin = self.rotary_emb(L, Q.device)
            Q, K = apply_complex_rotary_embedding(Q, K, cos, sin)

        # Compute attention scores based on attention type
        attn_scores = self._compute_attention_scores(Q, K)

    def _compute_attention_scores(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """
        Compute attention scores based on configured attention type.

        Full Complex: All three methods preserve the coupling between real and
        imaginary parts - they just define different similarity measures.

        Args:
            Q: [batch, heads, seq_len, head_dim] complex
            K: [batch, heads, seq_len, head_dim] complex
        Returns:
            Real-valued attention scores [batch, heads, seq_len, seq_len]
        """
        if self.attention_type == AttentionType.HERMITIAN:
            # Hermitian inner product: <q, k> = sum(q * conj(k))
            # Re(<q,k>) = Re(q)@Re(k)^T + Im(q)@Im(k)^T
            # This is the STANDARD complex inner product - recommended!
            scores = (
                torch.matmul(Q.real, K.real.transpose(-2, -1)) +
                torch.matmul(Q.imag, K.imag.transpose(-2, -1))
            )

        elif self.attention_type == AttentionType.REAL_PART:
            # Real part of regular product (non-conjugate)
            # Re(q @ k^T) = Re(q)@Re(k)^T - Im(q)@Im(k)^T
            scores = (
                torch.matmul(Q.real, K.real.transpose(-2, -1)) -
                torch.matmul(Q.imag, K.imag.transpose(-2, -1))
            )

        elif self.attention_type == AttentionType.MAGNITUDE:
            # Magnitude of complex product |Q @ K*|
            # More expensive but captures full complex relationship
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

    def forward(self, x, lat=None, lon=None):
        B, L, D = x.shape

        Q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE if enabled
        if self.rotary_emb is not None:
            cos, sin = self.rotary_emb(L, Q.device)
            Q, K = apply_complex_rotary_embedding(Q, K, cos, sin)

        # Compute attention scores based on attention type
        attn_scores = self._compute_attention_scores(Q, K)

        # Add spatial attention bias (geographic distance awareness)
        if self.spatial_bias is not None and lat is not None and lon is not None:
            spatial_bias = self.spatial_bias(lat, lon, L)
            attn_scores = attn_scores + spatial_bias

        # Causal mask
        attn_scores = attn_scores.masked_fill(self.tril[:L, :L] == 0, float('-inf'))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply to complex values
        out_real = attn_weights @ V.real
        out_imag = attn_weights @ V.imag
        out = torch.complex(out_real, out_imag)

        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(out)
        out = self.resid_dropout(out)

        return out


# =============================================================================
# Complex Feed-Forward Network
# =============================================================================

class ComplexGatedFeedForward(nn.Module):
    """
    Gated feed-forward network (SwiGLU-style) for complex transformers.

    GLU(x) = (x @ W_gate * activation) ⊗ (x @ W_up) @ W_down

    The KEY insight: use TRUE complex multiplication (⊗) for gating.
    This keeps real and imaginary parts coupled together, which is
    essential for complex networks to learn meaningful representations.

    Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    This naturally couples the components.
    """
    def __init__(self, embed_dim, dropout=0.1, expansion=4):
        super().__init__()
        hidden_dim = expansion * embed_dim

        # Three projections for gated architecture
        self.gate_proj = ComplexLinear(embed_dim, hidden_dim, bias=False)
        self.up_proj = ComplexLinear(embed_dim, hidden_dim, bias=False)
        self.down_proj = ComplexLinear(hidden_dim, embed_dim, bias=False)

        self.dropout = ComplexDropout(dropout)

    def forward(self, x):
        # Gate projection with ComplexSiLU activation
        gate = self.gate_proj(x)
        gate_activated = gate * torch.sigmoid(gate.abs())  # ComplexSiLU

        # Up projection (no activation)
        up = self.up_proj(x)

        # TRUE complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        # This is the key - keeps real/imag coupled!
        hidden_real = gate_activated.real * up.real - gate_activated.imag * up.imag
        hidden_imag = gate_activated.real * up.imag + gate_activated.imag * up.real
        hidden = torch.complex(hidden_real, hidden_imag)

        # Down projection
        output = self.down_proj(hidden)
        return self.dropout(output)


class ComplexFeedForward(nn.Module):
    """
    Standard feed-forward with ComplexGatedActivation.
    Simpler than gated version but still uses true complex operations.
    """
    def __init__(self, embed_dim, dropout=0.1, expansion=4):
        super().__init__()
        hidden_dim = expansion * embed_dim
        self.fc1 = ComplexLinear(embed_dim, hidden_dim)
        self.fc2 = ComplexLinear(hidden_dim, embed_dim)
        self.activation = ComplexGatedActivation(hidden_dim)
        self.dropout = ComplexDropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# =============================================================================
# Complex Transformer Block
# =============================================================================

class ComplexTransformerBlock(nn.Module):
    """
    Complex transformer block with pre-norm architecture and spatial attention bias.

    Uses ComplexGatedFeedForward (SwiGLU-style) with TRUE complex multiplication
    to keep real/imaginary parts coupled throughout the forward pass.
    """
    def __init__(self, embed_dim, num_heads, block_size, dropout=0.1, use_rope=True, use_spatial_bias=True):
        super().__init__()
        self.norm1 = ComplexLayerNorm(embed_dim)
        self.attn = ComplexMultiHeadAttention(embed_dim, num_heads, block_size, dropout, use_rope, use_spatial_bias)
        self.norm2 = ComplexLayerNorm(embed_dim)
        # Use gated FFN with true complex multiplication
        self.ffwd = ComplexGatedFeedForward(embed_dim, dropout)
        self.dropout = ComplexDropout(dropout)

    def forward(self, x, lat=None, lon=None):
        # Pre-norm architecture: norm -> sublayer -> residual
        # Pass lat/lon to attention for spatial bias
        x = x + self.attn(self.norm1(x), lat=lat, lon=lon)
        x = x + self.dropout(self.ffwd(self.norm2(x)))
        return x


# =============================================================================
# Geographic Positional Encoding (GPE) - Enhanced Location Awareness
# =============================================================================

class SphericalHarmonicEncoding(nn.Module):
    """
    Encode lat/lon using spherical harmonics - natural basis for spherical surfaces.
    This captures global patterns and tectonic relationships better than simple embeddings.

    Uses complex exponentials: e^(i*m*lon) * P_l^m(cos(lat))
    where P_l^m are associated Legendre polynomials.
    """
    def __init__(self, max_degree=8, embed_dim=64):
        super().__init__()
        self.max_degree = max_degree
        self.embed_dim = embed_dim

        # Number of spherical harmonic coefficients: sum(2l+1) for l=0 to max_degree
        self.n_harmonics = (max_degree + 1) ** 2

        # Learnable weights for each harmonic
        self.harmonic_weights_real = nn.Parameter(torch.randn(self.n_harmonics, embed_dim) * 0.02)
        self.harmonic_weights_imag = nn.Parameter(torch.randn(self.n_harmonics, embed_dim) * 0.02)

    def _compute_legendre(self, cos_lat, l, m):
        """Compute associated Legendre polynomial P_l^m(x) using recurrence."""
        x = cos_lat
        abs_m = abs(m)

        # Start with P_m^m
        pmm = torch.ones_like(x)
        if abs_m > 0:
            somx2 = torch.sqrt((1 - x) * (1 + x))
            fact = 1.0
            for i in range(1, abs_m + 1):
                pmm = pmm * (-fact * somx2)
                fact += 2.0

        if l == abs_m:
            return pmm

        # Compute P_{m+1}^m
        pmmp1 = x * (2 * abs_m + 1) * pmm
        if l == abs_m + 1:
            return pmmp1

        # Use recurrence for higher l
        pll = pmmp1
        for ll in range(abs_m + 2, l + 1):
            pll = ((2 * ll - 1) * x * pmmp1 - (ll + abs_m - 1) * pmm) / (ll - abs_m)
            pmm = pmmp1
            pmmp1 = pll

        return pll

    def forward(self, lat_encoded, lon_encoded):
        """
        Args:
            lat_encoded: [B, T] latitude (0-180, where 90 is equator)
            lon_encoded: [B, T] longitude (0-360)
        Returns:
            Complex tensor [B, T, embed_dim]
        """
        B, T = lat_encoded.shape
        device = lat_encoded.device

        # Convert to radians (lat: 0-180 -> 0-pi, lon: 0-360 -> 0-2pi)
        lat_rad = (lat_encoded.float() / 180.0) * math.pi  # colatitude
        lon_rad = (lon_encoded.float() / 360.0) * 2 * math.pi

        cos_lat = torch.cos(lat_rad)

        harmonics_real = []
        harmonics_imag = []

        idx = 0
        for l in range(self.max_degree + 1):
            for m in range(-l, l + 1):
                # Y_l^m = P_l^|m|(cos(lat)) * e^(i*m*lon)
                plm = self._compute_legendre(cos_lat, l, abs(m))

                # Complex exponential for longitude
                phase = m * lon_rad
                ylm_real = plm * torch.cos(phase)
                ylm_imag = plm * torch.sin(phase)

                harmonics_real.append(ylm_real)
                harmonics_imag.append(ylm_imag)
                idx += 1

        # Stack harmonics: [B, T, n_harmonics]
        harmonics_real = torch.stack(harmonics_real, dim=-1)
        harmonics_imag = torch.stack(harmonics_imag, dim=-1)

        # Project to embedding dimension
        out_real = harmonics_real @ self.harmonic_weights_real - harmonics_imag @ self.harmonic_weights_imag
        out_imag = harmonics_real @ self.harmonic_weights_imag + harmonics_imag @ self.harmonic_weights_real

        return torch.complex(out_real, out_imag)


class FourierLocationEncoding(nn.Module):
    """
    Multi-scale Fourier features for geographic coordinates.
    Captures patterns at different spatial scales (local faults to global plates).
    """
    def __init__(self, embed_dim=64, num_frequencies=16):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_frequencies = num_frequencies

        # Frequencies for multi-scale encoding (log-spaced for different scales)
        # Small frequencies = large-scale patterns (plates)
        # Large frequencies = small-scale patterns (local faults)
        frequencies = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
        self.register_buffer('frequencies', frequencies)

        # Input: 2 coords * 2 (sin/cos) * num_frequencies = 4 * num_frequencies
        fourier_dim = 4 * num_frequencies

        # Complex projection
        self.proj_real = nn.Linear(fourier_dim, embed_dim)
        self.proj_imag = nn.Linear(fourier_dim, embed_dim)

        nn.init.normal_(self.proj_real.weight, std=0.02)
        nn.init.normal_(self.proj_imag.weight, std=0.02)

    def forward(self, lat_encoded, lon_encoded):
        """
        Args:
            lat_encoded: [B, T] latitude (0-180)
            lon_encoded: [B, T] longitude (0-360)
        Returns:
            Complex tensor [B, T, embed_dim]
        """
        # Normalize to [-1, 1]
        lat_norm = (lat_encoded.float() - 90) / 90.0  # -1 to 1
        lon_norm = (lon_encoded.float() - 180) / 180.0  # -1 to 1

        # Apply frequencies
        lat_scaled = lat_norm.unsqueeze(-1) * self.frequencies * math.pi
        lon_scaled = lon_norm.unsqueeze(-1) * self.frequencies * math.pi

        # Fourier features: [B, T, 4 * num_frequencies]
        features = torch.cat([
            torch.sin(lat_scaled),
            torch.cos(lat_scaled),
            torch.sin(lon_scaled),
            torch.cos(lon_scaled)
        ], dim=-1)

        # Project to complex embedding
        out_real = self.proj_real(features)
        out_imag = self.proj_imag(features)

        return torch.complex(out_real, out_imag)


class TectonicZoneEncoding(nn.Module):
    """
    Learn embeddings for major tectonic zones / earthquake-prone regions.
    Uses a soft assignment based on location to blend zone embeddings.
    """
    def __init__(self, embed_dim=64, num_zones=32):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_zones = num_zones

        # Learnable zone centers (lat, lon)
        # Initialize with known earthquake-prone regions
        self.zone_centers = nn.Parameter(self._init_zone_centers())

        # Learnable zone scales (how wide each zone is)
        self.zone_scales = nn.Parameter(torch.ones(num_zones) * 0.1)

        # Complex zone embeddings
        self.zone_embed_real = nn.Parameter(torch.randn(num_zones, embed_dim) * 0.02)
        self.zone_embed_imag = nn.Parameter(torch.randn(num_zones, embed_dim) * 0.02)

    def _init_zone_centers(self):
        """Initialize zone centers near known tectonic boundaries."""
        # Major earthquake zones (normalized: lat 0-1, lon 0-1)
        known_zones = torch.tensor([
            [0.25, 0.42],   # Pacific Ring of Fire - Japan
            [0.28, 0.33],   # Pacific Ring of Fire - Philippines
            [0.55, 0.65],   # Pacific Ring of Fire - Chile
            [0.50, 0.67],   # Pacific Ring of Fire - Peru
            [0.45, 0.70],   # Central America
            [0.40, 0.69],   # Mexico
            [0.38, 0.68],   # California
            [0.30, 0.60],   # Alaska/Aleutians
            [0.22, 0.35],   # Indonesia
            [0.25, 0.30],   # Sumatra
            [0.35, 0.22],   # Mediterranean
            [0.35, 0.15],   # Turkey/Iran
            [0.30, 0.20],   # Himalayas/Nepal
            [0.40, 0.50],   # Mid-Atlantic Ridge
            [0.20, 0.45],   # New Zealand
            [0.15, 0.40],   # Fiji/Tonga
        ], dtype=torch.float32)

        # Fill remaining zones with random positions
        n_known = known_zones.shape[0]
        n_random = self.num_zones - n_known
        random_zones = torch.rand(n_random, 2)

        return torch.cat([known_zones, random_zones], dim=0)

    def forward(self, lat_encoded, lon_encoded):
        """
        Args:
            lat_encoded: [B, T] latitude (0-180)
            lon_encoded: [B, T] longitude (0-360)
        Returns:
            Complex tensor [B, T, embed_dim]
        """
        B, T = lat_encoded.shape

        # Normalize coordinates to [0, 1]
        lat_norm = lat_encoded.float() / 180.0
        lon_norm = lon_encoded.float() / 360.0

        # Stack coordinates: [B, T, 2]
        coords = torch.stack([lat_norm, lon_norm], dim=-1)

        # Compute distances to all zone centers: [B, T, num_zones]
        # Using great circle approximation on normalized coords
        centers = self.zone_centers.unsqueeze(0).unsqueeze(0)  # [1, 1, num_zones, 2]
        coords_exp = coords.unsqueeze(2)  # [B, T, 1, 2]

        # Squared Euclidean distance (approximation)
        dist_sq = ((coords_exp - centers) ** 2).sum(dim=-1)  # [B, T, num_zones]

        # Soft assignment using scaled distances
        scales = FN.softplus(self.zone_scales).unsqueeze(0).unsqueeze(0)
        weights = FN.softmax(-dist_sq / (scales ** 2), dim=-1)  # [B, T, num_zones]

        # Weighted combination of zone embeddings
        out_real = weights @ self.zone_embed_real
        out_imag = weights @ self.zone_embed_imag

        return torch.complex(out_real, out_imag)


class RelativeLocationEncoding(nn.Module):
    """
    Encode relative positions between earthquakes in a sequence.
    Helps model learn spatial propagation patterns (aftershock sequences, etc.)
    """
    def __init__(self, embed_dim=64, max_distance_km=2000):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_distance_km = max_distance_km

        # Learnable distance encoding
        self.dist_proj_real = nn.Linear(1, embed_dim)
        self.dist_proj_imag = nn.Linear(1, embed_dim)

        # Learnable bearing encoding (direction)
        self.bearing_embed_dim = embed_dim

        nn.init.normal_(self.dist_proj_real.weight, std=0.02)
        nn.init.normal_(self.dist_proj_imag.weight, std=0.02)

    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """Compute haversine distance in km between two points."""
        R = 6371.0  # Earth radius in km

        # Convert encoded values to radians
        lat1_rad = ((lat1.float() - 90) / 180.0) * math.pi
        lat2_rad = ((lat2.float() - 90) / 180.0) * math.pi
        lon1_rad = (lon1.float() / 360.0) * 2 * math.pi
        lon2_rad = (lon2.float() / 360.0) * 2 * math.pi

        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = torch.sin(dlat/2)**2 + torch.cos(lat1_rad) * torch.cos(lat2_rad) * torch.sin(dlon/2)**2
        c = 2 * torch.asin(torch.sqrt(torch.clamp(a, 0, 1)))

        return R * c

    def _bearing(self, lat1, lon1, lat2, lon2):
        """Compute bearing from point 1 to point 2."""
        lat1_rad = ((lat1.float() - 90) / 180.0) * math.pi
        lat2_rad = ((lat2.float() - 90) / 180.0) * math.pi
        lon1_rad = (lon1.float() / 360.0) * 2 * math.pi
        lon2_rad = (lon2.float() / 360.0) * 2 * math.pi

        dlon = lon2_rad - lon1_rad

        x = torch.sin(dlon) * torch.cos(lat2_rad)
        y = torch.cos(lat1_rad) * torch.sin(lat2_rad) - torch.sin(lat1_rad) * torch.cos(lat2_rad) * torch.cos(dlon)

        return torch.atan2(x, y)

    def forward(self, lat_encoded, lon_encoded):
        """
        Compute relative location encoding from previous earthquake to current.

        Args:
            lat_encoded: [B, T] latitude (0-180)
            lon_encoded: [B, T] longitude (0-360)
        Returns:
            Complex tensor [B, T, embed_dim]
        """
        B, T = lat_encoded.shape
        device = lat_encoded.device

        # Initialize output
        out_real = torch.zeros(B, T, self.embed_dim, device=device)
        out_imag = torch.zeros(B, T, self.embed_dim, device=device)

        if T > 1:
            # Compute distances from previous earthquake
            lat_prev = lat_encoded[:, :-1]
            lat_curr = lat_encoded[:, 1:]
            lon_prev = lon_encoded[:, :-1]
            lon_curr = lon_encoded[:, 1:]

            distances = self._haversine_distance(lat_prev, lon_prev, lat_curr, lon_curr)
            bearings = self._bearing(lat_prev, lon_prev, lat_curr, lon_curr)

            # Normalize distance
            dist_norm = (distances / self.max_distance_km).unsqueeze(-1).clamp(0, 1)

            # Distance encoding
            dist_enc_real = self.dist_proj_real(dist_norm)
            dist_enc_imag = self.dist_proj_imag(dist_norm)

            # Bearing as phase rotation
            bearing_phase = bearings.unsqueeze(-1).expand(-1, -1, self.embed_dim)

            # Combine distance magnitude with bearing direction
            out_real[:, 1:] = dist_enc_real * torch.cos(bearing_phase) - dist_enc_imag * torch.sin(bearing_phase)
            out_imag[:, 1:] = dist_enc_real * torch.sin(bearing_phase) + dist_enc_imag * torch.cos(bearing_phase)

        return torch.complex(out_real, out_imag)


class GeographicPositionalEncoding(nn.Module):
    """
    Combined Geographic Positional Encoding.
    Merges multiple location encoding strategies for rich spatial representation.
    """
    def __init__(self, embed_dim=168, spherical_degree=6, num_frequencies=12, num_zones=24):
        super().__init__()
        self.embed_dim = embed_dim

        # Divide embedding dimension among different encodings
        self.spherical_dim = embed_dim // 4
        self.fourier_dim = embed_dim // 4
        self.zone_dim = embed_dim // 4
        self.relative_dim = embed_dim - 3 * (embed_dim // 4)

        # Sub-encoders
        self.spherical = SphericalHarmonicEncoding(max_degree=spherical_degree, embed_dim=self.spherical_dim)
        self.fourier = FourierLocationEncoding(embed_dim=self.fourier_dim, num_frequencies=num_frequencies)
        self.tectonic = TectonicZoneEncoding(embed_dim=self.zone_dim, num_zones=num_zones)
        self.relative = RelativeLocationEncoding(embed_dim=self.relative_dim)

        # Learnable combination weights
        self.combine_weights = nn.Parameter(torch.ones(4) / 4)

        # Final projection to ensure proper dimension
        self.out_proj = ComplexLinear(embed_dim, embed_dim)

    def forward(self, lat_encoded, lon_encoded):
        """
        Args:
            lat_encoded: [B, T] latitude (0-180)
            lon_encoded: [B, T] longitude (0-360)
        Returns:
            Complex tensor [B, T, embed_dim]
        """
        # Compute all encodings
        spherical_enc = self.spherical(lat_encoded, lon_encoded)
        fourier_enc = self.fourier(lat_encoded, lon_encoded)
        tectonic_enc = self.tectonic(lat_encoded, lon_encoded)
        relative_enc = self.relative(lat_encoded, lon_encoded)

        # Normalize combination weights
        weights = FN.softmax(self.combine_weights, dim=0)

        # Concatenate all encodings
        combined = torch.cat([
            spherical_enc * weights[0],
            fourier_enc * weights[1],
            tectonic_enc * weights[2],
            relative_enc * weights[3]
        ], dim=-1)

        # Project to output dimension
        return self.out_proj(combined)


# =============================================================================
# Custom Embedding for Earthquake Features (Enhanced with GPE)
# =============================================================================

class LogScaleDtEncoding(nn.Module):
    """
    Log-scale encoding for earthquake inter-event times (dt).

    Earthquake inter-event times follow a log-normal distribution:
    most events cluster at short intervals (1-30 min) with a long tail.
    Standard uniform embedding wastes capacity on rare long intervals.

    This encoding creates multi-resolution features in log-space,
    giving fine resolution for short intervals and coarse for long ones.
    """
    def __init__(self, embed_dim, max_dt=720, num_scales=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_dt = max_dt
        self.num_scales = num_scales

        # Log-scale frequencies (captures patterns at different time scales)
        # Small freq = long-term patterns, large freq = short-term patterns
        log_freqs = 2.0 ** torch.linspace(0, num_scales - 1, num_scales)
        self.register_buffer('log_freqs', log_freqs)

        # Features: log(1+dt), dt/max_dt, plus sin/cos at each frequency scale
        input_dim = 2 + 2 * num_scales  # 2 + 16 = 18

        # Complex projection
        self.proj_real = nn.Linear(input_dim, embed_dim)
        self.proj_imag = nn.Linear(input_dim, embed_dim)
        nn.init.normal_(self.proj_real.weight, std=0.02)
        nn.init.normal_(self.proj_imag.weight, std=0.02)

    def forward(self, dt_values):
        """
        Args:
            dt_values: [B, T] integer dt values (0-480)
        Returns:
            Complex tensor [B, T, embed_dim]
        """
        dt_float = dt_values.float()

        # Log-scale feature: log(1 + dt) normalized
        log_dt = torch.log1p(dt_float) / math.log(1 + self.max_dt)  # [0, 1]

        # Linear-scale feature
        lin_dt = dt_float / self.max_dt  # [0, 1]

        # Multi-scale periodic features in log-space
        log_scaled = log_dt.unsqueeze(-1) * self.log_freqs * math.pi  # [B, T, num_scales]

        features = torch.cat([
            log_dt.unsqueeze(-1),       # [B, T, 1]
            lin_dt.unsqueeze(-1),       # [B, T, 1]
            torch.sin(log_scaled),      # [B, T, num_scales]
            torch.cos(log_scaled),      # [B, T, num_scales]
        ], dim=-1)

        out_real = self.proj_real(features)
        out_imag = self.proj_imag(features)

        return torch.complex(out_real, out_imag)


class CustomComplexEmbedding(nn.Module):
    """Complex-valued embedding for earthquake features with enhanced geographic encoding.

    Embeds 7 features: year, month, latitude, longitude, magnitude, depth, time_diff

    Location-Priority Architecture:
    - GPE (lat/lon) gets 40% of embedding space (was 16.7%)
    - dt gets enhanced encoding: standard embedding + log-scale encoding
    - Other features (yr, mt, mag, depth) share remaining space equally

    This captures:
    - Spherical geometry of Earth (spherical harmonics)
    - Multi-scale spatial patterns (faults to plates)
    - Tectonic zone relationships
    - Relative positions between earthquakes
    - Log-normal distribution of inter-event times
    """
    def __init__(self, sizes, embed_dim, use_gpe=True):
        super().__init__()
        self.yr_size = sizes['yr_size'] + 1
        self.mt_size = sizes['mt_size'] + 1
        self.x_size = sizes['x_size'] + 1
        self.y_size = sizes['y_size'] + 1
        self.m_size = sizes['m_size'] + 1
        self.d_size = sizes['d_size'] + 1
        self.t_size = sizes['t_size'] + 1

        self.embed_dim = embed_dim
        self.use_gpe = use_gpe

        if use_gpe:
            # LOCATION-PRIORITY ALLOCATION:
            # GPE (lat/lon): 40% of embedding space → location is most important for eq prediction
            # dt: 14% (standard embed + log-scale encoding combined)
            # yr, mt, mag, depth: ~11.5% each
            #
            # For n_embed=1176:
            #   gpe_dim    = 472  (40.1%)
            #   dt_dim     = 168  (14.3%) - split: 84 standard + 84 log-scale
            #   other_dim  = 134  each (11.4%)
            #   Total: 472 + 168 + 134*4 = 472 + 168 + 536 = 1176 ✓

            self.gpe_dim = embed_dim * 2 // 5          # 40% for location
            self.dt_dim = embed_dim // 7               # 14% for dt (total)
            remaining = embed_dim - self.gpe_dim - self.dt_dim
            self.feature_dim = remaining // 4          # yr, mt, mag, depth
            # Absorb any rounding remainder into GPE
            self.gpe_dim = embed_dim - self.dt_dim - 4 * self.feature_dim

            # dt split: half for standard embedding, half for log-scale encoding
            self.dt_embed_dim = self.dt_dim // 2
            self.dt_log_dim = self.dt_dim - self.dt_embed_dim

            # Complex embeddings for non-location features
            self.yr_embed = ComplexEmbedding(self.yr_size, self.feature_dim)
            self.mt_embed = ComplexEmbedding(self.mt_size, self.feature_dim)
            self.m_embed = ComplexEmbedding(self.m_size, self.feature_dim)
            self.d_embed = ComplexEmbedding(self.d_size, self.feature_dim)

            # dt: standard discrete embedding
            self.t_embed = ComplexEmbedding(self.t_size, self.dt_embed_dim)

            # dt: log-scale continuous encoding (captures log-normal distribution)
            self.t_log_embed = LogScaleDtEncoding(
                embed_dim=self.dt_log_dim,
                max_dt=sizes['t_size'],  # 720
                num_scales=8
            )

            # Geographic Positional Encoding for lat/lon (with MORE capacity)
            self.gpe = GeographicPositionalEncoding(
                embed_dim=self.gpe_dim,
                spherical_degree=8,      # increased from 6 for finer spatial detail
                num_frequencies=16,      # increased from 12 for more scales
                num_zones=32             # increased from 24 for better zone coverage
            )
        else:
            # Original behavior: simple embeddings for all features
            self.feature_dim = embed_dim // 7

            self.yr_embed = ComplexEmbedding(self.yr_size, self.feature_dim)
            self.mt_embed = ComplexEmbedding(self.mt_size, self.feature_dim)
            self.x_embed = ComplexEmbedding(self.x_size, self.feature_dim)
            self.y_embed = ComplexEmbedding(self.y_size, self.feature_dim)
            self.m_embed = ComplexEmbedding(self.m_size, self.feature_dim)
            self.d_embed = ComplexEmbedding(self.d_size, self.feature_dim)
            self.t_embed = ComplexEmbedding(self.t_size, self.feature_dim)
            self.gpe = None
            self.t_log_embed = None

    def forward(self, data):
        """
        Args:
            data: Tensor of shape [B, T, 7] with earthquake features
                  [year, month, lat, lon, mag, depth, time_diff]
        Returns:
            Complex tensor of shape [B, T, embed_dim]
        """
        yr_emb = self.yr_embed(data[:, :, 0])
        mt_emb = self.mt_embed(data[:, :, 1])
        m_emb = self.m_embed(data[:, :, 4])
        d_emb = self.d_embed(data[:, :, 5])
        t_emb = self.t_embed(data[:, :, 6])

        if self.use_gpe:
            # Use Geographic Positional Encoding for location
            lat_encoded = data[:, :, 2]
            lon_encoded = data[:, :, 3]
            loc_emb = self.gpe(lat_encoded, lon_encoded)

            # Log-scale dt encoding (captures log-normal distribution)
            t_log_emb = self.t_log_embed(data[:, :, 6])

            # Concatenate: location first (most important), then dt, then others
            emb = torch.cat((loc_emb, yr_emb, mt_emb, m_emb, d_emb, t_emb, t_log_emb), dim=-1)
        else:
            # Original: separate embeddings for lat/lon
            x_emb = self.x_embed(data[:, :, 2])
            y_emb = self.y_embed(data[:, :, 3])
            emb = torch.cat((yr_emb, mt_emb, x_emb, y_emb, m_emb, d_emb, t_emb), dim=-1)

        return emb


class ComplexPositionalEncoding(nn.Module):
    """Complex-valued positional encoding using Euler's formula."""
    def __init__(self, max_len, embed_dim, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))

        pe_real = torch.zeros(max_len, embed_dim)
        pe_real[:, 0::2] = torch.sin(position * div_term)  # sin for even indices
        pe_real[:, 1::2] = torch.cos(position * div_term)  # cos for odd indices

        pe_imag = torch.zeros(max_len, embed_dim)
        pe_imag[:, 0::2] = torch.cos(position * div_term)  # cos for even indices (phase shifted)
        pe_imag[:, 1::2] = torch.sin(position * div_term)  # sin for odd indices (phase shifted)

        pe = torch.complex(pe_real, pe_imag).unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        T = x.size(1)
        x_real = x.real + self.pe[:, :T].real
        x_imag = x.imag + self.pe[:, :T].imag
        x_real = self.dropout(x_real)
        x_imag = self.dropout(x_imag)
        return torch.complex(x_real, x_imag)


# =============================================================================
# EqModelComplex - Main Model Class
# =============================================================================

class EqModelComplex(nn.Module):
    """
    Advanced Complex-valued Earthquake Prediction Model.

    Uses features from complex_transformer_complete.py:
    - Rotary Position Embeddings (RoPE)
    - PhaseGatedSiLU activation (smooth gradients + phase-aware gating)
    - Hermitian attention scores
    - Pre-norm architecture
    - Complex dropout
    - Complex layer normalization with complex variance

    Predicts 4 values:
    - Latitude (0-180 encoded, actual -90 to +90)
    - Longitude (0-360 encoded, actual -180 to +180)
    - Time difference (0-150 minutes)
    - Magnitude (0-91 encoded, actual 0.0 to 9.1)
    """

    def __init__(self, sizes, B, T, n_embed, n_heads, n_layer, dropout, device, p_max, use_rope=True, use_gpe=True):
        super().__init__()
        self.n_layer = n_layer
        self.B = B
        self.T = T
        self.n_embed = n_embed
        self.device = device
        self.use_rope = use_rope
        self.use_gpe = use_gpe

        # Complex embedding for earthquake features with Geographic Positional Encoding
        self.embed = CustomComplexEmbedding(sizes, n_embed, use_gpe=use_gpe)

        # Optional positional encoding (disabled if using RoPE)
        if not use_rope:
            self.pos_enc = ComplexPositionalEncoding(T, n_embed, dropout)
        else:
            self.pos_enc = None

        # Embedding dropout
        self.embed_dropout = ComplexDropout(dropout)

        # Complex transformer blocks with spatial attention bias
        self.blocks = nn.ModuleList([
            ComplexTransformerBlock(n_embed, n_heads, T, dropout, use_rope, use_spatial_bias=use_gpe)
            for _ in range(n_layer)
        ])

        # Final layer norm
        self.ln_f = ComplexLayerNorm(n_embed)

        # Output heads (real-valued for classification)
        # Input is concatenation of real and imaginary parts
        self.lat_head = nn.Linear(n_embed * 2, 181, bias=False)   # Latitude: 0-180
        self.lon_head = nn.Linear(n_embed * 2, 361, bias=False)   # Longitude: 0-360
        self.dt_head = nn.Linear(n_embed * 2, 721, bias=False)    # Time diff: 0-720 minutes (12 hours)
        self.mag_head = nn.Linear(n_embed * 2, 92, bias=False)    # Magnitude: 0-91

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = (2 * self.n_layer) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, label_smoothing=0.0, use_all_positions=False):
        """
        Forward pass for 4-parameter prediction.

        Args:
            idx: Input tensor [B, T, 7] - sequence of earthquakes
            targets: Dict {'lat': [B], 'lon': [B], 'dt': [B], 'mag': [B]} or None
            label_smoothing: Label smoothing factor (0.0-0.2 recommended)
            use_all_positions: If True, compute loss on all positions (more training signal)

        Returns:
            logits: Dict with 'lat', 'lon', 'dt', 'mag' logits
            loss: Combined loss or None
        """
        B_actual, T_actual = idx.shape[0], idx.shape[1]

        # Extract lat/lon for spatial attention bias
        lat = idx[:, :, 2] if self.use_gpe else None
        lon = idx[:, :, 3] if self.use_gpe else None

        # Complex embedding
        x = self.embed(idx)

        # Add positional encoding if not using RoPE
        if self.pos_enc is not None:
            x = self.pos_enc(x)

        # Embedding dropout
        x = self.embed_dropout(x)

        # Pass through transformer blocks with spatial attention bias
        for block in self.blocks:
            x = block(x, lat=lat, lon=lon)

        # Final layer norm
        x = self.ln_f(x)

        # Combine real and imaginary for output heads
        x_combined = torch.cat([x.real, x.imag], dim=-1)

        # Compute logits for each head
        lat_logits = self.lat_head(x_combined)
        lon_logits = self.lon_head(x_combined)
        dt_logits = self.dt_head(x_combined)
        mag_logits = self.mag_head(x_combined)

        logits = {
            'lat': lat_logits,
            'lon': lon_logits,
            'dt': dt_logits,
            'mag': mag_logits
        }

        if targets is None:
            loss = None
        else:
            # Compute cross-entropy loss with label smoothing
            # Weight losses inversely proportional to number of classes to balance gradients
            # lat: 181 classes -> weight ~1.0 (reference)
            # lon: 361 classes -> weight ~0.5 (has 2x classes)
            # dt:  481 classes -> weight ~1.2 (time diff 0-480 minutes)
            # mag:  92 classes -> weight ~1.3 (fewest classes, most important)

            # Use last position only (standard approach for next-token prediction)
            lat_loss = FN.cross_entropy(lat_logits[:, -1, :], targets['lat'], label_smoothing=label_smoothing)
            lon_loss = FN.cross_entropy(lon_logits[:, -1, :], targets['lon'], label_smoothing=label_smoothing)
            dt_loss = FN.cross_entropy(dt_logits[:, -1, :], targets['dt'], label_smoothing=label_smoothing)
            mag_loss = FN.cross_entropy(mag_logits[:, -1, :], targets['mag'], label_smoothing=label_smoothing)

            # Balanced loss weighting (normalized to sum to 4.0 for comparable total)
            loss = 1.0 * lat_loss + 0.5 * lon_loss + 1.2 * dt_loss + 1.3 * mag_loss

        return logits, loss

    @torch.no_grad()
    def generate(self, x_test, temperature=2.0):
        """Generate predictions for latitude, longitude, time difference, and magnitude.

        Args:
            x_test: Input tensor [B, T, 7]
            temperature: Sampling temperature (higher = more random)

        Returns:
            dict with 'lat', 'lon', 'dt', 'mag' encoded values
        """
        # Extract lat/lon for spatial attention bias
        lat = x_test[:, :, 2] if self.use_gpe else None
        lon = x_test[:, :, 3] if self.use_gpe else None

        # Complex embedding
        x = self.embed(x_test)

        # Add positional encoding if not using RoPE
        if self.pos_enc is not None:
            x = self.pos_enc(x)

        # Pass through transformer blocks with spatial attention bias
        for block in self.blocks:
            x = block(x, lat=lat, lon=lon)

        # Final layer norm
        x = self.ln_f(x)

        # Get last position and combine real/imaginary
        x_last = torch.cat([x[:, -1, :].real, x[:, -1, :].imag], dim=-1)

        # Get logits from each head
        lat_logits = self.lat_head(x_last)
        lon_logits = self.lon_head(x_last)
        dt_logits = self.dt_head(x_last)
        mag_logits = self.mag_head(x_last)

        # Handle NaN/Inf
        for logits in [lat_logits, lon_logits, dt_logits, mag_logits]:
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                logits.copy_(torch.nan_to_num(logits, nan=0.0, posinf=100.0, neginf=-100.0))

        # Apply temperature and softmax
        lat_probs = FN.softmax(lat_logits / temperature, dim=-1)
        lon_probs = FN.softmax(lon_logits / temperature, dim=-1)
        dt_probs = FN.softmax(dt_logits / temperature, dim=-1)
        mag_probs = FN.softmax(mag_logits / temperature, dim=-1)

        # Sample from distributions
        lat_pred = torch.multinomial(lat_probs, num_samples=1)
        lon_pred = torch.multinomial(lon_probs, num_samples=1)
        dt_pred = torch.multinomial(dt_probs, num_samples=1)
        mag_pred = torch.multinomial(mag_probs, num_samples=1)

        # Clamp to valid ranges (no artificial minimum on magnitude)
        lat_val = min(max(lat_pred.item(), 0), 180)
        lon_val = min(max(lon_pred.item(), 0), 360)
        dt_val = min(max(dt_pred.item(), 1), 720)    # Min 1 minute, max 12 hours between events
        mag_val = min(max(mag_pred.item(), 0), 91)   # Full magnitude range 0-91 (0.0-9.1)

        return {
            'lat': lat_val,
            'lon': lon_val,
            'dt': dt_val,
            'mag': mag_val
        }


# =============================================================================
# Utility Functions
# =============================================================================

def count_parameters(model):
    """Count model parameters including complex parameters."""
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


# =============================================================================
# Complex-Aware Optimizer
# =============================================================================

class ComplexAdamW(torch.optim.Optimizer):
    """
    AdamW optimizer adapted for complex parameters.

    KEY for Full Complex: Handles complex gradients by treating them as 2D real vectors,
    which is the standard approach in complex optimization. The complex number stays
    as a unit - we optimize both real and imaginary parts together.

    For complex z = a + bi, we view optimization as jointly optimizing (a, b) in R².
    This maintains the coupling between real and imaginary parts throughout training.
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
                    # Convert to real view for optimization - keeps real/imag coupled!
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


class GradientClipper:
    """
    Gradient clipping for complex parameters.

    Full Complex: Uses magnitude of complex gradient for norm computation,
    treating real and imaginary parts as a coupled unit.
    """

    def __init__(self, max_norm: float = 1.0):
        self.max_norm = max_norm

    def __call__(self, parameters):
        """Clip gradients by global norm."""
        parameters = list(filter(lambda p: p.grad is not None, parameters))

        total_norm = 0.0
        for p in parameters:
            if p.is_complex():
                # Use magnitude of complex gradient - treats real/imag as unit
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


if __name__ == "__main__":
    print("=" * 70)
    print("EqModelComplex - Advanced Complex-Valued Earthquake Prediction Model")
    print("=" * 70)

    # Test configuration
    sizes = {
        'yr_size': 60,      # Years 0-60 (1970-2030)
        'mt_size': 11,      # Months 0-11
        'x_size': 180,      # Latitude 0-180
        'y_size': 360,      # Longitude 0-360
        'm_size': 91,       # Magnitude 0-91 (0.0-9.1)
        'd_size': 200,      # Depth 0-200 km
        't_size': 720       # Time diff 0-720 minutes (12 hours)
    }

    B = 4       # Batch size
    T = 64      # Sequence length
    n_embed = 112  # Embedding dimension (divisible by 7)
    n_heads = 8
    n_layer = 4
    dropout = 0.1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\nConfiguration:")
    print(f"  Batch size: {B}")
    print(f"  Sequence length: {T}")
    print(f"  Embedding dim: {n_embed}")
    print(f"  Num heads: {n_heads}")
    print(f"  Num layers: {n_layer}")
    print(f"  Device: {device}")
    print(f"  Use RoPE: True")

    # Create model
    model = EqModelComplex(sizes, B, T, n_embed, n_heads, n_layer, dropout, device, p_max=181, use_rope=True)
    model = model.to(device)

    # Count parameters
    param_counts = count_parameters(model)
    print(f"\nParameters:")
    print(f"  Total: {param_counts['total']:,}")
    print(f"  Trainable: {param_counts['trainable']:,}")
    print(f"  Complex equivalent: {param_counts['complex_equivalent']:,}")

    # Create sample input with proper bounds for each feature
    sample_input = torch.zeros(B, T, 7, dtype=torch.long).to(device)
    sample_input[:, :, 0] = torch.randint(0, sizes['yr_size'], (B, T))  # year
    sample_input[:, :, 1] = torch.randint(0, sizes['mt_size'], (B, T))  # month
    sample_input[:, :, 2] = torch.randint(0, sizes['x_size'], (B, T))   # latitude
    sample_input[:, :, 3] = torch.randint(0, sizes['y_size'], (B, T))   # longitude
    sample_input[:, :, 4] = torch.randint(0, sizes['m_size'], (B, T))   # magnitude
    sample_input[:, :, 5] = torch.randint(0, sizes['d_size'], (B, T))   # depth
    sample_input[:, :, 6] = torch.randint(0, sizes['t_size'], (B, T))   # time diff

    print(f"\nSample input shape: {sample_input.shape}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        targets = {
            'lat': torch.randint(0, 181, (B,)).to(device),
            'lon': torch.randint(0, 361, (B,)).to(device),
            'dt': torch.randint(0, 721, (B,)).to(device),  # 0-720 minutes (12 hours)
            'mag': torch.randint(0, 92, (B,)).to(device)
        }
        logits, loss = model(sample_input, targets)

    print(f"\nOutput shapes:")
    print(f"  lat_logits: {logits['lat'].shape}")
    print(f"  lon_logits: {logits['lon'].shape}")
    print(f"  dt_logits: {logits['dt'].shape}")
    print(f"  mag_logits: {logits['mag'].shape}")
    print(f"  Loss: {loss.item():.4f}")

    # Test generation
    print("\nTesting generation...")
    prediction = model.generate(sample_input[:1])
    print(f"  Predicted lat: {prediction['lat']} (encoded)")
    print(f"  Predicted lon: {prediction['lon']} (encoded)")
    print(f"  Predicted dt: {prediction['dt']} minutes")
    print(f"  Predicted mag: {prediction['mag']/10:.1f}")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
