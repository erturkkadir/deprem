"""
EqModelComplex - Advanced Complex-Valued Transformer for Earthquake Prediction
===============================================================================
Based on complex_transformer_complete.py architecture with:
- Rotary Position Embeddings (RoPE)
- ModReLU activation (phase-preserving)
- Hermitian attention scores
- Pre-norm architecture
- Complex dropout

Author: Generated from complex_transformer_complete.py
"""

import torch
import torch.nn as nn
from torch.nn import functional as FN
import math


# =============================================================================
# Complex-valued Neural Network Components
# =============================================================================

class ComplexLinear(nn.Module):
    """Complex-valued linear layer using native PyTorch complex tensors."""
    def __init__(self, in_features, out_features, bias=True, init_std=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Complex Glorot initialization
        if init_std is None:
            std = 1.0 / math.sqrt(2 * (in_features + out_features))
        else:
            std = init_std / math.sqrt(2)

        self.real_weight = nn.Parameter(torch.randn(out_features, in_features) * std)
        self.imag_weight = nn.Parameter(torch.randn(out_features, in_features) * std)

        if bias:
            self.real_bias = nn.Parameter(torch.zeros(out_features))
            self.imag_bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('real_bias', None)
            self.register_parameter('imag_bias', None)

    def forward(self, x):
        """Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i"""
        real = x.real @ self.real_weight.T - x.imag @ self.imag_weight.T
        imag = x.real @ self.imag_weight.T + x.imag @ self.real_weight.T

        if self.real_bias is not None:
            real = real + self.real_bias
            imag = imag + self.imag_bias

        return torch.complex(real, imag)


class ComplexEmbedding(nn.Module):
    """Complex-valued embedding layer."""
    def __init__(self, num_embeddings, embedding_dim, init_std=0.02):
        super().__init__()
        std = init_std / math.sqrt(2)
        self.real = nn.Embedding(num_embeddings, embedding_dim)
        self.imag = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.normal_(self.real.weight, mean=0.0, std=std)
        nn.init.normal_(self.imag.weight, mean=0.0, std=std)

    def forward(self, x):
        return torch.complex(self.real(x), self.imag(x))


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


class ModReLU(nn.Module):
    """
    Modulus ReLU: applies ReLU to magnitude, preserves phase.
    modReLU(z) = ReLU(|z| + b) * e^(i * angle(z))

    Better gradient flow and phase preservation than CReLU.
    """
    def __init__(self, num_features):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, z):
        magnitude = z.abs()
        phase = z.angle()
        activated_magnitude = FN.relu(magnitude + self.bias)
        return activated_magnitude * torch.exp(1j * phase)


class ComplexGELU(nn.Module):
    """Complex GELU: applies GELU separately to real and imaginary parts."""
    def forward(self, z):
        return torch.complex(FN.gelu(z.real), FN.gelu(z.imag))


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

class ComplexMultiHeadAttention(nn.Module):
    """
    Multi-head attention with complex Q, K, V and optional Rotary Position Embeddings.
    Uses Hermitian inner product: Re(Q @ K*) = Re(Q)@Re(K)^T + Im(Q)@Im(K)^T
    """
    def __init__(self, embed_dim, num_heads, block_size, dropout=0.1, use_rope=True):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.embed_dim = embed_dim
        self.use_rope = use_rope

        self.q_proj = ComplexLinear(embed_dim, embed_dim, bias=False)
        self.k_proj = ComplexLinear(embed_dim, embed_dim, bias=False)
        self.v_proj = ComplexLinear(embed_dim, embed_dim, bias=False)
        self.out_proj = ComplexLinear(embed_dim, embed_dim)

        if use_rope:
            self.rotary_emb = ComplexRotaryEmbedding(self.head_dim, block_size)
        else:
            self.rotary_emb = None

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.resid_dropout = ComplexDropout(dropout)

    def forward(self, x):
        B, L, D = x.shape

        Q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE if enabled
        if self.rotary_emb is not None:
            cos, sin = self.rotary_emb(L, Q.device)
            Q, K = apply_complex_rotary_embedding(Q, K, cos, sin)

        # Hermitian attention: Re(Q @ K*)
        attn_scores = (Q.real @ K.real.transpose(-2, -1)) + (Q.imag @ K.imag.transpose(-2, -1))
        attn_scores = attn_scores * self.scale

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

class ComplexFeedForward(nn.Module):
    """Feed-forward network with ModReLU activation for better phase preservation."""
    def __init__(self, embed_dim, dropout=0.1, expansion=4):
        super().__init__()
        hidden_dim = expansion * embed_dim
        self.fc1 = ComplexLinear(embed_dim, hidden_dim)
        self.fc2 = ComplexLinear(hidden_dim, embed_dim)
        self.activation = ModReLU(hidden_dim)
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
    """Complex transformer block with pre-norm architecture."""
    def __init__(self, embed_dim, num_heads, block_size, dropout=0.1, use_rope=True):
        super().__init__()
        self.norm1 = ComplexLayerNorm(embed_dim)
        self.attn = ComplexMultiHeadAttention(embed_dim, num_heads, block_size, dropout, use_rope)
        self.norm2 = ComplexLayerNorm(embed_dim)
        self.ffwd = ComplexFeedForward(embed_dim, dropout)
        self.dropout = ComplexDropout(dropout)

    def forward(self, x):
        # Pre-norm architecture: norm -> sublayer -> residual
        x = x + self.attn(self.norm1(x))
        x = x + self.dropout(self.ffwd(self.norm2(x)))
        return x


# =============================================================================
# Custom Embedding for Earthquake Features
# =============================================================================

class CustomComplexEmbedding(nn.Module):
    """Complex-valued embedding for earthquake features.

    Embeds 7 features: year, month, latitude, longitude, magnitude, depth, time_diff
    """
    def __init__(self, sizes, embed_dim):
        super().__init__()
        self.yr_size = sizes['yr_size'] + 1
        self.mt_size = sizes['mt_size'] + 1
        self.x_size = sizes['x_size'] + 1
        self.y_size = sizes['y_size'] + 1
        self.m_size = sizes['m_size'] + 1
        self.d_size = sizes['d_size'] + 1
        self.t_size = sizes['t_size'] + 1

        self.embed_dim = embed_dim
        self.feature_dim = embed_dim // 7

        # Complex embeddings for each feature
        self.yr_embed = ComplexEmbedding(self.yr_size, self.feature_dim)
        self.mt_embed = ComplexEmbedding(self.mt_size, self.feature_dim)
        self.x_embed = ComplexEmbedding(self.x_size, self.feature_dim)
        self.y_embed = ComplexEmbedding(self.y_size, self.feature_dim)
        self.m_embed = ComplexEmbedding(self.m_size, self.feature_dim)
        self.d_embed = ComplexEmbedding(self.d_size, self.feature_dim)
        self.t_embed = ComplexEmbedding(self.t_size, self.feature_dim)

    def forward(self, data):
        """
        Args:
            data: Tensor of shape [B, T, 7] with earthquake features
        Returns:
            Complex tensor of shape [B, T, embed_dim]
        """
        yr_emb = self.yr_embed(data[:, :, 0])
        mt_emb = self.mt_embed(data[:, :, 1])
        x_emb = self.x_embed(data[:, :, 2])
        y_emb = self.y_embed(data[:, :, 3])
        m_emb = self.m_embed(data[:, :, 4])
        d_emb = self.d_embed(data[:, :, 5])
        t_emb = self.t_embed(data[:, :, 6])

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
        pe_real[:, 0::2] = torch.cos(position * div_term)
        pe_real[:, 1::2] = torch.cos(position * div_term)

        pe_imag = torch.zeros(max_len, embed_dim)
        pe_imag[:, 0::2] = torch.sin(position * div_term)
        pe_imag[:, 1::2] = torch.sin(position * div_term)

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
    - ModReLU activation (phase-preserving)
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

    def __init__(self, sizes, B, T, n_embed, n_heads, n_layer, dropout, device, p_max, use_rope=True):
        super().__init__()
        self.n_layer = n_layer
        self.B = B
        self.T = T
        self.n_embed = n_embed
        self.device = device
        self.use_rope = use_rope

        # Complex embedding for earthquake features
        self.embed = CustomComplexEmbedding(sizes, n_embed)

        # Optional positional encoding (disabled if using RoPE)
        if not use_rope:
            self.pos_enc = ComplexPositionalEncoding(T, n_embed, dropout)
        else:
            self.pos_enc = None

        # Embedding dropout
        self.embed_dropout = ComplexDropout(dropout)

        # Complex transformer blocks
        self.blocks = nn.ModuleList([
            ComplexTransformerBlock(n_embed, n_heads, T, dropout, use_rope)
            for _ in range(n_layer)
        ])

        # Final layer norm
        self.ln_f = ComplexLayerNorm(n_embed)

        # Output heads (real-valued for classification)
        # Input is concatenation of real and imaginary parts
        self.lat_head = nn.Linear(n_embed * 2, 181, bias=False)   # Latitude: 0-180
        self.lon_head = nn.Linear(n_embed * 2, 361, bias=False)   # Longitude: 0-360
        self.dt_head = nn.Linear(n_embed * 2, 151, bias=False)    # Time diff: 0-150 minutes
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

    def forward(self, idx, targets=None):
        """
        Forward pass for 4-parameter prediction.

        Args:
            idx: Input tensor [B, T, 7] - sequence of earthquakes
            targets: Dict {'lat': [B], 'lon': [B], 'dt': [B], 'mag': [B]} or None

        Returns:
            logits: Dict with 'lat', 'lon', 'dt', 'mag' logits
            loss: Combined loss or None
        """
        # Complex embedding
        x = self.embed(idx)

        # Add positional encoding if not using RoPE
        if self.pos_enc is not None:
            x = self.pos_enc(x)

        # Embedding dropout
        x = self.embed_dropout(x)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)

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
            # Compute cross-entropy loss for each target (using last position)
            lat_loss = FN.cross_entropy(lat_logits[:, -1, :], targets['lat'])
            lon_loss = FN.cross_entropy(lon_logits[:, -1, :], targets['lon'])
            dt_loss = FN.cross_entropy(dt_logits[:, -1, :], targets['dt'])
            mag_loss = FN.cross_entropy(mag_logits[:, -1, :], targets['mag'])
            loss = lat_loss + lon_loss + dt_loss + mag_loss

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
        # Complex embedding
        x = self.embed(x_test)

        # Add positional encoding if not using RoPE
        if self.pos_enc is not None:
            x = self.pos_enc(x)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)

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

        # Clamp to valid ranges
        lat_val = min(max(lat_pred.item(), 0), 180)
        lon_val = min(max(lon_pred.item(), 0), 360)
        dt_val = min(max(dt_pred.item(), 1), 150)
        mag_val = min(max(mag_pred.item(), 40), 90)

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


if __name__ == "__main__":
    print("=" * 70)
    print("EqModelComplex - Advanced Complex-Valued Earthquake Prediction Model")
    print("=" * 70)

    # Test configuration
    sizes = {
        'yr_size': 56,
        'mt_size': 12,
        'x_size': 180,
        'y_size': 360,
        'm_size': 91,
        'd_size': 70,
        't_size': 150
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
            'dt': torch.randint(0, 151, (B,)).to(device),
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
