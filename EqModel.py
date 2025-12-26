import torch
import torch.nn as nn
from torch.nn import functional as FN
import math

# =============================================================================
# Native PyTorch Complex-valued Neural Network Components
# Using torch.complex for efficient complex tensor operations
# =============================================================================

class ComplexLinear(nn.Module):
    """Complex-valued linear layer using native PyTorch complex tensors.

    Computes: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Separate real and imaginary weight matrices
        self.real_weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.imag_weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)

        if bias:
            self.real_bias = nn.Parameter(torch.zeros(out_features))
            self.imag_bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('real_bias', None)
            self.register_parameter('imag_bias', None)

    def forward(self, x):
        """Forward pass with complex tensor input.

        Args:
            x: Complex tensor of shape [..., in_features]
        Returns:
            Complex tensor of shape [..., out_features]
        """
        # Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        real = x.real @ self.real_weight.T - x.imag @ self.imag_weight.T
        imag = x.real @ self.imag_weight.T + x.imag @ self.real_weight.T

        if self.real_bias is not None:
            real = real + self.real_bias
            imag = imag + self.imag_bias

        return torch.complex(real, imag)


class ComplexEmbedding(nn.Module):
    """Complex-valued embedding layer using native PyTorch complex tensors."""

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.real = nn.Embedding(num_embeddings, embedding_dim)
        self.imag = nn.Embedding(num_embeddings, embedding_dim)

        # Initialize with small values
        nn.init.normal_(self.real.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.imag.weight, mean=0.0, std=0.01)

    def forward(self, x):
        """Returns complex tensor from indices."""
        return torch.complex(self.real(x), self.imag(x))


class ComplexLayerNorm(nn.Module):
    """Layer normalization for complex tensors.

    Normalizes real and imaginary parts separately, then applies complex scaling.
    """
    def __init__(self, features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma_real = nn.Parameter(torch.ones(features))
        self.gamma_imag = nn.Parameter(torch.zeros(features))
        self.beta_real = nn.Parameter(torch.zeros(features))
        self.beta_imag = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        """Normalize complex tensor.

        Args:
            x: Complex tensor of shape [..., features]
        Returns:
            Normalized complex tensor
        """
        # Normalize real and imaginary parts separately
        real_mean = x.real.mean(-1, keepdim=True)
        imag_mean = x.imag.mean(-1, keepdim=True)

        real_centered = x.real - real_mean
        imag_centered = x.imag - imag_mean

        # Use combined variance for stability
        var = (real_centered.pow(2) + imag_centered.pow(2)).mean(-1, keepdim=True)
        std = torch.sqrt(var + self.eps)

        real_norm = real_centered / std
        imag_norm = imag_centered / std

        # Apply complex scaling: (gamma_r + i*gamma_i) * (x_r + i*x_i) + (beta_r + i*beta_i)
        out_real = self.gamma_real * real_norm - self.gamma_imag * imag_norm + self.beta_real
        out_imag = self.gamma_real * imag_norm + self.gamma_imag * real_norm + self.beta_imag

        return torch.complex(out_real, out_imag)


class ComplexPositionalEncoding(nn.Module):
    """Complex-valued positional encoding using Euler's formula: e^(i*theta) = cos(theta) + i*sin(theta)"""

    def __init__(self, max_len, embed_dim, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))

        # Real part: cos(position * div_term)
        pe_real = torch.zeros(max_len, embed_dim)
        pe_real[:, 0::2] = torch.cos(position * div_term)
        pe_real[:, 1::2] = torch.cos(position * div_term)

        # Imaginary part: sin(position * div_term)
        pe_imag = torch.zeros(max_len, embed_dim)
        pe_imag[:, 0::2] = torch.sin(position * div_term)
        pe_imag[:, 1::2] = torch.sin(position * div_term)

        # Register as buffer (not a parameter)
        pe = torch.complex(pe_real, pe_imag).unsqueeze(0)  # [1, max_len, embed_dim]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Add positional encoding to complex tensor.

        Args:
            x: Complex tensor of shape [B, T, D]
        """
        T = x.size(1)
        # Add positional encoding
        x_real = x.real + self.pe[:, :T].real
        x_imag = x.imag + self.pe[:, :T].imag

        # Apply dropout to real and imaginary separately
        x_real = self.dropout(x_real)
        x_imag = self.dropout(x_imag)

        return torch.complex(x_real, x_imag)


class ComplexAttention(nn.Module):
    """Complex-valued multi-head attention using native complex tensors.

    Attention scores are computed using complex dot product:
    Re(Q) @ Re(K)^T + Im(Q) @ Im(K)^T for the real part
    """
    def __init__(self, embed_dim, num_heads, block_size, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.embed_dim = embed_dim

        # Complex linear projections
        self.q_proj = ComplexLinear(embed_dim, embed_dim, bias=False)
        self.k_proj = ComplexLinear(embed_dim, embed_dim, bias=False)
        self.v_proj = ComplexLinear(embed_dim, embed_dim, bias=False)
        self.out_proj = ComplexLinear(embed_dim, embed_dim)

        # Causal mask
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: Complex tensor of shape [B, L, D]
        Returns:
            Complex tensor of shape [B, L, D]
        """
        B, L, D = x.shape

        # Project to Q, K, V
        Q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Complex dot product for attention scores
        # Re(Q @ K*) = Re(Q) @ Re(K)^T + Im(Q) @ Im(K)^T
        attn_real = (Q.real @ K.real.transpose(-2, -1)) + (Q.imag @ K.imag.transpose(-2, -1))
        attn_scores = attn_real * self.scale

        # Apply causal mask
        attn_scores = attn_scores.masked_fill(self.tril[:L, :L] == 0, float('-inf'))

        # Softmax on real-valued scores
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention weights to complex values
        out_real = attn_weights @ V.real
        out_imag = attn_weights @ V.imag

        out = torch.complex(out_real, out_imag)
        out = out.transpose(1, 2).contiguous().view(B, L, D)

        return self.out_proj(out)


class ComplexFeedForward(nn.Module):
    """Complex-valued feed-forward network with CReLU activation.

    CReLU(z) = ReLU(Re(z)) + i*ReLU(Im(z))
    """
    def __init__(self, embed_dim, dropout=0.1, expansion=4):
        super().__init__()
        self.linear1 = ComplexLinear(embed_dim, expansion * embed_dim)
        self.linear2 = ComplexLinear(expansion * embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: Complex tensor of shape [..., embed_dim]
        """
        x = self.linear1(x)

        # CReLU activation: ReLU on real and imaginary parts separately
        x_real = FN.relu(x.real)
        x_imag = FN.relu(x.imag)
        x = torch.complex(x_real, x_imag)

        x = self.linear2(x)

        # Apply dropout to real and imaginary parts
        out_real = self.dropout(x.real)
        out_imag = self.dropout(x.imag)

        return torch.complex(out_real, out_imag)


class ComplexBlock(nn.Module):
    """Complex-valued transformer block with pre-norm architecture."""

    def __init__(self, embed_dim, num_heads, block_size, dropout=0.1):
        super().__init__()
        self.ln1 = ComplexLayerNorm(embed_dim)
        self.attn = ComplexAttention(embed_dim, num_heads, block_size, dropout)
        self.ln2 = ComplexLayerNorm(embed_dim)
        self.ffwd = ComplexFeedForward(embed_dim, dropout)

    def forward(self, x):
        """
        Args:
            x: Complex tensor of shape [B, T, D]
        """
        # Self-attention with residual
        x = x + self.attn(self.ln1(x))
        # Feed-forward with residual
        x = x + self.ffwd(self.ln2(x))
        return x


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
        self.feature_dim = embed_dim // 7  # Divide embedding among 7 features

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
        # Get complex embeddings for each feature
        yr_emb = self.yr_embed(data[:, :, 0])
        mt_emb = self.mt_embed(data[:, :, 1])
        x_emb = self.x_embed(data[:, :, 2])
        y_emb = self.y_embed(data[:, :, 3])
        m_emb = self.m_embed(data[:, :, 4])
        d_emb = self.d_embed(data[:, :, 5])
        t_emb = self.t_embed(data[:, :, 6])

        # Concatenate all embeddings (complex tensors)
        # torch.cat works with complex tensors
        emb = torch.cat((yr_emb, mt_emb, x_emb, y_emb, m_emb, d_emb, t_emb), dim=-1)

        return emb


class ComplexEqModel(nn.Module):
    """Complex-valued Earthquake Prediction Model using native PyTorch complex tensors.

    Predicts 4 values:
    - Latitude (0-180 encoded, actual -90 to +90)
    - Longitude (0-360 encoded, actual -180 to +180)
    - Time difference to next earthquake (0-150 in minutes)
    - Magnitude (0-91 encoded, actual 0.0 to 9.1)
    """

    def __init__(self, sizes, B, T, n_embed, n_heads, n_layer, dropout, device, p_max):
        super().__init__()
        self.n_layer = n_layer
        self.B = B
        self.T = T
        self.n_embed = n_embed
        self.device = device

        # Complex embedding and positional encoding
        self.embed = CustomComplexEmbedding(sizes, n_embed)
        self.pos_enc = ComplexPositionalEncoding(T, n_embed, dropout)

        # Complex transformer blocks
        self.blocks = nn.ModuleList([
            ComplexBlock(n_embed, n_heads, T, dropout)
            for _ in range(n_layer)
        ])

        # Final layer norm
        self.ln_f = ComplexLayerNorm(n_embed)

        # Output heads (real-valued for classification)
        # Input is concatenation of real and imaginary parts
        self.lat_head = nn.Linear(n_embed * 2, 181, bias=False)   # Latitude: 0-180
        self.lon_head = nn.Linear(n_embed * 2, 361, bias=False)   # Longitude: 0-360
        self.dt_head = nn.Linear(n_embed * 2, 151, bias=False)    # Time diff: 0-150 minutes
        self.mag_head = nn.Linear(n_embed * 2, 92, bias=False)    # Magnitude: 0-91 (mag*10)

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
        x = self.embed(idx)  # [B, T, n_embed] complex

        # Add positional encoding
        x = self.pos_enc(x)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm
        x = self.ln_f(x)

        # Combine real and imaginary for output heads
        x_combined = torch.cat([x.real, x.imag], dim=-1)  # [B, T, n_embed*2]

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

        # Add positional encoding
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
# Legacy Real-valued Components (kept for backward compatibility)
# =============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, T, n_embed):
        super().__init__()
        self.dropout = nn.Dropout(p=0.1)
        pe = torch.zeros(T, n_embed)
        position = torch.arange(0, T, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_embed, 2).float() * (-torch.log(torch.tensor(10000.0)) / n_embed))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        y = self.pe[:, :x.size(1)]
        x = x + y
        return self.dropout(x)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class CustomEmbedding(nn.Module):
    """Standard embedding layer (kept for backward compatibility)"""

    def __init__(self, sizes, n_embed):
        super().__init__()
        self.yr_size = sizes['yr_size'] + 1
        self.mt_size = sizes['mt_size'] + 1
        self.x_size = sizes['x_size'] + 1
        self.y_size = sizes['y_size'] + 1
        self.m_size = sizes['m_size'] + 1
        self.d_size = sizes['d_size'] + 1
        self.t_size = sizes['t_size'] + 1

        self.n_embed = n_embed
        n2_embed = n_embed // 7

        self.yr_embed = nn.Embedding(self.yr_size, n2_embed)
        self.mt_embed = nn.Embedding(self.mt_size, n2_embed)
        self.x_embed = nn.Embedding(self.x_size, n2_embed)
        self.y_embed = nn.Embedding(self.y_size, n2_embed)
        self.m_embed = nn.Embedding(self.m_size, n2_embed)
        self.d_embed = nn.Embedding(self.d_size, n2_embed)
        self.t_embed = nn.Embedding(self.t_size, n2_embed)

    def forward(self, data, target=None):
        yr_emb = self.yr_embed(data[:, :, 0])
        mn_emb = self.mt_embed(data[:, :, 1])
        x_emb = self.x_embed(data[:, :, 2])
        y_emb = self.y_embed(data[:, :, 3])
        m_emb = self.m_embed(data[:, :, 4])
        d_emb = self.d_embed(data[:, :, 5])
        t_emb = self.t_embed(data[:, :, 6])

        emb_ = torch.cat((yr_emb, mn_emb, x_emb, y_emb, m_emb, d_emb, t_emb), dim=-1)
        return emb_


class FeedForward(nn.Module):
    def __init__(self, n_embed, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Head(nn.Module):
    """One head for self attention"""
    def __init__(self, head_size, n_embed, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** 0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = FN.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self attentions in parallel"""
    def __init__(self, n_embed, num_heads, head_size, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embed, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class Block(nn.Module):
    def __init__(self, n_embed, n_heads, seq_len, dropout):
        super().__init__()
        head_size = n_embed // n_heads
        self.sa = MultiHeadAttention(n_embed, n_heads, head_size, seq_len, dropout)
        self.ffwd = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class EqModel(nn.Module):
    """Standard Earthquake Model (kept for backward compatibility)"""

    def __init__(self, sizes, B, T, n_embed, n_heads, n_layer, dropout, device, p_max):
        super().__init__()
        self.n_layer = n_layer
        self.B = B
        self.T = T
        self.embed = CustomEmbedding(sizes, n_embed)
        self.p_embed = PositionalEncoding(T, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_heads, T, dropout) for _ in range(n_layer)])
        self.layer_norm = nn.LayerNorm(n_embed, T)
        self.lm_linear = nn.Linear(n_embed, p_max, bias=False)
        self.apply(self._init_weights)
        self.device = device

    def _init_weights(self, module):
        std = 0.2
        if isinstance(module, nn.Linear):
            std = (2*self.n_layer)**-0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T, F = idx.shape
        x = self.embed(idx)
        x = x + self.p_embed(x)
        x = self.blocks(x)
        x = self.layer_norm(x)
        logits = self.lm_linear(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            lgt = logits.view(-1, logits.size(-1))
            tgt = targets.view(-1)
            loss = FN.cross_entropy(lgt, tgt, ignore_index=0)

        return logits, loss

    @torch.no_grad()
    def generate(self, x_test):
        logits, _ = self(x_test)
        logits = logits[:, -1, :]

        has_nan = torch.isnan(logits).any()
        has_inf = torch.isinf(logits).any()
        if has_nan or has_inf:
           print(f"Warning: Logits contain NaN: {has_nan} or Inf: {has_inf}")
           logits = torch.nan_to_num(logits, nan=0.0, posinf=100.0, neginf=-100.0)

        temperature = 1.0
        logits = logits / temperature
        probs = FN.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        return idx_next
