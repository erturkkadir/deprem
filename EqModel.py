import torch
import torch.nn as nn
from torch.nn import functional as FN
import math

# Complex-valued neural network components as specified in CLAUDE.md
class ComplexLinear(nn.Module):
    """Complex-valued linear layer using real and imaginary parts"""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.real_linear = nn.Linear(in_features, out_features, bias=bias)
        self.imag_linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x_real, x_imag):
        # Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        out_real = self.real_linear(x_real) - self.imag_linear(x_imag)
        out_imag = self.real_linear(x_imag) + self.imag_linear(x_real)
        return out_real, out_imag

class ComplexEmbedding(nn.Module):
    """Complex-valued embedding layer"""
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.real_embed = nn.Embedding(num_embeddings, embedding_dim)
        self.imag_embed = nn.Embedding(num_embeddings, embedding_dim)
        # Initialize imaginary part with smaller values
        nn.init.normal_(self.real_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.imag_embed.weight, mean=0.0, std=0.01)

    def forward(self, x):
        return self.real_embed(x), self.imag_embed(x)

class ComplexLayerNorm(nn.Module):
    """Layer normalization for complex values"""
    def __init__(self, features, eps=1e-4):
        super().__init__()
        self.eps = eps
        self.gamma_real = nn.Parameter(torch.ones(features))
        self.gamma_imag = nn.Parameter(torch.zeros(features))
        self.beta_real = nn.Parameter(torch.zeros(features))
        self.beta_imag = nn.Parameter(torch.zeros(features))

    def forward(self, x_real, x_imag):
        # Compute magnitude for normalization
        magnitude = torch.sqrt(x_real**2 + x_imag**2 + self.eps)
        mean_mag = magnitude.mean(-1, keepdim=True)
        std_mag = magnitude.std(-1, keepdim=True)

        # Normalize
        x_real_norm = (x_real - x_real.mean(-1, keepdim=True)) / (std_mag + self.eps)
        x_imag_norm = (x_imag - x_imag.mean(-1, keepdim=True)) / (std_mag + self.eps)

        # Apply complex scaling
        out_real = self.gamma_real * x_real_norm - self.gamma_imag * x_imag_norm + self.beta_real
        out_imag = self.gamma_real * x_imag_norm + self.gamma_imag * x_real_norm + self.beta_imag
        return out_real, out_imag

class ComplexPositionalEncoding(nn.Module):
    """Complex-valued positional encoding using Euler's formula"""
    def __init__(self, T, n_embed):
        super().__init__()
        self.dropout = nn.Dropout(p=0.1)

        position = torch.arange(0, T, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_embed, 2).float() * (-math.log(10000.0) / n_embed))

        # Real part: cos(position * div_term)
        pe_real = torch.zeros(T, n_embed)
        pe_real[:, 0::2] = torch.cos(position * div_term)
        pe_real[:, 1::2] = torch.cos(position * div_term)

        # Imaginary part: sin(position * div_term)
        pe_imag = torch.zeros(T, n_embed)
        pe_imag[:, 0::2] = torch.sin(position * div_term)
        pe_imag[:, 1::2] = torch.sin(position * div_term)

        self.register_buffer('pe_real', pe_real.unsqueeze(0))
        self.register_buffer('pe_imag', pe_imag.unsqueeze(0))

    def forward(self, x_real, x_imag):
        T = x_real.size(1)
        # Complex addition
        out_real = x_real + self.pe_real[:, :T]
        out_imag = x_imag + self.pe_imag[:, :T]
        return self.dropout(out_real), self.dropout(out_imag)

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

class CustomComplexEmbedding(nn.Module):
    """Complex-valued embedding for earthquake features as specified in CLAUDE.md"""

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

        # Complex embeddings for each feature
        self.yr_embed = ComplexEmbedding(self.yr_size, n2_embed)
        self.mt_embed = ComplexEmbedding(self.mt_size, n2_embed)
        self.x_embed = ComplexEmbedding(self.x_size, n2_embed)
        self.y_embed = ComplexEmbedding(self.y_size, n2_embed)
        self.m_embed = ComplexEmbedding(self.m_size, n2_embed)
        self.d_embed = ComplexEmbedding(self.d_size, n2_embed)
        self.t_embed = ComplexEmbedding(self.t_size, n2_embed)

    def forward(self, data):
        # Get complex embeddings (real, imag) for each feature
        yr_real, yr_imag = self.yr_embed(data[:, :, 0])
        mt_real, mt_imag = self.mt_embed(data[:, :, 1])
        x_real, x_imag = self.x_embed(data[:, :, 2])
        y_real, y_imag = self.y_embed(data[:, :, 3])
        m_real, m_imag = self.m_embed(data[:, :, 4])
        d_real, d_imag = self.d_embed(data[:, :, 5])
        t_real, t_imag = self.t_embed(data[:, :, 6])

        # Concatenate all embeddings
        emb_real = torch.cat((yr_real, mt_real, x_real, y_real, m_real, d_real, t_real), dim=-1)
        emb_imag = torch.cat((yr_imag, mt_imag, x_imag, y_imag, m_imag, d_imag, t_imag), dim=-1)

        return emb_real, emb_imag

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
    
class ComplexFeedForward(nn.Module):
    """Complex-valued feed-forward network"""
    def __init__(self, n_embed, dropout):
        super().__init__()
        self.linear1 = ComplexLinear(n_embed, 4 * n_embed)
        self.linear2 = ComplexLinear(4 * n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_real, x_imag):
        # First layer with complex ReLU (modReLU)
        x_real, x_imag = self.linear1(x_real, x_imag)
        # modReLU activation: ReLU on magnitude, preserve phase
        # Use larger epsilon for bfloat16 stability
        eps = 1e-4
        magnitude = torch.sqrt(x_real**2 + x_imag**2 + eps)
        phase_real = x_real / magnitude
        phase_imag = x_imag / magnitude
        activated_mag = FN.relu(magnitude)
        x_real = activated_mag * phase_real
        x_imag = activated_mag * phase_imag
        # Second layer
        x_real, x_imag = self.linear2(x_real, x_imag)
        return self.dropout(x_real), self.dropout(x_imag)

class ComplexHead(nn.Module):
    """Complex-valued attention head"""
    def __init__(self, head_size, n_embed, block_size, dropout):
        super().__init__()
        self.key = ComplexLinear(n_embed, head_size, bias=False)
        self.query = ComplexLinear(n_embed, head_size, bias=False)
        self.value = ComplexLinear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.head_size = head_size

    def forward(self, x_real, x_imag):
        B, T, C = x_real.shape

        k_real, k_imag = self.key(x_real, x_imag)
        q_real, q_imag = self.query(x_real, x_imag)

        # Complex dot product: (a+bi)(c-di) = (ac+bd) + (bc-ad)i
        # For attention, we use magnitude of complex dot product
        wei_real = q_real @ k_real.transpose(-2, -1) + q_imag @ k_imag.transpose(-2, -1)
        wei_imag = q_imag @ k_real.transpose(-2, -1) - q_real @ k_imag.transpose(-2, -1)

        # Use magnitude for attention weights (larger eps for bfloat16)
        wei_magnitude = torch.sqrt(wei_real**2 + wei_imag**2 + 1e-4) * (self.head_size ** -0.5)
        wei_magnitude = wei_magnitude.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = FN.softmax(wei_magnitude, dim=-1)
        wei = self.dropout(wei)

        v_real, v_imag = self.value(x_real, x_imag)
        out_real = wei @ v_real
        out_imag = wei @ v_imag
        return out_real, out_imag

class ComplexMultiHeadAttention(nn.Module):
    """Complex-valued multi-head attention"""
    def __init__(self, n_embed, num_heads, head_size, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([ComplexHead(head_size, n_embed, block_size, dropout) for _ in range(num_heads)])
        self.proj = ComplexLinear(head_size * num_heads, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_real, x_imag):
        outs_real = []
        outs_imag = []
        for h in self.heads:
            out_r, out_i = h(x_real, x_imag)
            outs_real.append(out_r)
            outs_imag.append(out_i)

        out_real = torch.cat(outs_real, dim=-1)
        out_imag = torch.cat(outs_imag, dim=-1)
        out_real, out_imag = self.proj(out_real, out_imag)
        return self.dropout(out_real), self.dropout(out_imag)

class ComplexBlock(nn.Module):
    """Complex-valued transformer block"""
    def __init__(self, n_embed, n_heads, seq_len, dropout):
        super().__init__()
        head_size = n_embed // n_heads
        self.sa = ComplexMultiHeadAttention(n_embed, n_heads, head_size, seq_len, dropout)
        self.ffwd = ComplexFeedForward(n_embed, dropout)
        self.ln1 = ComplexLayerNorm(n_embed)
        self.ln2 = ComplexLayerNorm(n_embed)

    def forward(self, x_real, x_imag):
        # Pre-norm with residual
        ln1_real, ln1_imag = self.ln1(x_real, x_imag)
        sa_real, sa_imag = self.sa(ln1_real, ln1_imag)
        x_real = x_real + sa_real
        x_imag = x_imag + sa_imag

        ln2_real, ln2_imag = self.ln2(x_real, x_imag)
        ff_real, ff_imag = self.ffwd(ln2_real, ln2_imag)
        x_real = x_real + ff_real
        x_imag = x_imag + ff_imag
        return x_real, x_imag

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

class ComplexEqModel(nn.Module):
    """Complex-valued Earthquake Model as specified in CLAUDE.md

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

        # Complex embedding and positional encoding
        self.embed = CustomComplexEmbedding(sizes, n_embed)
        self.p_embed = ComplexPositionalEncoding(T, n_embed)

        # Complex transformer blocks
        self.blocks = nn.ModuleList([ComplexBlock(n_embed, n_heads, T, dropout) for _ in range(n_layer)])
        self.layer_norm = ComplexLayerNorm(n_embed)

        # Output projection heads for 4-parameter prediction
        self.lat_head = nn.Linear(n_embed * 2, 181, bias=False)   # Latitude: 0-180
        self.lon_head = nn.Linear(n_embed * 2, 361, bias=False)   # Longitude: 0-360
        self.dt_head = nn.Linear(n_embed * 2, 151, bias=False)    # Time diff: 0-150 minutes
        self.mag_head = nn.Linear(n_embed * 2, 92, bias=False)    # Magnitude: 0-91 (mag*10)

        self.apply(self._init_weights)
        self.device = device

    def _init_weights(self, module):
        std = 0.2
        if isinstance(module, nn.Linear):
            std = (2 * self.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        Forward pass for 4-parameter prediction (lat, lon, dt, mag).

        Args:
            idx: Input tensor [B, T, 7] - sequence of earthquakes
            targets: Dict {'lat': [B], 'lon': [B], 'dt': [B], 'mag': [B]}
        """
        B, T, F = idx.shape

        # Complex embedding
        x_real, x_imag = self.embed(idx)

        # Complex positional encoding
        x_real, x_imag = self.p_embed(x_real, x_imag)

        # Pass through complex transformer blocks
        for block in self.blocks:
            x_real, x_imag = block(x_real, x_imag)

        # Complex layer norm
        x_real, x_imag = self.layer_norm(x_real, x_imag)

        # Combine real and imaginary parts for output
        x_combined = torch.cat([x_real, x_imag], dim=-1)

        # Compute all 4 heads
        lat_logits = self.lat_head(x_combined)
        lon_logits = self.lon_head(x_combined)
        dt_logits = self.dt_head(x_combined)
        mag_logits = self.mag_head(x_combined)

        if targets is None:
            loss = None
        else:
            # 4-parameter training (no ignore_index - all values are valid)
            lat_loss = FN.cross_entropy(lat_logits[:, -1, :], targets['lat'])
            lon_loss = FN.cross_entropy(lon_logits[:, -1, :], targets['lon'])
            dt_loss = FN.cross_entropy(dt_logits[:, -1, :], targets['dt'])
            mag_loss = FN.cross_entropy(mag_logits[:, -1, :], targets['mag'])
            loss = lat_loss + lon_loss + dt_loss + mag_loss

        # Return logits dict for consistency
        logits = {
            'lat': lat_logits,
            'lon': lon_logits,
            'dt': dt_logits,
            'mag': mag_logits
        }
        return logits, loss

    @torch.no_grad()
    def generate(self, x_test):
        """Generate predictions for latitude, longitude, time difference, and magnitude

        Returns:
            dict with 'lat', 'lon', 'dt', 'mag' encoded values
        """
        B, T, F = x_test.shape

        # Complex embedding
        x_real, x_imag = self.embed(x_test)

        # Complex positional encoding
        x_real, x_imag = self.p_embed(x_real, x_imag)

        # Pass through complex transformer blocks
        for block in self.blocks:
            x_real, x_imag = block(x_real, x_imag)

        # Complex layer norm
        x_real, x_imag = self.layer_norm(x_real, x_imag)

        # Combine real and imaginary parts
        x_combined = torch.cat([x_real, x_imag], dim=-1)

        # Get last position
        x_last = x_combined[:, -1, :]

        # Get logits from each head
        lat_logits = self.lat_head(x_last)
        lon_logits = self.lon_head(x_last)
        dt_logits = self.dt_head(x_last)
        mag_logits = self.mag_head(x_last)

        # Handle NaN/Inf
        for logits in [lat_logits, lon_logits, dt_logits, mag_logits]:
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                logits = torch.nan_to_num(logits, nan=0.0, posinf=100.0, neginf=-100.0)

        # Temperature for sampling (higher = more random, lower = more confident)
        temperature = 2.0
        lat_probs = FN.softmax(lat_logits / temperature, dim=-1)
        lon_probs = FN.softmax(lon_logits / temperature, dim=-1)
        dt_probs = FN.softmax(dt_logits / temperature, dim=-1)
        mag_probs = FN.softmax(mag_logits / temperature, dim=-1)

        # Use multinomial sampling for variety
        lat_pred = torch.multinomial(lat_probs, num_samples=1)
        lon_pred = torch.multinomial(lon_probs, num_samples=1)
        dt_pred = torch.multinomial(dt_probs, num_samples=1)
        mag_pred = torch.multinomial(mag_probs, num_samples=1)

        # Clamp values to valid ranges
        lat_val = min(max(lat_pred.item(), 0), 180)
        lon_val = min(max(lon_pred.item(), 0), 360)
        dt_val = min(max(dt_pred.item(), 1), 150)  # At least 1 minute
        mag_val = min(max(mag_pred.item(), 40), 90)  # M4.0 to M9.0

        return {
            'lat': lat_val,   # 0-180 encoded
            'lon': lon_val,   # 0-360 encoded
            'dt': dt_val,     # 1-150 minutes
            'mag': mag_val    # 40-90 (mag * 10, so M4.0-M9.0)
        }

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
        x = self.embed(idx)                     # B,T, 7 ==> B,T,96
        x = x + self.p_embed(x)
        x = self.blocks(x);                     # B,T,96 ==> B,T,96         
        x = self.layer_norm(x);                 # B,T,96 ==> B,T,96
        logits = self.lm_linear(x);             # B,T,96 ==> B,T,180
        # print(f"logits.shape : {logits.shape}")
        # print(f"logits : {logits}")
       
        
        if(targets is None):            
            loss = None
        else:
            B, T, C = logits.shape  # (B,T,180)
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)            
            lgt = logits.view(-1, logits.size(-1))
            tgt = targets.view(-1)

            # print(f"lgt.shape : {lgt.shape}")
            # print(f"tgt.shape : {tgt.shape}")
            # print(f"lgt : {lgt}")
            # print(f"tgt : {tgt}")
           
            loss = FN.cross_entropy(lgt, tgt, ignore_index=0)
            # print(f"loss : {loss}")
            # print(f"{1/0}")
        return logits, loss    

    @torch.no_grad()
    def generate(self, x_test):

        # print(f"Inference x_test.shape : {x_test.shape}")
        # print(f"Inference x_test : {x_test}")
        
        logits, _ = self(x_test)

        # print(f"Inference logits.shape : {logits.shape}")
        # print(f"Inference logits1 : {logits}")
       
        
        logits = logits[:, -1, :] # becomes (B, C)
        # print(f"Inference logits2 : {logits}")

        has_nan = torch.isnan(logits).any()
        has_inf = torch.isinf(logits).any()
        if has_nan or has_inf:
           print(f"Warning: Logits contain NaN: {has_nan} or Inf: {has_inf}")
           # Fix extreme values
           logits = torch.nan_to_num(logits, nan=0.0, posinf=100.0, neginf=-100.0)
    
        # Add temperature to soften the distribution if needed
        temperature = 1.0  # Adjust this value between 0.5 and 1.0
        logits = logits / temperature

        # apply softmax to get probabilities
        probs = FN.softmax(logits, dim=-1) # (B, C)
        # print(f"probs : {probs}")

        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
        # print(f"idx next : {idx_next}")
        #print(f"{1/0}")

        return idx_next
