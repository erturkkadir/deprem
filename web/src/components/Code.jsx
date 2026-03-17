import { useState } from 'react';
import { Link } from 'react-router-dom';
import Header from './Header';
import Changelog from './Changelog';

const archLayers = [
  {
    title: '1. Input: Earthquake Sequence',
    shape: '[Batch, 384, 12]',
    idea: 'The model reads a sequence of 384 earthquakes, each described by 12 features. Like reading a paragraph of seismic events to predict the next sentence.',
    code: `# 12 features per earthquake:
# [year, month, lat, lon, mag, depth,
#  global_dt, local_dt, hour, doy,
#  moon_phase, moon_dist]
x = dataC.getLastFromDB(T=384)  # → [1, 384, 12]`,
    details: [
      'Features 0-1: Year (0-60) and Month (0-11) — long-term trends and seasonal patterns',
      'Features 2-3: Latitude (0-180) and Longitude (0-360) — encoded location on Earth',
      'Feature 4: Magnitude (0-91, at 0.1 resolution) — event strength',
      'Feature 5: Depth (0-200 km) — shallow vs deep quakes behave differently',
      'Features 6-7: Global dt (0-9) and Local dt (0-25) — log-binned minutes since last M4+',
      'Features 8-11: Hour, Day of year, Moon phase, Moon distance — Earth-state context',
    ],
  },
  {
    title: '2. Complex Embedding',
    shape: '[B, 384, 12] → [B, 384, 1024] complex',
    idea: 'Each feature becomes a complex vector with magnitude + phase. Location gets ~27% of capacity because WHERE matters most.',
    code: `class ComplexEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        # Magnitude embedding (log-scale, always positive)
        self.log_magnitude = nn.Embedding(num_embeddings, embedding_dim)
        # Phase embedding (full circle [-π, π])
        self.phase = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x):
        magnitude = torch.exp(self.log_magnitude(x))
        phase = self.phase(x)
        # → complex: |e| * e^(iθ)
        return torch.complex(
            magnitude * torch.cos(phase),
            magnitude * torch.sin(phase)
        )`,
    details: [
      'GPE (lat+lon): 276 dims (~27%) — Spherical Harmonics + Fourier + Tectonic Zones + Relative Position',
      'Global dt + Local dt: 102 dims each (10%) — discrete embedding + continuous log-scale encoding',
      '8 other features: 68 dims each (~6.6%) — standard ComplexEmbedding lookup',
      'Total: 276 + 102 + 102 + (8×68) = 1,024 complex dims = 2,048 coupled real values',
    ],
  },
  {
    title: '3. Hermitian Multi-Head Attention',
    shape: '[B, 384, 1024] → [B, 384, 1024] complex',
    idea: '8 parallel attention heads — each earthquake "looks at" every previous one. QK-Norm stabilizes attention, PerDimScale lets each dimension learn its own importance. Spatial bias makes nearby events attend more strongly.',
    code: `def forward(self, x, lat=None, lon=None):
    Q = self.q_proj(x)  # ComplexLinear
    K = self.k_proj(x)
    V = self.v_proj(x)

    # QK-Norm: stabilize Q,K magnitudes before attention
    Q = self.q_norm(Q)  # ComplexRMSNorm per head
    K = self.k_norm(K)

    # Inject sequence order via rotation
    Q, K = apply_complex_rotary_embedding(Q, K, cos, sin)

    # PerDimScale: learned per-dimension importance
    dim_scale = base_scale * softplus(self.per_dim_scale)
    Q_scaled = Q * dim_scale  # each dim weighted differently

    # Hermitian inner product (proper complex dot product)
    scores = (
        Q_scaled.real @ K.real.T + Q_scaled.imag @ K.imag.T
    )

    # Geographic distance bias + causal mask
    scores = scores + self.spatial_bias(lat, lon)
    scores = scores.masked_fill(causal_mask == 0, -inf)

    out = torch.complex(softmax(scores) @ V.real,
                        softmax(scores) @ V.imag)`,
    details: [
      'QK-Norm: RMSNorm on Q and K prevents attention logits from exploding — critical for stable complex-valued training (from TimesFM/PaLM)',
      'PerDimScale: replaces fixed 1/√d with learned softplus(weight) per dimension — some dimensions matter more than others',
      'RoPE: Position encoded as rotation — nearby events have similar rotations, far ones diverge',
      'Hermitian: Re(Q)·Re(K)ᵀ + Im(Q)·Im(K)ᵀ — the proper complex inner product',
      'Spatial bias: Per-head learnable distance-decay — half heads prefer nearby (aftershocks), half neutral (tectonic)',
    ],
  },
  {
    title: '4. Complex Gated Feed-Forward (SwiGLU)',
    shape: '[B, 384, 1024] → [B, 384, 4096] → [B, 384, 1024]',
    idea: 'Processes each position independently. The critical part: TRUE complex multiplication for gating, which keeps real and imaginary parts coupled.',
    code: `def forward(self, x):
    # Gate: decides "how much to keep" (magnitude + phase)
    gate = self.gate_proj(x)           # ComplexLinear
    gate = gate * torch.sigmoid(gate.abs())  # ComplexSiLU

    # Up: decides "what to transform into"
    up = self.up_proj(x)               # ComplexLinear

    # TRUE complex multiply: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    # This is where real/imaginary EXCHANGE information
    hidden_real = gate.real * up.real - gate.imag * up.imag
    hidden_imag = gate.real * up.imag + gate.imag * up.real
    hidden = torch.complex(hidden_real, hidden_imag)

    return self.down_proj(hidden)       # Back to 1176 dims`,
    details: [
      'Without true complex multiply, the gate and up paths would stay independent — just two separate real networks pretending to be complex',
      '4× expansion (1024→4096) gives room for complex feature interactions before compressing back',
    ],
  },
  {
    title: '5. Transformer Block (×6 stacked)',
    shape: 'Pre-norm → Attn → Post-norm → Add → Pre-norm → FFN → Post-norm → Add',
    idea: '6 blocks stacked, each with 4 norms (pre+post for both attention and FFN). Early blocks learn "these events are near each other", later blocks learn "this aftershock cascade is accelerating toward a larger event".',
    code: `class ComplexTransformerBlock(nn.Module):
    def forward(self, x, lat=None, lon=None):
        # 4-norm design (pre-norm → sublayer → post-norm → residual)

        # Attention: mix information across sequence
        x = x + self.post_norm1(
            self.attn(self.norm1(x), lat, lon)
        )

        # Feed-forward: process each position independently
        x = x + self.post_norm2(
            self.ffn(self.norm2(x))
        )

        return x

# Main model stacks 6 blocks (203M params total):
for block in self.blocks:      # 6 layers
    x = block(x, lat=lat, lon=lon)
x = self.ln_f(x)               # final norm`,
    details: [
      'Pre-norm (ComplexLayerNorm) + Post-norm (ComplexRMSNorm) — 4-norm design stabilizes residual stream',
      'Residual connections: each layer learns only the delta, not the full representation',
      '6 layers at n_embed=1024, T=384 — sized to share GPU with Ollama (~1 GiB reserved)',
    ],
  },
  {
    title: '6. Output: Mixture Density Network Heads',
    shape: '[B, T, 1024] complex → SpatialMDN(K=20) + MagnitudeMDN(K=8)',
    idea: 'Complex representation is split into real + imaginary (→2,048 dims). Two MDN heads predict full probability distributions — not just a point estimate. At inference, the highest-weight Gaussian component becomes the prediction.',
    code: `# Leave complex space: cat real and imaginary
x_combined = torch.cat([x.real, x.imag], dim=-1)  # [B, T, 2048]

# SpatialMDNHead: K=20 bivariate Gaussians for (lat, lon)
spatial_params = self.spatial_head(x_combined)
# → {'pi': [B,T,20], 'mu_lat': [B,T,20], 'mu_lon': [B,T,20],
#    'sigma_lat': [B,T,20], 'sigma_lon': [B,T,20], 'rho': [B,T,20]}

# MagnitudeMDNHead: K=8 univariate Gaussians
mag_params = self.mag_head(x_combined)
# → {'pi': [B,T,8], 'mu': [B,T,8], 'sigma': [B,T,8]}

# At inference: MAP decode (highest-weight component)
top_k = pi.argsort(descending=True)[0]  # best component
lat = mu_lat[top_k]; lon = mu_lon[top_k]
# Uncertainty radius in km:
sigma_km = 111 * sqrt((sigma_lat[top_k]**2 + sigma_lon[top_k]**2) / 2)`,
    details: [
      'K=20 spatial components: each Gaussian covers a different seismically active zone (Pacific Ring, Alpide belt, etc.)',
      'Bivariate Gaussian: σ_lat, σ_lon, ρ (correlation) — non-axis-aligned uncertainty ellipses',
      'sigma_km: RMS of σ_lat and σ_lon converted to km (1°≈111km) — shown as ±NNN km in UI',
      'MagnitudeMDN K=8 Gaussians — GR prior removed (was suppressing predictions above M4.43)',
      'pi (mixture weight) is returned as a confidence score per prediction',
    ],
  },
  {
    title: '7. Training Loss (5 signals)',
    shape: 'loss = NLL_spatial + NLL_mag + 3.0×hav + 0.5×entropy + 1.0×diversity',
    idea: 'Five simultaneous training signals: MDN NLL teaches the full distribution, min-k haversine pulls the closest component toward targets, magnitude weighting prioritizes big quakes, Omori decay weights recent events more, entropy + diversity keep components spread across the globe.',
    code: `# 1. MDN negative log-likelihood (all K components compete)
spatial_nll = SpatialMDNHead.nll_loss(sp, lat_actual, lon_actual)
mag_nll = MagnitudeMDNHead.nll_loss(mp, mag_actual)

# 2. Combined weight: magnitude × Omori temporal decay
mag_w = 10 ** (0.5 * (mag_actual - 4.0))  # M6→10×, M7→31.6×
omori_w = exp(3.0 * (pos/(T-1) - 1.0))    # newest→1.0×, oldest→0.05×
combined_w = (mag_w * omori_w) / (mag_w * omori_w).mean()

# 3. Haversine loss: min-k (closest component wins)
# dist from each of K=20 mu_k to target [N, K]
hav_k = haversine(mu_lat, mu_lon, lat_tgt, lon_tgt)
hav_loss = log1p(hav_k.min(dim=-1).values).mean()  # best component

# 4. Entropy regularizer (keep mixture weights diverse)
ent_loss = SpatialMDNHead.entropy_loss(sp)  # max entropy = uniform pi

# 5. Diversity loss (Gaussian repulsion between K component means)
div_loss = SpatialMDNHead.diversity_loss(sp)  # tau=30°, prevents Fiji/Tonga collapse

loss = (spatial_nll + mag_nll) * combined_w + 3.0*hav_loss + 0.5*ent_loss + 1.0*div_loss`,
    details: [
      'Min-k haversine: gradient only flows through the CLOSEST component — creates geographic diversity, avoids mid-ocean averaging',
      'Omori decay: pos=0 (oldest) → 0.05× weight; pos=T-1 (newest) → 1.0× weight — model focuses on near-future',
      'Magnitude weights: M4.0→1×, M5.0→3.2×, M6.0→10×, M7.0→31.6×',
      'Entropy loss: prevents all 20 components from collapsing to the same pi weight',
      'Diversity loss: Gaussian repulsion (τ=30°) pushes component means apart — fixed mode collapse where all 20 pointed at Fiji/Tonga',
      'Validation: pure NLL only (no aux losses, no weighting) — true generalization signal',
    ],
  },
];

export default function Code() {
  const [expandedLayer, setExpandedLayer] = useState(null);

  return (
    <div className="min-h-screen bg-zinc-900">
      <Header />

      {/* Page hero */}
      <section className="py-8 border-b border-zinc-800">
        <div className="max-w-5xl mx-auto px-4">
          <div className="flex items-center gap-3 mb-1">
            <div className="w-8 h-8 rounded-lg bg-purple-500/20 flex items-center justify-center">
              <svg className="w-4 h-4 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
              </svg>
            </div>
            <h1 className="text-2xl font-bold text-white">Code & Architecture</h1>
          </div>
          <p className="text-zinc-500 text-sm ml-11">
            Layer-by-layer walkthrough of the complex-valued transformer, plus release history.
          </p>
        </div>
      </section>

      {/* Architecture Deep Dive */}
      <section className="py-8">
        <div className="max-w-5xl mx-auto px-4">
          <h2 className="text-base font-bold text-purple-400 uppercase tracking-wider mb-1">Architecture Deep Dive</h2>
          <p className="text-zinc-500 text-xs mb-5">
            Data flow: <span className="text-green-400">Input</span> → <span className="text-purple-400">Embed</span> → <span className="text-blue-400">Attention</span> → <span className="text-cyan-400">FFN</span> → <span className="text-yellow-400">×6</span> → <span className="text-orange-400">MDN</span> → <span className="text-red-400">Loss</span>
          </p>

          <div className="space-y-3">
            {archLayers.map((layer, idx) => {
              const isOpen = expandedLayer === idx;
              return (
                <div
                  key={idx}
                  className={`rounded-xl border transition-all ${
                    isOpen
                      ? 'border-purple-500/40 bg-zinc-800/80'
                      : 'border-zinc-700/40 bg-zinc-800/40 hover:border-zinc-600'
                  }`}
                >
                  <button
                    onClick={() => setExpandedLayer(isOpen ? null : idx)}
                    className="w-full text-left p-4"
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 flex-wrap">
                          <h3 className="text-white font-bold text-sm">{layer.title}</h3>
                          <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-zinc-700/50 text-zinc-400 border border-zinc-600/50">
                            {layer.shape}
                          </span>
                        </div>
                        <p className="text-zinc-500 text-xs mt-1 leading-relaxed">{layer.idea}</p>
                      </div>
                      <svg
                        className={`w-4 h-4 text-zinc-500 flex-shrink-0 mt-1 transition-transform ${isOpen ? 'rotate-180' : ''}`}
                        fill="none" stroke="currentColor" viewBox="0 0 24 24"
                      >
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                      </svg>
                    </div>
                  </button>

                  {isOpen && (
                    <div className="px-4 pb-4 space-y-3">
                      <div className="rounded-lg bg-zinc-950 border border-zinc-700/50 overflow-hidden">
                        <div className="px-3 py-1.5 bg-zinc-800/80 border-b border-zinc-700/50">
                          <span className="text-[10px] font-mono text-zinc-500">Python — EqModelComplex.py</span>
                        </div>
                        <pre className="p-3 overflow-x-auto text-xs leading-relaxed">
                          <code className="text-zinc-300 font-mono whitespace-pre">{layer.code}</code>
                        </pre>
                      </div>
                      {layer.details && layer.details.length > 0 && (
                        <div className="space-y-1.5 pl-1">
                          {layer.details.map((detail, di) => (
                            <div key={di} className="flex items-start gap-2 text-xs">
                              <span className="text-purple-400 mt-0.5 flex-shrink-0">&#9656;</span>
                              <span className="text-zinc-400 leading-relaxed">{detail}</span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Changelog */}
      <section className="py-4 pb-12">
        <div className="max-w-5xl mx-auto px-4">
          <Changelog bare />
        </div>
      </section>

      <footer className="bg-zinc-900 border-t border-zinc-800 py-6">
        <div className="max-w-5xl mx-auto px-4 text-center">
          <Link to="/" className="text-orange-500 hover:text-orange-400 text-sm font-medium transition-colors">
            ← Back to Dashboard
          </Link>
        </div>
      </footer>
    </div>
  );
}
