import { useState } from 'react';
import { Link } from 'react-router-dom';

const sections = [
  {
    id: 'idea',
    title: 'The Core Idea',
    icon: (
      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
      </svg>
    ),
    content: [
      {
        heading: 'Can earthquakes be predicted?',
        text: 'The scientific consensus says no — earthquakes are chaotic, and no reliable short-term prediction method exists. This project challenges that assumption by treating seismic activity as a language: a sequence of events where patterns might exist but are too complex for humans to see.',
      },
      {
        heading: 'Earthquakes as a sequence problem',
        text: 'Just as a language model learns to predict the next word from context, our model learns to predict the next significant earthquake from a sequence of recent seismic events. Each earthquake is described by its location, magnitude, depth, timing, and environmental context — forming a "vocabulary" the model learns to read.',
      },
      {
        heading: 'Why complex-valued embeddings?',
        text: 'Standard neural networks embed features as real-valued vectors — a single number per dimension. Our model embeds each feature as a complex number, which has both a magnitude and a phase angle. This means each embedding dimension carries two pieces of information: "how strongly this feature is present" (magnitude) and "what role it plays in this context" (phase). Think of it as giving the model a richer, two-dimensional workspace to represent relationships between earthquake features. When the model combines embeddings through attention and feed-forward layers, it uses true complex multiplication — where magnitude and phase interact — rather than treating them as independent numbers.',
      },
      {
        heading: 'What does the complex space give us?',
        text: 'In a real-valued network, the relationship between "latitude 45" and "magnitude 5.0" is a single number (their dot product). In complex space, their relationship has both strength AND direction — the phase angle encodes how these features relate to each other in a way real numbers cannot. This is especially relevant for geospatial patterns: cyclic features like longitude (where 179° and -179° are neighbors), time of day, and lunar phase are naturally represented as rotations in complex space. The entire transformer — embeddings, attention, feed-forward layers, activations — operates in this complex domain.',
      },
    ],
  },
  {
    id: 'data',
    title: 'Data Pipeline',
    icon: (
      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4" />
      </svg>
    ),
    content: [
      {
        heading: 'Source: EMSC (European-Mediterranean Seismological Centre)',
        text: 'Every 2 minutes, the system pulls the latest earthquake data from EMSC, which reports global seismic events with 10-15 minute delay. Each event includes timestamp, latitude, longitude, depth, magnitude, and type classification.',
      },
      {
        heading: 'Feature encoding',
        text: 'Raw data is encoded into discrete bins the model can embed: latitude (181 bins covering -90° to 90°), longitude (361 bins covering -180° to 180°), magnitude (92 bins at 0.1 resolution), depth (201 bins in km). This discretization turns continuous values into a vocabulary.',
      },
      {
        heading: 'Derived temporal features',
        text: 'Two critical timing features are computed for each earthquake. Global dt: minutes since the last M4+ earthquake anywhere on Earth, log-binned into 10 bins following Omori\'s aftershock decay law (aftershock rate falls as 1/t). Local dt: minutes since the last M4+ earthquake within 1000km, log-binned into 26 bins — captures local aftershock cascades.',
      },
      {
        heading: 'Earth-state features',
        text: 'The model also receives contextual information about the state of the Earth at each event: hour of day (0-23, captures diurnal tidal stress from solid Earth tides), day of year (0-365, seasonal patterns + Earth-Sun distance variation), lunar phase (0-29, tidal force direction from new moon to full moon), and lunar distance (0-9, perigee to apogee — tidal amplitude varies ±12%).',
      },
    ],
  },
  {
    id: 'model',
    title: 'The Neural Network',
    icon: (
      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
      </svg>
    ),
    content: [
      {
        heading: 'Complex-valued transformer (357M parameters)',
        text: 'The architecture is an 8-layer transformer with 8 attention heads, operating entirely in complex number space. Each layer has complex multi-head attention (with QK-Norm and learned per-dimension scaling) and complex gated feed-forward networks, both wrapped in pre+post normalization. The model processes sequences of 512 earthquakes and predicts the next significant event through Mixture Density Network output heads.',
      },
      {
        heading: 'How input becomes prediction',
        text: 'Each earthquake\'s 12 features are independently embedded into complex vectors (total embedding dimension: 1176), then concatenated. The sequence passes through 8 transformer layers. The final output goes through two Mixture Density Network heads: SpatialMDNHead (K=20 bivariate Gaussians for joint latitude/longitude) and MagnitudeMDNHead (K=8 univariate Gaussians). At inference, the highest-weight component is used as the prediction, along with a sigma_km uncertainty radius.',
      },
      {
        heading: 'Spatial attention bias',
        text: 'Standard attention treats all earthquakes equally regardless of distance. Our model adds a spatial bias: nearby earthquakes naturally receive higher attention weights, with the decay following haversine (great-circle) distance. Each attention head learns its own distance-decay profile — half are initialized to prefer nearby events (aftershock patterns), half start neutral (tectonic interactions). Attention is stabilized by QK-Norm (RMSNorm on queries and keys) and PerDimScale (learned per-dimension importance weighting replacing fixed 1/√d).',
      },
      {
        heading: 'Geographic Positional Encoding (GPE)',
        text: 'Instead of treating latitude and longitude as independent numbers, GPE encodes location using spherical harmonics and Fourier features on the Earth\'s surface. This gives the model an intrinsic understanding of spherical geometry — it knows that longitude 179° and -179° are neighbors, not opposites.',
      },
      {
        heading: 'Hermitian attention',
        text: 'In complex-valued attention, the inner product must preserve complex structure. We use the Hermitian inner product: Re(Q)·Re(K) + Im(Q)·Im(K). This is the natural analog of the dot-product attention but for complex vectors, maintaining the coupling between real and imaginary parts that carries physical meaning.',
      },
    ],
  },
  {
    id: 'training',
    title: 'Training Strategy',
    icon: (
      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
      </svg>
    ),
    content: [
      {
        heading: 'Hybrid approach: see everything, predict the important',
        text: 'The model sees ALL earthquakes M2.0+ as input context (including small ones that form background seismicity patterns), but only predicts M4.0+ events as targets. This gives rich context — small foreshock patterns often precede significant events — while focusing learning on earthquakes that matter.',
      },
      {
        heading: 'Multi-position training with Omori decay',
        text: 'The model computes loss at every M4+ position within the sequence simultaneously (~400 target positions per batch), dramatically increasing training signal density. Critically, each position is weighted by its recency using Omori\'s aftershock decay law: the most recent event in the sequence gets weight 1.0×, while the oldest gets ~0.05×. This focuses the model on predicting near-future events rather than ancient history.',
      },
      {
        heading: 'Mixture Density Network loss (NLL)',
        text: 'The model outputs K=20 spatial Gaussians competing to explain each target. The loss is the negative log-likelihood of the true location under the mixture — this forces multiple components to cover different seismically active regions. Magnitude uses K=8 Gaussians with a Gutenberg-Richter prior (enforcing realistic b-value ~1.0 in the magnitude-frequency relationship). An entropy regularizer keeps components diverse rather than collapsing.',
      },
      {
        heading: 'Magnitude-weighted loss',
        text: 'Not all earthquakes are equal. An M6.0 prediction is far more valuable than M4.0. The loss function weights each target by its magnitude: M4.0→1×, M5.0→3.2×, M6.0→10×, M7.0→31.6×. Combined with the Omori temporal weight, the total weight prioritizes large and recent events.',
      },
      {
        heading: 'Haversine loss: min-k modal approach',
        text: 'An auxiliary haversine loss (3.0× weight) penalizes geographic distance. Crucially, we compute the distance from each of the K=20 Gaussian components to the target separately, then take the minimum. This pulls the CLOSEST component toward the target — encouraging geographic diversity across components — rather than pushing the mixture mean, which would average across all 20 and point at the ocean when the distribution is multimodal.',
      },
      {
        heading: 'Continuous learning',
        text: 'Training runs 24/7 on GPU (RTX 4060 Ti), saving checkpoints every 600 steps. The prediction server runs on CPU and automatically picks up new checkpoints. The model is retrained from scratch when significant architectural changes are made.',
      },
    ],
  },
  {
    id: 'prediction',
    title: 'Prediction & Verification',
    icon: (
      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
    ),
    content: [
      {
        heading: 'How predictions are made',
        text: 'Every 5 minutes, the server feeds the latest 512 earthquakes into the model. The MDN outputs 20 spatial Gaussian components — the highest-weight component gives the predicted (lat, lon). The prediction also includes sigma_km: the uncertainty radius computed from the Gaussian\'s standard deviations (σ_lat, σ_lon), indicating how confident the model is spatially.',
      },
      {
        heading: '120-minute prediction window',
        text: 'Each prediction has a 120-minute validity window. During this window, the system monitors incoming earthquakes for a match. A prediction is considered successful if an M4+ earthquake occurs within 500km of the predicted location during the window. When a prediction expires, a new one is created immediately.',
      },
      {
        heading: 'Transparent verification',
        text: 'All predictions are recorded and displayed with their outcomes — hit, miss, or pending. Success rates are computed from verified predictions only. The system does not hide failures; full transparency is a core design principle.',
      },
      {
        heading: 'Email alerts',
        text: 'Users can subscribe to receive email notifications when the model predicts an M4+ earthquake near their chosen location. Each alert includes a link to the interactive map showing the prediction details and a direct link to the prediction verification page.',
      },
    ],
  },
  {
    id: 'limitations',
    title: 'Honest Limitations',
    icon: (
      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
      </svg>
    ),
    content: [
      {
        heading: 'This is an experiment, not an oracle',
        text: 'Earthquake prediction remains one of the hardest unsolved problems in geophysics. No AI model, including this one, can reliably predict earthquakes. This system is a research project exploring whether deep learning can find patterns that traditional methods miss.',
      },
      {
        heading: 'Spatial accuracy is the biggest challenge',
        text: 'The model has learned to predict magnitude reasonably well (38% improvement over random baseline) but spatial prediction — knowing WHERE the next earthquake will strike — remains extremely difficult. Latitude and longitude predictions currently show only 14-15% improvement over random guessing.',
      },
      {
        heading: 'Do not rely on this for safety decisions',
        text: 'This system is for educational and research purposes. Do not use it to make evacuation decisions, cancel plans, or take any action that would normally require an official government seismic warning. Always follow your local authorities\' guidance.',
      },
      {
        heading: 'Statistics reset with architecture changes',
        text: 'When we make significant improvements to the model architecture, training restarts from scratch and all prediction statistics reset. This means the reported success rate reflects only the current version of the model, not cumulative history.',
      },
    ],
  },
];

const archLayers = [
  {
    title: '1. Input: Earthquake Sequence',
    shape: '[Batch, 512, 12]',
    idea: 'The model reads a sequence of 512 earthquakes, each described by 12 features. Like reading a paragraph of seismic events to predict the next sentence.',
    code: `# 12 features per earthquake:
# [year, month, lat, lon, mag, depth,
#  global_dt, local_dt, hour, doy,
#  moon_phase, moon_dist]
x = dataC.getLastFromDB(T=512)  # → [1, 512, 12]`,
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
    shape: '[B, 512, 12] → [B, 512, 1176] complex',
    idea: 'Each feature becomes a complex vector with magnitude + phase. Location gets 27% of capacity because WHERE matters most.',
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
      'GPE (lat+lon): 318 dims (27%) — Spherical Harmonics + Fourier + Tectonic Zones + Relative Position',
      'Global dt + Local dt: 117 dims each (10%) — discrete embedding + continuous log-scale encoding',
      '8 other features: 78 dims each (6.6%) — standard ComplexEmbedding lookup',
      'Total: 318 + 117 + 117 + (8×78) = 1,176 complex dims = 2,352 coupled real values',
    ],
  },
  {
    title: '3. Hermitian Multi-Head Attention',
    shape: '[B, 512, 1176] → [B, 512, 1176] complex',
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
    shape: '[B, 512, 1176] → [B, 512, 4704] → [B, 512, 1176]',
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
      '4× expansion (1176→4704) gives room for complex feature interactions before compressing back',
    ],
  },
  {
    title: '5. Transformer Block (×8 stacked)',
    shape: 'Pre-norm → Attn → Post-norm → Add → Pre-norm → FFN → Post-norm → Add',
    idea: '8 blocks stacked, each with 4 norms (pre+post for both attention and FFN). Early blocks learn "these events are near each other", later blocks learn "this aftershock cascade is accelerating toward a larger event".',
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

# Main model stacks 8 blocks (357M params total):
for block in self.blocks:      # 8 layers
    x = block(x, lat=lat, lon=lon)
x = self.ln_f(x)               # final norm`,
    details: [
      'Pre-norm (ComplexLayerNorm) + Post-norm (ComplexRMSNorm) — 4-norm design stabilizes residual stream',
      'Residual connections: each layer learns only the delta, not the full representation',
      '8 layers at n_embed=1176 fits RTX 4060 Ti (16 GB) with expandable_segments:True — 14.5 GB VRAM used',
    ],
  },
  {
    title: '6. Output: Mixture Density Network Heads',
    shape: '[B, T, 1176] complex → SpatialMDN(K=20) + MagnitudeMDN(K=8)',
    idea: 'Complex representation is split into real + imaginary (→2,352 dims). Two MDN heads predict full probability distributions — not just a point estimate. At inference, the highest-weight Gaussian component becomes the prediction.',
    code: `# Leave complex space: cat real and imaginary
x_combined = torch.cat([x.real, x.imag], dim=-1)  # [B, T, 2352]

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
      'Gutenberg-Richter prior in MagnitudeMDN enforces realistic b≈1.0 magnitude distribution',
      'pi (mixture weight) is returned as a confidence score per prediction',
    ],
  },
  {
    title: '7. Training Loss (5 signals)',
    shape: 'loss = NLL_spatial + NLL_mag + 3.0×hav + 0.1×GR + 0.5×entropy',
    idea: 'Five simultaneous training signals: MDN NLL teaches the full distribution, min-k haversine pulls the closest component toward targets, magnitude weighting prioritizes big quakes, Omori decay weights recent events more, GR prior and entropy keep the distribution well-behaved.',
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

# 4. Gutenberg-Richter prior (b≈1.0 regularization)
gr_loss = MagnitudeMDNHead.gr_prior_loss(mp)

# 5. Entropy regularizer (keep 20 components diverse)
ent_loss = SpatialMDNHead.entropy_loss(sp)  # max entropy = uniform pi

loss = (spatial_nll + mag_nll) * combined_w + 3.0*hav_loss + 0.1*gr_loss + 0.5*ent_loss`,
    details: [
      'Min-k haversine: gradient only flows through the CLOSEST component — creates geographic diversity, avoids mid-ocean averaging',
      'Omori decay: pos=0 (oldest) → 0.05× weight; pos=T-1 (newest) → 1.0× weight — model focuses on near-future',
      'Magnitude weights: M4.0→1×, M5.0→3.2×, M6.0→10×, M7.0→31.6×',
      'GR prior: penalizes non-b≈1.0 magnitude distributions — prevents collapse to single magnitude',
      'Entropy loss: prevents all 20 components from pointing at the same location',
      'Validation: pure NLL only (no aux losses, no weighting) — true generalization signal',
    ],
  },
];

export default function HowItWorks() {
  const [openSection, setOpenSection] = useState(null);
  const [expandedLayer, setExpandedLayer] = useState(null);

  const toggle = (id) => setOpenSection(openSection === id ? null : id);

  return (
    <div className="min-h-screen bg-zinc-900">
      {/* Header */}
      <header className="bg-zinc-900 border-b border-zinc-800">
        <div className="max-w-4xl mx-auto px-4 py-4 flex items-center justify-between">
          <Link to="/" className="flex items-center gap-2 text-zinc-400 hover:text-white transition-colors">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
            <span className="text-sm">Back to Dashboard</span>
          </Link>
          <span className="text-zinc-600 text-xs font-mono">quake.syshuman.com</span>
        </div>
      </header>

      {/* Title */}
      <section className="py-10">
        <div className="max-w-4xl mx-auto px-4">
          <h1 className="text-3xl md:text-4xl font-bold text-white mb-3">
            How It Works
          </h1>
          <p className="text-zinc-400 text-lg max-w-2xl">
            The idea, the reasoning, and the technical approach behind this AI earthquake prediction system.
          </p>

          {/* Flow diagram */}
          <div className="bg-zinc-800/50 rounded-xl p-5 border border-zinc-700 mt-8">
            <h3 className="text-zinc-400 text-xs font-semibold uppercase tracking-wider mb-4">System Flow</h3>
            <div className="flex flex-wrap items-center justify-center gap-2 text-xs">
              {[
                { label: 'EMSC Data', color: 'text-green-400 bg-green-500/10 border-green-500/30' },
                null,
                { label: 'MySQL DB', color: 'text-blue-400 bg-blue-500/10 border-blue-500/30' },
                null,
                { label: 'Feature Encoding', color: 'text-cyan-400 bg-cyan-500/10 border-cyan-500/30' },
                null,
                { label: 'Complex Transformer', color: 'text-purple-400 bg-purple-500/10 border-purple-500/30' },
                null,
                { label: 'Prediction', color: 'text-orange-400 bg-orange-500/10 border-orange-500/30' },
                null,
                { label: 'Verification', color: 'text-yellow-400 bg-yellow-500/10 border-yellow-500/30' },
              ].map((item, i) =>
                item === null ? (
                  <svg key={i} className="w-4 h-4 text-zinc-600 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                ) : (
                  <span key={i} className={`px-3 py-1.5 rounded-lg border font-medium ${item.color}`}>
                    {item.label}
                  </span>
                )
              )}
            </div>
          </div>
        </div>
      </section>

      {/* All sections as accordions */}
      <section className="pb-8">
        <div className="max-w-4xl mx-auto px-4 space-y-2">

          {/* Main sections */}
          {sections.map((section, sIdx) => {
            const isOpen = openSection === section.id;
            const isLimitations = section.id === 'limitations';
            const borderColor = isLimitations
              ? (isOpen ? 'border-amber-500/40' : 'border-amber-500/20 hover:border-amber-500/40')
              : (isOpen ? 'border-orange-500/40' : 'border-zinc-700/50 hover:border-zinc-600');
            const bgColor = isLimitations
              ? (isOpen ? 'bg-amber-500/5' : 'bg-zinc-800/40')
              : (isOpen ? 'bg-zinc-800/80' : 'bg-zinc-800/40');

            return (
              <div key={section.id} id={section.id} className={`rounded-xl border transition-all ${borderColor} ${bgColor}`}>
                <button
                  onClick={() => toggle(section.id)}
                  className="w-full text-left p-4"
                >
                  <div className="flex items-center justify-between gap-3">
                    <div className="flex items-center gap-3 min-w-0">
                      <div className={`w-9 h-9 rounded-lg flex items-center justify-center flex-shrink-0 ${
                        isLimitations ? 'bg-amber-500/20 text-amber-400' : 'bg-orange-500/15 text-orange-500'
                      }`}>
                        {section.icon}
                      </div>
                      <div className="min-w-0">
                        <div className="flex items-center gap-2">
                          <span className={`text-xs font-mono ${isLimitations ? 'text-amber-500' : 'text-orange-500'}`}>
                            {String(sIdx + 1).padStart(2, '0')}
                          </span>
                          <h2 className={`text-sm md:text-base font-bold ${isLimitations ? 'text-amber-400' : 'text-white'}`}>
                            {section.title}
                          </h2>
                        </div>
                        <p className="text-zinc-500 text-xs mt-0.5 truncate">
                          {section.content[0].heading}
                          {section.content.length > 1 && ` + ${section.content.length - 1} more`}
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center gap-2 flex-shrink-0">
                      <span className="text-zinc-600 text-xs hidden sm:inline">{section.content.length} items</span>
                      <svg
                        className={`w-5 h-5 text-zinc-500 transition-transform ${isOpen ? 'rotate-180' : ''}`}
                        fill="none" stroke="currentColor" viewBox="0 0 24 24"
                      >
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                      </svg>
                    </div>
                  </div>
                </button>

                {isOpen && (
                  <div className="px-4 pb-4">
                    <div className="border-t border-zinc-700/50 pt-4">
                      {isLimitations ? (
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                          {section.content.map((item, i) => (
                            <div key={i} className="bg-zinc-900/60 rounded-lg p-4 border border-amber-500/15">
                              <div className="flex items-start gap-2.5 mb-2">
                                <svg className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01" />
                                </svg>
                                <h3 className="text-white font-semibold text-sm">{item.heading}</h3>
                              </div>
                              <p className="text-zinc-400 text-sm leading-relaxed pl-7">{item.text}</p>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div className="space-y-4">
                          {section.content.map((item, i) => (
                            <div key={i} className="pl-4 border-l-2 border-zinc-700 hover:border-orange-500/50 transition-colors">
                              <h3 className="text-white font-semibold text-sm mb-1.5">{item.heading}</h3>
                              <p className="text-zinc-400 text-sm leading-relaxed">{item.text}</p>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            );
          })}

          {/* Architecture Deep Dive accordion */}
          {(() => {
            const isOpen = openSection === 'arch';
            return (
              <div className={`rounded-xl border transition-all ${
                isOpen ? 'border-purple-500/40 bg-zinc-800/80' : 'border-purple-500/20 bg-zinc-800/40 hover:border-purple-500/40'
              }`}>
                <button
                  onClick={() => toggle('arch')}
                  className="w-full text-left p-4"
                >
                  <div className="flex items-center justify-between gap-3">
                    <div className="flex items-center gap-3">
                      <div className="w-9 h-9 rounded-lg bg-purple-500/20 flex items-center justify-center flex-shrink-0">
                        <svg className="w-5 h-5 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
                        </svg>
                      </div>
                      <div>
                        <div className="flex items-center gap-2">
                          <span className="text-xs font-mono text-purple-400">07</span>
                          <h2 className="text-sm md:text-base font-bold text-white">Architecture Deep Dive</h2>
                        </div>
                        <p className="text-zinc-500 text-xs mt-0.5">
                          Layer-by-layer walkthrough with code from the model
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center gap-2 flex-shrink-0">
                      <span className="text-purple-400 text-xs font-mono hidden sm:inline">7 layers</span>
                      <svg
                        className={`w-5 h-5 text-zinc-500 transition-transform ${isOpen ? 'rotate-180' : ''}`}
                        fill="none" stroke="currentColor" viewBox="0 0 24 24"
                      >
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                      </svg>
                    </div>
                  </div>
                </button>

                {isOpen && (
                  <div className="px-4 pb-4">
                    <div className="border-t border-zinc-700/50 pt-4 space-y-3">
                      {/* Data flow */}
                      <div className="flex items-center gap-2 text-xs text-zinc-500 px-1 mb-2">
                        <span className="font-mono">Data flow:</span>
                        <span className="text-green-400">Input</span>
                        <span>→</span>
                        <span className="text-purple-400">Embed</span>
                        <span>→</span>
                        <span className="text-blue-400">Attention</span>
                        <span>→</span>
                        <span className="text-cyan-400">FFN</span>
                        <span>→</span>
                        <span className="text-yellow-400">×8</span>
                        <span>→</span>
                        <span className="text-orange-400">MDN</span>
                        <span>→</span>
                        <span className="text-red-400">Loss</span>
                      </div>

                      {archLayers.map((layer, idx) => {
                        const isLayerOpen = expandedLayer === idx;
                        return (
                          <div
                            key={idx}
                            className={`rounded-lg border transition-all ${
                              isLayerOpen
                                ? 'border-purple-500/30 bg-zinc-900/60'
                                : 'border-zinc-700/30 bg-zinc-900/30 hover:border-zinc-600'
                            }`}
                          >
                            <button
                              onClick={() => setExpandedLayer(isLayerOpen ? null : idx)}
                              className="w-full text-left p-3"
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
                                  className={`w-4 h-4 text-zinc-500 flex-shrink-0 mt-1 transition-transform ${isLayerOpen ? 'rotate-180' : ''}`}
                                  fill="none" stroke="currentColor" viewBox="0 0 24 24"
                                >
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                                </svg>
                              </div>
                            </button>

                            {isLayerOpen && (
                              <div className="px-3 pb-3 space-y-3">
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
                )}
              </div>
            );
          })()}

        </div>
      </section>

      {/* Technical specs — always visible compact grid */}
      <section className="py-8">
        <div className="max-w-4xl mx-auto px-4">
          <h2 className="text-sm font-bold text-zinc-400 uppercase tracking-wider mb-4">Technical Specifications</h2>
          <div className="grid grid-cols-3 md:grid-cols-6 gap-2">
            {[
              { label: 'Parameters', value: '357M' },
              { label: 'Features', value: '12' },
              { label: 'Seq Length', value: '512' },
              { label: 'Embed Dim', value: '1,176' },
              { label: 'Heads', value: '8' },
              { label: 'Layers', value: '8' },
              { label: 'Spatial MDN', value: 'K=20' },
              { label: 'Mag MDN', value: 'K=8' },
              { label: 'Data', value: '1.5M+' },
              { label: 'Input', value: 'M2.0+' },
              { label: 'Window', value: '120 min' },
              { label: 'Source', value: 'EMSC' },
            ].map((spec, i) => (
              <div key={i} className="bg-zinc-800/60 rounded-lg p-2.5 border border-zinc-700/50 text-center">
                <p className="text-zinc-500 text-[10px] uppercase tracking-wider">{spec.label}</p>
                <p className="text-white text-sm font-bold font-mono mt-0.5">{spec.value}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-zinc-900 border-t border-zinc-800 py-8">
        <div className="max-w-4xl mx-auto px-4 text-center">
          <Link to="/" className="text-orange-500 hover:text-orange-400 text-sm font-medium transition-colors">
            Back to Dashboard
          </Link>
          <p className="text-zinc-600 text-xs mt-3">
            This is an experimental research project. Not intended for safety-critical decisions.
          </p>
        </div>
      </footer>
    </div>
  );
}
