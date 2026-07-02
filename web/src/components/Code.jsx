import { useState } from 'react';
import { Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import Header from './Header';
import Changelog from './Changelog';

// Code snippets (language-independent — not translated)
const LAYER_CODE = [
  `# 12 features per earthquake:
# [year, month, lat, lon, mag, depth,
#  global_dt, local_dt, hour, doy,
#  moon_phase, moon_dist]
x = dataC.getLastFromDB(T=384)  # → [1, 384, 12]`,
  `class ComplexEmbedding(nn.Module):
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
  `def forward(self, x, lat=None, lon=None):
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
  `def forward(self, x):
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
  `class ComplexTransformerBlock(nn.Module):
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
  `# Leave complex space: cat real and imaginary
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
  `# 1. Winner-Takes-All NLL (CVPR 2019): only winning component
#    gets gradient for mu/sigma — prevents all-to-Pacific collapse
spatial_nll = SpatialMDNHead.wta_nll_loss(sp, lat, lon)  # 70% WTA + 30% full
mag_nll = MagnitudeMDNHead.nll_loss(mp, mag_actual)

# 2. Combined weight: magnitude × Omori temporal decay
mag_w = clamp(10 ** (0.5 * (mag - 4.0)), 1.0, 10.0)  # M6→10× (capped)
omori_w = exp(3.0 * (pos/(T-1) - 1.0))  # newest→1.0×, oldest→0.05×

# 3. Energy Score: proper multivariate scoring rule
# ES = E[d(X,y)] - 0.5·E[d(X,X')]
es_loss = SpatialMDNHead.energy_score_loss(sp, lat, lon)

# 4. Diversity loss: pairwise repulsion (τ=2° — Turkey scale)
div_loss = SpatialMDNHead.diversity_loss(sp)

# 5. Entropy regularizer (uniform pi weights)
ent_loss = SpatialMDNHead.entropy_loss(sp)

# 6. Sigma cap (σ>1°≈110km penalized) + Turkey bbox containment
sigma_reg = clamp(sigma_mean - 1.0, min=0.0)
bbox_loss = out_of_bbox_penalty(mu_lat, mu_lon)  # lat[35,43] lon[25,48]

# 7. Occurrence BCE: dense hazard signal at ALL 384 positions
occ_loss = BCEWithLogits(occ_logits, occ_labels)  # calibrated, no pos_weight

loss = spatial_nll + mag_nll + 3.0*es_loss + 1.0*div_loss
     + 0.5*ent_loss + 0.1*sigma_reg + 1.0*bbox_loss + 2.0*occ_loss`,
  `class OccurrenceHead(nn.Module):
    """P(Turkey M4+ within next 60 min | global event history)"""
    def __init__(self, in_dim, hidden_dim=512, base_rate=0.05):
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),   # 2048 → 512
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),        # → 1 hazard logit
        )
        # Final bias at log-odds of base rate → starts calibrated
        self.net[-1].bias.fill_(log(p / (1 - p)))

# Training label (DataClass): for EVERY global event i,
# occ_label[i] = 1 if a Turkey M4+ occurs within 60 min after it
tk_ts = sort(timestamps[turkey_m4_indices])
nxt = searchsorted(tk_ts, ts, side='right')   # next Turkey M4+
occ_label = (tk_ts[nxt] - ts <= 3600)          # base rate ~1.8%

# Server (alert gating): only speak when confident
p_event = sigmoid(occ_head(x_last))            # calibrated probability
if p_event >= ALERT_THRESHOLD:                 # 0.90
    issue_alert()      # real claim — counted in headline precision
    notify_subscribers()
else:
    monitor_only()     # no claim — excluded from headline stats`,
];

const LAYER_SHAPES = [
  '[Batch, 384, 12]',
  '[B, 384, 12] → [B, 384, 1024] complex',
  '[B, 384, 1024] → [B, 384, 1024] complex',
  '[B, 384, 1024] → [B, 384, 4096] → [B, 384, 1024]',
  'Pre-norm → Attn → Post-norm → Add → Pre-norm → FFN → Post-norm → Add',
  '[B, T, 1024] complex → SpatialMDN(K=20) + MagnitudeMDN(K=8)',
  'loss = WTA_NLL + NLL_mag + 3.0×ES + 1.0×div + 0.5×ent + 0.1×σ + bbox + 2.0×occ',
  '[B, T, 2048] → P(event) ∈ [0,1] → ALERT if p ≥ 0.90, else MONITOR',
];

export default function Code() {
  const [expandedLayer, setExpandedLayer] = useState(null);
  const { t } = useTranslation();

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
            <h1 className="text-2xl font-bold text-white">{t('code.title')}</h1>
          </div>
          <p className="text-zinc-500 text-sm ml-11">
            {t('code.subtitle')}
          </p>
        </div>
      </section>

      {/* Architecture Deep Dive */}
      <section className="py-8">
        <div className="max-w-5xl mx-auto px-4">
          <h2 className="text-base font-bold text-purple-400 uppercase tracking-wider mb-1">{t('code.archDeepDive')}</h2>
          <p className="text-zinc-500 text-xs mb-5">
            {t('code.dataFlow')} <span className="text-green-400">Input</span> → <span className="text-purple-400">Embed</span> → <span className="text-blue-400">Attention</span> → <span className="text-cyan-400">FFN</span> → <span className="text-yellow-400">×6</span> → <span className="text-orange-400">MDN</span> → <span className="text-red-400">Loss</span>
          </p>

          <div className="space-y-3">
            {LAYER_CODE.map((code, idx) => {
              const layerNum = idx + 1;
              const isOpen = expandedLayer === idx;
              const details = t(`arch.layer${layerNum}Details`, { returnObjects: true });
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
                          <h3 className="text-white font-bold text-sm">{t(`arch.layer${layerNum}Title`)}</h3>
                          <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-zinc-700/50 text-zinc-400 border border-zinc-600/50">
                            {LAYER_SHAPES[idx]}
                          </span>
                        </div>
                        <p className="text-zinc-500 text-xs mt-1 leading-relaxed">{t(`arch.layer${layerNum}Idea`)}</p>
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
                          <code className="text-zinc-300 font-mono whitespace-pre">{code}</code>
                        </pre>
                      </div>
                      {Array.isArray(details) && details.length > 0 && (
                        <div className="space-y-1.5 pl-1">
                          {details.map((detail, di) => (
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
            ← {t('code.backToDashboard')}
          </Link>
        </div>
      </footer>
    </div>
  );
}
