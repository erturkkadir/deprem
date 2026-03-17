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
        heading: 'Complex-valued transformer (203M parameters)',
        text: 'The architecture is a 6-layer transformer with 8 attention heads, operating entirely in complex number space. Each layer has complex multi-head attention (with QK-Norm and learned per-dimension scaling) and complex gated feed-forward networks, both wrapped in pre+post normalization. The model processes sequences of 384 earthquakes and predicts the next significant event through Mixture Density Network output heads.',
      },
      {
        heading: 'How input becomes prediction',
        text: 'Each earthquake\'s 12 features are independently embedded into complex vectors (total embedding dimension: 1024), then concatenated. The sequence passes through 6 transformer layers. The final output goes through two Mixture Density Network heads: SpatialMDNHead (K=20 bivariate Gaussians for joint latitude/longitude) and MagnitudeMDNHead (K=8 univariate Gaussians). At inference, the highest-weight component is used as the prediction, along with a sigma_km uncertainty radius.',
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
        text: 'The model outputs K=20 spatial Gaussians competing to explain each target. The loss is the negative log-likelihood of the true location under the mixture — this forces multiple components to cover different seismically active regions. Magnitude uses K=8 Gaussians. An entropy regularizer keeps components diverse rather than collapsing. A diversity loss (Gaussian repulsion, τ=30°) prevents all 20 spatial components from collapsing to a single seismic zone.',
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
        heading: '90-minute prediction window',
        text: 'Each prediction has a 90-minute validity window. During this window, the system monitors incoming earthquakes for a match. A prediction is considered successful if an M4+ earthquake occurs within 250km of the predicted location during the window. When a prediction expires, a new one is created immediately. Missed predictions are rechecked for up to 48 hours (late catch) — a match in that window is recorded as a Late Catch.',
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


export default function HowItWorks() {
  const [openSection, setOpenSection] = useState(null);

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


        </div>
      </section>

      {/* Technical specs — always visible compact grid */}
      <section className="py-8">
        <div className="max-w-4xl mx-auto px-4">
          <h2 className="text-sm font-bold text-zinc-400 uppercase tracking-wider mb-4">Technical Specifications</h2>
          <div className="grid grid-cols-3 md:grid-cols-6 gap-2">
            {[
              { label: 'Parameters', value: '203M' },
              { label: 'Features', value: '12' },
              { label: 'Seq Length', value: '384' },
              { label: 'Embed Dim', value: '1,024' },
              { label: 'Heads', value: '8' },
              { label: 'Layers', value: '6' },
              { label: 'Spatial MDN', value: 'K=20' },
              { label: 'Mag MDN', value: 'K=8' },
              { label: 'Data', value: '1.5M+' },
              { label: 'Input', value: 'M2.0+' },
              { label: 'Window', value: '90 min' },
              { label: 'Match Radius', value: '250 km' },
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
