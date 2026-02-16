import { useState } from 'react';

const releases = [
  {
    version: 'v0.9',
    date: '2026-02-16',
    title: 'Haversine Loss, Magnitude Weighting & Moon Storage',
    type: 'enhancement',
    notes: [
      'Added Haversine auxiliary loss — model now directly penalizes geographic distance between predicted and actual positions, not just classification bin mismatch',
      'Magnitude-weighted training loss — larger earthquakes (M5+, M6+) receive exponentially higher loss weight (M5→3x, M6→10x), forcing the model to prioritize significant events',
      'Moon phase and distance now stored as persistent columns in the database (us_moon_phase, us_moon_dist) — previously computed on-the-fly',
      'New EMSC records automatically get moon features computed on ingestion',
      'Stored procedure get_data_hybrid updated to return moon data directly (13 columns)',
    ],
  },
  {
    version: 'v0.8',
    date: '2026-02-15',
    title: 'Earth-State Features & EMSC Migration',
    type: 'feature',
    notes: [
      'Added 4 new Earth-state input features: hour of day (0-23), day of year (0-365), lunar phase (0-29), lunar distance bin (0-9)',
      'Input tensor expanded from [B, T, 8] to [B, T, 12] — model now "feels" tidal forces and diurnal cycles',
      'Moon features computed via pure numpy orbital mechanics (synodic period 29.53 days, anomalistic period 27.55 days) — no external dependencies',
      'Embedding reallocation: GPE 318 (27%), dt/lt 117 each (10%), 8 simple features 78 each (6.6%) — total 1176 unchanged',
      'Migrated data source from USGS to EMSC (European-Mediterranean Seismological Centre) — 10-15 min delay vs 1.5-2 hours',
      'Email alerts now link to local interactive map instead of Google Maps',
    ],
  },
  {
    version: 'v0.7',
    date: '2026-02-10',
    title: 'Gaussian Label Smoothing & dt Prediction Removal',
    type: 'enhancement',
    notes: [
      'Removed time-difference (dt) prediction head — analysis showed dt prediction was pure noise with no learning signal',
      'Model reduced from 4 output heads to 3: latitude, longitude, magnitude',
      'Gaussian label smoothing replaces uniform smoothing — spreads probability as a Gaussian centered on true class (lat/lon σ=2.0, mag σ=2.0)',
      'Teaches spatial proximity: predicting 1° off is better than 90° off, unlike standard cross-entropy where all wrong bins are equal',
      'Validation loss computed WITHOUT smoothing for true generalization signal',
    ],
  },
  {
    version: 'v0.6',
    date: '2026-02-07',
    title: 'Local Seismicity Feature (us_lt)',
    type: 'feature',
    notes: [
      'Added us_lt: time since last M4+ earthquake within 1000km of each event — captures local aftershock sequences',
      'Uses SQL haversine distance with bounding box pre-filter for performance',
      'Log-binned to 26 bins matching Omori aftershock decay law',
      'Backfilled 1.5M records — 99.8% have a local M4+ predecessor, 0.2% use sentinel value',
      'Combined with global us_t, gives the model both global and local temporal context',
    ],
  },
  {
    version: 'v0.5',
    date: '2026-02-05',
    title: 'Email Alerts & Multi-Position Training',
    type: 'feature',
    notes: [
      'Email alert system — subscribe to get notified when AI predicts M4+ earthquakes near your chosen location',
      'Location picker with adjustable radius (250-1000km) and email verification flow',
      'Multi-position training — loss computed at ALL M4+ positions per sequence (~400 targets/batch) instead of just the last position',
      'Dramatically improved training signal density and convergence speed',
      'Hybrid training pipeline: M2.0+ earthquakes as input context, M4.0+ as prediction targets',
    ],
  },
  {
    version: 'v0.4',
    date: '2026-01-29',
    title: 'Full Complex-Valued Transformer',
    type: 'architecture',
    notes: [
      'Migrated from real-valued to full complex-valued neural network throughout',
      'ComplexLinear with magnitude-phase parameterization: W = |W| * exp(i*θ)',
      'Hermitian attention: Re(Q)@Re(K)^T + Im(Q)@Im(K)^T — preserves complex inner product structure',
      'Phase-preserving activations: ModReLU, ComplexSiLU, Cardioid gating',
      'Rotary Position Embeddings (RoPE) adapted for complex-valued representations',
      'Geographic Positional Encoding (GPE) with spatial attention bias using haversine distance decay',
    ],
  },
  {
    version: 'v0.3',
    date: '2026-01-13',
    title: 'Real-Time Map & Live Dashboard',
    type: 'feature',
    notes: [
      'Interactive real-time earthquake map with Leaflet — shows live earthquakes and AI predictions',
      'Live dashboard with current prediction, recent earthquake feed, and success rate statistics',
      'Prediction verification system — automatically matches predictions against actual M4+ events within time window and radius',
      'Sound notifications for new predictions (optional)',
    ],
  },
  {
    version: 'v0.2',
    date: '2025-12-22',
    title: 'Prediction System & Web Interface',
    type: 'feature',
    notes: [
      'Initial prediction server with automated 5-minute prediction cycles',
      'API endpoints for predictions, stats, and model status',
      'Web frontend with predictions table, model configuration display, and training progress chart',
      '4-parameter prediction: latitude, longitude, magnitude, time difference',
      'Geolocation-based matching to verify predictions against actual earthquakes',
    ],
  },
  {
    version: 'v0.1',
    date: '2024-04-16',
    title: 'Project Genesis',
    type: 'foundation',
    notes: [
      'Initial earthquake data collection pipeline from seismological APIs',
      'Database schema design for earthquake storage with encoded features (lat→0-180, lon→0-360, mag×10)',
      'Basic embedding model concept: embed earthquakes by year, month, location, magnitude, depth, and time difference',
      'Data preprocessing and training infrastructure setup',
    ],
  },
];

const typeColors = {
  feature: 'bg-green-500/15 text-green-400 border-green-500/30',
  enhancement: 'bg-blue-500/15 text-blue-400 border-blue-500/30',
  architecture: 'bg-purple-500/15 text-purple-400 border-purple-500/30',
  foundation: 'bg-orange-500/15 text-orange-400 border-orange-500/30',
};

const typeLabels = {
  feature: 'New Feature',
  enhancement: 'Enhancement',
  architecture: 'Architecture',
  foundation: 'Foundation',
};

export default function Changelog() {
  const [isOpen, setIsOpen] = useState(false);
  const [expandedVersion, setExpandedVersion] = useState(null);

  return (
    <section className="py-4">
      <div className="max-w-7xl mx-auto px-4">
        <div className="card">
          <button
            onClick={() => setIsOpen(!isOpen)}
            className="w-full flex items-center justify-between text-left"
          >
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-lg bg-purple-500/15 flex items-center justify-center flex-shrink-0">
                <svg className="w-5 h-5 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                </svg>
              </div>
              <div>
                <h3 className="text-white font-bold text-sm sm:text-base">Changelog</h3>
                <p className="text-zinc-500 text-xs mt-0.5">
                  Development history and release notes — {releases.length} releases since {releases[releases.length - 1].date}
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2 flex-shrink-0 ml-2">
              <span className="text-purple-400 text-xs font-mono hidden sm:inline">{releases[0].version}</span>
              <svg
                className={`w-5 h-5 text-zinc-400 transition-transform ${isOpen ? 'rotate-180' : ''}`}
                fill="none" stroke="currentColor" viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </div>
          </button>

          {isOpen && (
            <div className="mt-4 pt-4 border-t border-zinc-700">
              <div className="relative">
                {/* Timeline line */}
                <div className="absolute left-[17px] top-2 bottom-2 w-px bg-zinc-700" />

                <div className="space-y-1">
                  {releases.map((release, idx) => {
                    const isExpanded = expandedVersion === release.version;
                    const isLatest = idx === 0;

                    return (
                      <div key={release.version} className="relative pl-10">
                        {/* Timeline dot */}
                        <div className={`absolute left-2.5 top-3 w-3 h-3 rounded-full border-2 ${
                          isLatest
                            ? 'bg-purple-500 border-purple-400'
                            : 'bg-zinc-800 border-zinc-600'
                        }`} />

                        <button
                          onClick={() => setExpandedVersion(isExpanded ? null : release.version)}
                          className={`w-full text-left rounded-lg px-3 py-2.5 transition-colors ${
                            isExpanded ? 'bg-zinc-800/80' : 'hover:bg-zinc-800/40'
                          }`}
                        >
                          <div className="flex items-center gap-2 flex-wrap">
                            <span className="text-white font-mono text-sm font-bold">{release.version}</span>
                            <span className={`text-[10px] px-1.5 py-0.5 rounded border ${typeColors[release.type]}`}>
                              {typeLabels[release.type]}
                            </span>
                            {isLatest && (
                              <span className="text-[10px] px-1.5 py-0.5 rounded bg-orange-500/20 text-orange-400 border border-orange-500/30">
                                Latest
                              </span>
                            )}
                            <span className="text-zinc-600 text-xs ml-auto">{release.date}</span>
                          </div>
                          <p className="text-zinc-300 text-xs mt-1">{release.title}</p>
                        </button>

                        {isExpanded && (
                          <div className="px-3 pb-3">
                            <ul className="space-y-1.5 mt-1">
                              {release.notes.map((note, i) => (
                                <li key={i} className="flex items-start gap-2 text-xs text-zinc-400">
                                  <span className="text-zinc-600 mt-1 flex-shrink-0">&#8226;</span>
                                  <span>{note}</span>
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
