import { useSelector } from 'react-redux';

export default function About() {
  const { modelStatus } = useSelector((state) => state.earthquake);

  const features = [
    {
      icon: (
        <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
        </svg>
      ),
      title: 'Complex-Valued Neural Network',
      description: 'Uses complex-valued embeddings and attention mechanisms to capture phase relationships in seismic patterns.',
    },
    {
      icon: (
        <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 10V3L4 14h7v7l9-11h-7z" />
        </svg>
      ),
      title: 'Real-time EMSC Data',
      description: 'Continuously updated with the latest global earthquake data from the European-Mediterranean Seismological Centre (EMSC), enriched with Earth-state features like lunar phase and tidal forces.',
    },
    {
      icon: (
        <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
        </svg>
      ),
      title: 'Success Rate Tracking',
      description: 'Predictions are verified against actual events to maintain transparent accuracy metrics.',
    },
  ];

  return (
    <section className="py-16 bg-zinc-800/50">
      <div className="max-w-7xl mx-auto px-4">
        <h2 className="text-3xl font-bold text-center text-orange-500 mb-4">
          About This System
        </h2>
        <p className="text-zinc-400 text-center max-w-2xl mx-auto mb-4">
          An experimental earthquake prediction system using deep learning to analyze
          patterns in historical seismic data.
        </p>
        <p className="text-zinc-500 text-center text-sm max-w-2xl mx-auto mb-12">
          This project is under continuous development. The AI model is regularly retrained
          with new features and improvements, so predictions and statistics may reset periodically.
        </p>

        <div className="grid md:grid-cols-3 gap-6 mb-12">
          {features.map((feature, index) => (
            <div
              key={index}
              className="card hover:border-orange-500/50 transition-all duration-300 hover:-translate-y-1"
            >
              <div className="text-orange-500 mb-4">{feature.icon}</div>
              <h3 className="text-lg font-semibold text-white mb-2">{feature.title}</h3>
              <p className="text-zinc-400 text-sm">{feature.description}</p>
            </div>
          ))}
        </div>

        {/* Contact + Model Configuration — 3 columns matching feature grid */}
        <div className="grid md:grid-cols-3 gap-6">

          {/* Col 1: Architecture */}
          <div className="card">
            <h3 className="text-base font-semibold text-orange-400 mb-4">Architecture</h3>
            <div className="space-y-3 text-sm">
              <div>
                <span className="text-zinc-500 block">Sequence Length</span>
                <span className="text-white font-mono">{modelStatus.config?.sequence_length ?? '384'}</span>
              </div>
              <div>
                <span className="text-zinc-500 block">Embedding Size</span>
                <span className="text-white font-mono">{modelStatus.config?.embedding_size ?? '1024'}</span>
              </div>
              <div>
                <span className="text-zinc-500 block">Attention Heads</span>
                <span className="text-white font-mono">{modelStatus.config?.num_heads ?? '8'}</span>
              </div>
              <div>
                <span className="text-zinc-500 block">Layers</span>
                <span className="text-white font-mono">{modelStatus.config?.num_layers ?? '6'}</span>
              </div>
            </div>
          </div>

          {/* Col 2: Training Status */}
          <div className="card">
            <h3 className="text-base font-semibold text-orange-400 mb-4">Training Status</h3>
            <div className="space-y-3 text-sm">
              {modelStatus.currentCheckpoint && (
                <div>
                  <span className="text-zinc-500 block">Checkpoint</span>
                  <span className="text-orange-400 font-mono">
                    {modelStatus.currentCheckpoint.replace('eqModel_complex_', '').replace('.pth', '')}
                  </span>
                </div>
              )}
              {modelStatus.training?.latestStep && (
                <>
                  <div>
                    <span className="text-zinc-500 block">Training Step</span>
                    <span className="text-purple-400 font-mono">
                      {modelStatus.training.latestStep.toLocaleString()}
                    </span>
                  </div>
                  <div>
                    <span className="text-zinc-500 block">Loss</span>
                    <span className="text-cyan-400 font-mono">
                      {modelStatus.training.latestLoss?.toFixed(4)}
                    </span>
                  </div>
                </>
              )}
              {!modelStatus.loaded && (
                <p className="text-zinc-600 text-xs">Model not loaded</p>
              )}
            </div>
          </div>

          {/* Col 3: Contact */}
          <div className="card">
            <h3 className="text-base font-semibold text-orange-400 mb-1">Contact</h3>
            <p className="text-zinc-500 text-xs mb-4">Questions, feedback, or collaboration.</p>
            <div className="flex flex-col gap-2">
              <a
                href="mailto:kadirerturk@gmail.com"
                className="flex items-center gap-2 px-3 py-2 rounded-lg bg-zinc-800 hover:bg-zinc-700 border border-zinc-700 hover:border-orange-500/50 transition-all text-xs text-zinc-300 hover:text-white"
              >
                <svg className="w-3.5 h-3.5 text-orange-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                </svg>
                kadirerturk@gmail.com
              </a>
              <a
                href="https://x.com/kadirerturk"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 px-3 py-2 rounded-lg bg-zinc-800 hover:bg-zinc-700 border border-zinc-700 hover:border-zinc-400/50 transition-all text-xs text-zinc-300 hover:text-white"
              >
                <svg className="w-3.5 h-3.5 text-zinc-300 flex-shrink-0" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
                </svg>
                @kadirerturk
              </a>
              <a
                href="https://www.youtube.com/@KadirErturk"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 px-3 py-2 rounded-lg bg-zinc-800 hover:bg-zinc-700 border border-zinc-700 hover:border-red-500/50 transition-all text-xs text-zinc-300 hover:text-white"
              >
                <svg className="w-3.5 h-3.5 text-red-500 flex-shrink-0" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M23.498 6.186a3.016 3.016 0 00-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 00.502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 002.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 002.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z" />
                </svg>
                @KadirErturk
              </a>
            </div>
          </div>

        </div>
      </div>
    </section>
  );
}
