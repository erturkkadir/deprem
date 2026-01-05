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
      title: 'Real-time USGS Data',
      description: 'Continuously updated with the latest earthquake data from the United States Geological Survey.',
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
        <p className="text-zinc-400 text-center max-w-2xl mx-auto mb-12">
          An experimental earthquake prediction system using deep learning to analyze
          patterns in historical seismic data.
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

        {/* Model Configuration */}
        {modelStatus.loaded && modelStatus.config && (
          <div className="card max-w-3xl mx-auto">
            <h3 className="text-lg font-semibold text-orange-400 mb-4">Model Configuration</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm mb-4">
              <div>
                <span className="text-zinc-500 block">Sequence Length</span>
                <span className="text-white font-mono">{modelStatus.config.sequence_length}</span>
              </div>
              <div>
                <span className="text-zinc-500 block">Embedding Size</span>
                <span className="text-white font-mono">{modelStatus.config.embedding_size}</span>
              </div>
              <div>
                <span className="text-zinc-500 block">Attention Heads</span>
                <span className="text-white font-mono">{modelStatus.config.num_heads}</span>
              </div>
              <div>
                <span className="text-zinc-500 block">Layers</span>
                <span className="text-white font-mono">{modelStatus.config.num_layers}</span>
              </div>
            </div>

            {/* Training Info */}
            <div className="border-t border-zinc-700 pt-4">
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
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
              </div>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
