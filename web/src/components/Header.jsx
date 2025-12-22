import { useSelector } from 'react-redux';

export default function Header() {
  const { modelStatus, isConnected } = useSelector((state) => state.earthquake);

  return (
    <header className="bg-gradient-to-r from-zinc-900 to-zinc-800 border-b-2 border-orange-500 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 py-4">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <h1 className="text-2xl md:text-3xl font-bold text-orange-500">
              Earthquake Prediction System
            </h1>
            <p className="text-zinc-400 text-sm mt-1">
              Complex-valued Transformer Neural Network
            </p>
          </div>

          <div className="flex items-center gap-4">
            {/* Connection Status */}
            <div className="flex items-center gap-2">
              <div className="relative">
                <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}>
                  {isConnected && (
                    <div className="absolute inset-0 rounded-full bg-green-500 animate-ping opacity-75"></div>
                  )}
                </div>
              </div>
              <span className={`text-sm ${isConnected ? 'text-green-400' : 'text-red-400'}`}>
                {isConnected ? 'Connected' : 'Offline'}
              </span>
            </div>

            {/* Model Status */}
            {modelStatus.loaded && (
              <div className="hidden md:flex items-center gap-2 bg-zinc-800 px-3 py-1.5 rounded-full">
                <span className="text-xs text-zinc-400">Model:</span>
                <span className="text-xs text-orange-400 font-semibold">
                  {modelStatus.device?.toUpperCase()}
                </span>
              </div>
            )}
          </div>
        </div>
      </div>
    </header>
  );
}
