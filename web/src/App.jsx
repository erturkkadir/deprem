import { useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { fetchModelStatus, fetchStats } from './store/earthquakeSlice';

import Header from './components/Header';
import StatsGrid from './components/StatsGrid';
import PredictionPanel from './components/PredictionPanel';
import PredictionsTable from './components/PredictionsTable';
import About from './components/About';

function App() {
  const dispatch = useDispatch();
  const { isConnected } = useSelector((state) => state.earthquake);

  // Initial fetch and polling
  useEffect(() => {
    // Initial data fetch
    dispatch(fetchModelStatus());
    dispatch(fetchStats());

    // Poll for updates every 10 seconds
    const statsInterval = setInterval(() => {
      dispatch(fetchStats());
    }, 10000);

    // Check model status every 30 seconds
    const statusInterval = setInterval(() => {
      dispatch(fetchModelStatus());
    }, 30000);

    return () => {
      clearInterval(statsInterval);
      clearInterval(statusInterval);
    };
  }, [dispatch]);

  return (
    <div className="min-h-screen bg-zinc-900">
      <Header />

      {/* Hero Section */}
      <section className="bg-gradient-to-br from-orange-600 to-orange-500 py-16">
        <div className="max-w-7xl mx-auto px-4 text-center">
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-4">
            AI-Powered Earthquake Prediction
          </h2>
          <p className="text-xl text-orange-100 mb-8 max-w-2xl mx-auto">
            Using complex-valued transformer neural networks trained on 500,000+
            historical earthquakes from USGS
          </p>

          <div className="flex flex-wrap justify-center gap-4">
            <div className="flex items-center gap-2 bg-white/10 backdrop-blur px-4 py-2 rounded-full">
              <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-400' : 'bg-red-400'}`} />
              <span className="text-white text-sm">
                {isConnected ? 'Model Online' : 'Model Offline'}
              </span>
            </div>
            <div className="flex items-center gap-2 bg-white/10 backdrop-blur px-4 py-2 rounded-full">
              <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
              <span className="text-white text-sm">Real-time Updates</span>
            </div>
          </div>
        </div>
      </section>

      <StatsGrid />
      <PredictionPanel />
      <PredictionsTable />
      <About />

      {/* Footer */}
      <footer className="bg-zinc-900 border-t border-zinc-800 py-8">
        <div className="max-w-7xl mx-auto px-4 text-center">
          <p className="text-zinc-500 text-sm">
            Earthquake Prediction System — Powered by Complex-Valued Transformer AI
          </p>
          <p className="text-zinc-600 text-xs mt-2">
            Data sourced from USGS | Built with React + Redux + Tailwind
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
