import { useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { fetchModelStatus, fetchStats, fetchLiveData, fetchPredictions } from './store/earthquakeSlice';

import Header from './components/Header';
import LiveDashboard from './components/LiveDashboard';
import StatsGrid from './components/StatsGrid';
import PredictionsTable from './components/PredictionsTable';
import About from './components/About';

function App() {
  const dispatch = useDispatch();
  const { isConnected, modelStatus } = useSelector((state) => state.earthquake);

  // Initial fetch and auto-polling
  useEffect(() => {
    // Initial data fetch
    dispatch(fetchModelStatus());
    dispatch(fetchStats());
    dispatch(fetchLiveData());
    dispatch(fetchPredictions(20));

    // Poll for earthquakes every 60 seconds (1 minute)
    // This fetches latest earthquakes - prediction updates are handled by LiveDashboard
    const earthquakeInterval = setInterval(() => {
      dispatch(fetchLiveData());
    }, 60000);

    // Poll stats & predictions table every 2 minutes
    const statsInterval = setInterval(() => {
      dispatch(fetchStats());
      dispatch(fetchPredictions(20));
    }, 120000);

    // Check model status every 5 minutes
    const statusInterval = setInterval(() => {
      dispatch(fetchModelStatus());
    }, 300000);

    return () => {
      clearInterval(earthquakeInterval);
      clearInterval(statsInterval);
      clearInterval(statusInterval);
    };
  }, [dispatch]);

  return (
    <div className="min-h-screen bg-zinc-900">
      <Header />

      {/* Hero Section */}
      <section className="py-8">
        <div className="max-w-7xl mx-auto px-4">
          <div className="bg-gradient-to-br from-orange-600 to-orange-500 rounded-xl py-12 px-6 text-center">
            <h2 className="text-4xl md:text-5xl font-bold text-white mb-4">
              AI-Powered Earthquake Prediction
            </h2>
            <p className="text-xl text-orange-100 mb-6 max-w-2xl mx-auto">
              Fully automated system predicting latitude, longitude, magnitude, and timing
              using complex-valued transformer neural networks
            </p>

            <div className="flex flex-wrap justify-center gap-4">
              <div className="flex items-center gap-2 bg-white/10 backdrop-blur px-4 py-2 rounded-full">
                <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-400 animate-pulse' : 'bg-red-400'}`} />
                <span className="text-white text-sm">
                  {isConnected ? 'System Online' : 'System Offline'}
                </span>
              </div>
              <div className="flex items-center gap-2 bg-white/10 backdrop-blur px-4 py-2 rounded-full">
                <svg className="w-4 h-4 text-white animate-spin-slow" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
                <span className="text-white text-sm">Auto-updates every 5 min</span>
              </div>
              <div className="flex items-center gap-2 bg-white/10 backdrop-blur px-4 py-2 rounded-full">
                <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
                </svg>
                <span className="text-white text-sm">
                  {modelStatus?.latestCheckpoint || 'Loading...'}
                </span>
              </div>
              {modelStatus?.device === 'cuda' && (
                <div className="flex items-center gap-2 bg-white/10 backdrop-blur px-4 py-2 rounded-full">
                  <span className="text-white text-sm">GPU Accelerated</span>
                </div>
              )}
            </div>
          </div>
        </div>
      </section>

      <LiveDashboard />
      <StatsGrid />
      <PredictionsTable />
      <About />

      {/* Footer */}
      <footer className="bg-zinc-900 border-t border-zinc-800 py-8">
        <div className="max-w-7xl mx-auto px-4 text-center">
          <p className="text-zinc-500 text-sm">
            Earthquake Prediction System — Fully Automated, No User Interaction Required
          </p>
          <p className="text-zinc-600 text-xs mt-2">
            Data sourced from USGS every 5 minutes | Predictions verified after 24 hours
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
