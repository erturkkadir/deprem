import { useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { Routes, Route, Link } from 'react-router-dom';
import { fetchModelStatus, fetchStats, fetchLiveData, fetchPredictions } from './store/earthquakeSlice';

import Header from './components/Header';
import LiveDashboard from './components/LiveDashboard';
import PredictionsTable from './components/PredictionsTable';
import PredictionDetail from './components/PredictionDetail';
import About from './components/About';
import RealtimeMap from './components/RealtimeMap';
import AlertSubscription from './components/AlertSubscription';
import TrainingLossChart from './components/TrainingLossChart';
import Changelog from './components/Changelog';
import EarthquakeHistory from './components/EarthquakeHistory';
import HowItWorks from './components/HowItWorks';

function App() {
  const dispatch = useDispatch();
  const { isConnected, modelStatus } = useSelector((state) => state.earthquake);

  // Initial fetch and auto-polling
  useEffect(() => {
    // Initial data fetch
    dispatch(fetchModelStatus());
    dispatch(fetchStats());
    dispatch(fetchLiveData());
    dispatch(fetchPredictions({ limit: 20 }));

    // Poll for live data every 10 seconds
    const earthquakeInterval = setInterval(() => {
      dispatch(fetchLiveData());
    }, 10000);

    // Poll stats & predictions table every 2 minutes
    const statsInterval = setInterval(() => {
      dispatch(fetchStats());
      dispatch(fetchPredictions({ limit: 20 }));
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
    <Routes>
      <Route path="/map" element={<RealtimeMap />} />
      <Route path="/how-it-works" element={<HowItWorks />} />
      <Route path="/prediction/:id" element={<PredictionDetail />} />
      <Route path="*" element={
        <div className="min-h-screen bg-zinc-900">
          <Header />

          {/* Hero Section - Compact */}
          <section className="py-4">
            <div className="max-w-7xl mx-auto px-4">
              <div className="bg-gradient-to-br from-orange-600 to-orange-500 rounded-xl py-6 px-4 text-center">
                <h2 className="text-2xl md:text-3xl font-bold text-white mb-2">
                  AI-Powered Earthquake Prediction
                </h2>
                <p className="text-sm text-orange-100 mb-4 max-w-xl mx-auto">
                  Fully automated prediction using complex-valued transformer neural networks
                </p>

                <div className="flex flex-wrap justify-center gap-2 mb-4">
                  <div className="flex items-center gap-1.5 bg-white/10 backdrop-blur px-3 py-1.5 rounded-full">
                    <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-400 animate-pulse' : 'bg-red-400'}`} />
                    <span className="text-white text-xs">
                      {isConnected ? 'Online' : 'Offline'}
                    </span>
                  </div>
                  <div className="flex items-center gap-1.5 bg-white/10 backdrop-blur px-3 py-1.5 rounded-full">
                    <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
                    </svg>
                    <span className="text-white text-xs font-mono">
                      {modelStatus?.currentCheckpoint?.replace('eqModel_complex_', '').replace('.pth', '') || '...'}
                    </span>
                  </div>
                  {modelStatus?.device === 'cuda' && (
                    <div className="flex items-center gap-1.5 bg-white/10 backdrop-blur px-3 py-1.5 rounded-full">
                      <span className="text-white text-xs">GPU</span>
                    </div>
                  )}
                </div>

                {/* Action Links */}
                <div className="flex flex-wrap justify-center gap-3">
                  <Link
                    to="/map"
                    className="inline-flex items-center gap-2 bg-zinc-900 hover:bg-zinc-800 border-2 border-white px-6 py-3 rounded-full text-white font-bold transition-all hover:scale-105 shadow-lg shadow-black/30 group"
                  >
                    <svg className="w-6 h-6 text-green-400 group-hover:animate-pulse" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <span>View Real-Time Earthquake Map</span>
                    <svg className="w-5 h-5 text-green-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </Link>
                  <Link
                    to="/how-it-works"
                    className="inline-flex items-center gap-2 bg-white/10 hover:bg-white/20 backdrop-blur border border-white/30 px-5 py-3 rounded-full text-white font-medium transition-all hover:scale-105 group"
                  >
                    <svg className="w-5 h-5 text-orange-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                    </svg>
                    <span>How It Works</span>
                    <svg className="w-4 h-4 text-orange-300 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </Link>
                </div>
              </div>
            </div>
          </section>

          <LiveDashboard />
          <AlertSubscription />
          <PredictionsTable />
          <About />
          <Changelog />

          <EarthquakeHistory />

          {/* Training Progress Section */}
          <section className="py-6 bg-zinc-900">
            <div className="max-w-7xl mx-auto px-4">
              <TrainingLossChart />
            </div>
          </section>

          {/* Footer */}
          <footer className="bg-zinc-900 border-t border-zinc-800 py-8">
            <div className="max-w-7xl mx-auto px-4 text-center">
              <p className="text-zinc-500 text-sm">
                Earthquake Prediction System — Fully Automated, No User Interaction Required
              </p>
              <p className="text-zinc-600 text-xs mt-2">
                Data sourced from EMSC every 2 minutes | Predictions verified after 2 hours
              </p>
            </div>
          </footer>
        </div>
      } />
    </Routes>
  );
}

export default App;
