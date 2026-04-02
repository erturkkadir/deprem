import { useEffect, lazy, Suspense } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { Routes, Route, Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { fetchModelStatus, fetchStats, fetchLiveData, fetchPredictions } from './store/earthquakeSlice';

// Critical path: loaded immediately (landing page)
import Header from './components/Header';
import LiveDashboard from './components/LiveDashboard';
import PredictionsTable from './components/PredictionsTable';
import About from './components/About';

// Lazy-loaded: only fetched when user navigates to these routes
const PredictionDetail = lazy(() => import('./components/PredictionDetail'));
const RealtimeMap = lazy(() => import('./components/RealtimeMap'));
const AlertSubscription = lazy(() => import('./components/AlertSubscription'));
const TrainingLossChart = lazy(() => import('./components/TrainingLossChart'));
const Changelog = lazy(() => import('./components/Changelog'));
const EarthquakeHistory = lazy(() => import('./components/EarthquakeHistory'));
const HowItWorks = lazy(() => import('./components/HowItWorks'));
const Code = lazy(() => import('./components/Code'));

function LazyFallback() {
  return (
    <div className="min-h-screen bg-zinc-900 flex items-center justify-center">
      <div className="w-8 h-8 border-2 border-orange-500 border-t-transparent rounded-full animate-spin" />
    </div>
  );
}

function App() {
  const { t } = useTranslation();
  const dispatch = useDispatch();
  const { isConnected, modelStatus } = useSelector((state) => state.earthquake);

  // Initial fetch and auto-polling
  useEffect(() => {
    // Initial data fetch
    dispatch(fetchModelStatus());
    dispatch(fetchStats());
    dispatch(fetchLiveData());
    // PredictionsTable fetches its own data on mount with the correct filter

    // Poll for live data every 1 minute
    const earthquakeInterval = setInterval(() => {
      dispatch(fetchLiveData());
    }, 60000);

    // Poll stats every 2 minutes
    // Note: do NOT call fetchPredictions here — PredictionsTable owns its own
    // paginated/filtered state; calling it here overwrites the user's active filter
    const statsInterval = setInterval(() => {
      dispatch(fetchStats());
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
    <Suspense fallback={<LazyFallback />}>
    <Routes>
      <Route path="/map" element={<RealtimeMap />} />
      <Route path="/how-it-works" element={<HowItWorks />} />
      <Route path="/code" element={<Code />} />
      <Route path="/prediction/:id" element={<PredictionDetail />} />
      <Route path="/alerts" element={
        <div className="min-h-screen bg-zinc-900">
          <header className="bg-zinc-900 border-b border-zinc-800">
            <div className="max-w-4xl mx-auto px-4 py-4 flex items-center justify-between">
              <Link to="/" className="flex items-center gap-2 text-zinc-400 hover:text-white transition-colors">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                </svg>
                <span className="text-sm">{t('common.backToDashboard')}</span>
              </Link>
              <span className="text-zinc-600 text-xs font-mono">quake.syshuman.com</span>
            </div>
          </header>
          <div className="max-w-4xl mx-auto px-4 py-8">
            <h1 className="text-3xl font-bold text-white mb-2">{t('alerts.title')}</h1>
            <p className="text-zinc-400 text-base mb-8">{t('alerts.subtitle')}</p>
            <AlertSubscription alwaysOpen />
          </div>
        </div>
      } />
      <Route path="/history" element={
        <div className="min-h-screen bg-zinc-900">
          <header className="bg-zinc-900 border-b border-zinc-800">
            <div className="max-w-4xl mx-auto px-4 py-4 flex items-center justify-between">
              <Link to="/" className="flex items-center gap-2 text-zinc-400 hover:text-white transition-colors">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                </svg>
                <span className="text-sm">{t('common.backToDashboard')}</span>
              </Link>
              <span className="text-zinc-600 text-xs font-mono">quake.syshuman.com</span>
            </div>
          </header>
          <div className="max-w-4xl mx-auto px-4 py-8">
            <h1 className="text-3xl font-bold text-white mb-2">{t('history.title')}</h1>
            <p className="text-zinc-400 text-base mb-8">{t('history.subtitle')}</p>
            <EarthquakeHistory alwaysOpen />
          </div>
        </div>
      } />
      <Route path="*" element={
        <div className="min-h-screen bg-zinc-900">
          <Header />

          {/* Hero — text only */}
          <section className="py-4">
            <div className="max-w-7xl mx-auto px-4">
              <div className="bg-gradient-to-br from-orange-600 to-orange-500 rounded-xl py-8 px-4 text-center">
                <h2 className="text-2xl md:text-3xl font-bold text-white mb-2">
                  {t('hero.title')}
                </h2>
                <p className="text-sm text-orange-100 mb-4 max-w-xl mx-auto">
                  {t('hero.subtitle')}
                </p>
                <div className="flex flex-wrap justify-center gap-2">
                  <div className="flex items-center gap-1.5 bg-white/10 backdrop-blur px-3 py-1.5 rounded-full">
                    <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-400 animate-pulse' : 'bg-red-400'}`} />
                    <span className="text-white text-xs">{isConnected ? t('hero.connected') : t('hero.offline')}</span>
                  </div>
                  <div className="flex items-center gap-1.5 bg-white/10 backdrop-blur px-3 py-1.5 rounded-full">
                    <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
                    </svg>
                    <span className="text-white text-xs font-mono">
                      {modelStatus?.currentCheckpoint?.replace('eqModel_complex_', '').replace('.pth', '') || '...'}
                    </span>
                  </div>
                  <div className="flex items-center gap-1.5 bg-white/10 backdrop-blur px-3 py-1.5 rounded-full">
                    <span className="text-white text-xs font-mono">
                      {modelStatus?.device?.toUpperCase() || 'CPU'}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </section>

          <LiveDashboard />
          <PredictionsTable />
          <About />

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
                {t('footer.mainText')}
              </p>
              <p className="text-zinc-600 text-xs mt-2">
                {t('footer.dataSource')}
              </p>
            </div>
          </footer>
        </div>
      } />
    </Routes>
    </Suspense>
  );
}

export default App;
