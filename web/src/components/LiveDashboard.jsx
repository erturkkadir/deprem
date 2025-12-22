import { useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { fetchLiveData } from '../store/earthquakeSlice';

function LiveDashboard() {
  const dispatch = useDispatch();
  const { liveData, isLoadingLive } = useSelector((state) => state.earthquake);

  // Auto-refresh every 30 seconds
  useEffect(() => {
    dispatch(fetchLiveData());

    const interval = setInterval(() => {
      dispatch(fetchLiveData());
    }, 30000);

    return () => clearInterval(interval);
  }, [dispatch]);

  const { latest_prediction, recent_earthquakes, stats, timestamp } = liveData || {};

  const formatTime = (isoString) => {
    if (!isoString) return 'N/A';
    const date = new Date(isoString);
    return date.toLocaleString();
  };

  const formatTimeAgo = (isoString) => {
    if (!isoString) return '';
    const date = new Date(isoString);
    const now = new Date();
    const diff = Math.floor((now - date) / 1000);

    if (diff < 60) return `${diff}s ago`;
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    return `${Math.floor(diff / 86400)}d ago`;
  };

  const getStatusBadge = (prediction) => {
    if (!prediction) return null;

    if (!prediction.verified) {
      return (
        <span className="px-2 py-1 bg-yellow-500/20 text-yellow-400 text-xs rounded-full animate-pulse">
          Pending
        </span>
      );
    }

    if (prediction.correct) {
      return (
        <span className="px-2 py-1 bg-green-500/20 text-green-400 text-xs rounded-full flex items-center gap-1">
          <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
          </svg>
          Correct
        </span>
      );
    }

    return (
      <span className="px-2 py-1 bg-red-500/20 text-red-400 text-xs rounded-full flex items-center gap-1">
        <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
        </svg>
        Incorrect
      </span>
    );
  };

  return (
    <section className="py-8 bg-zinc-900">
      <div className="max-w-7xl mx-auto px-4">
        {/* Section Header */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-2xl font-bold text-white flex items-center gap-2">
              <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse" />
              Live Dashboard
            </h2>
            <p className="text-zinc-400 text-sm mt-1">
              Auto-updates every 30 seconds | Last update: {formatTime(timestamp)}
            </p>
          </div>

          {isLoadingLive && (
            <div className="flex items-center gap-2 text-zinc-400">
              <svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              Updating...
            </div>
          )}
        </div>

        {/* Main Grid */}
        <div className="grid lg:grid-cols-2 gap-6">
          {/* Left: Latest Prediction */}
          <div className="card p-6">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
              <svg className="w-5 h-5 text-orange-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
              Latest Prediction
            </h3>

            {latest_prediction ? (
              <div className="space-y-4">
                {/* Prediction Grid */}
                <div className="grid grid-cols-2 gap-3">
                  {/* Latitude */}
                  <div className="bg-zinc-800/50 rounded-lg p-3">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-zinc-400 text-xs">Latitude</span>
                      {getStatusBadge(latest_prediction)}
                    </div>
                    <div className="text-2xl font-bold text-orange-500">
                      {latest_prediction.predicted_lat?.toFixed(1)}°
                    </div>
                    {latest_prediction.verified && latest_prediction.actual_lat !== null && (
                      <div className="text-xs text-zinc-500 mt-1">
                        Actual: {latest_prediction.actual_lat?.toFixed(1)}° (diff: {latest_prediction.diff_lat}°)
                      </div>
                    )}
                  </div>

                  {/* Longitude */}
                  <div className="bg-zinc-800/50 rounded-lg p-3">
                    <span className="text-zinc-400 text-xs block mb-1">Longitude</span>
                    <div className="text-2xl font-bold text-blue-500">
                      {latest_prediction.predicted_lon?.toFixed(1)}°
                    </div>
                    {latest_prediction.verified && latest_prediction.actual_lon !== null && (
                      <div className="text-xs text-zinc-500 mt-1">
                        Actual: {latest_prediction.actual_lon?.toFixed(1)}° (diff: {latest_prediction.diff_lon}°)
                      </div>
                    )}
                  </div>

                  {/* Time Difference */}
                  <div className="bg-zinc-800/50 rounded-lg p-3">
                    <span className="text-zinc-400 text-xs block mb-1">Time to Next EQ</span>
                    <div className="text-2xl font-bold text-purple-500">
                      {latest_prediction.predicted_dt} min
                    </div>
                    {latest_prediction.verified && latest_prediction.actual_dt !== null && (
                      <div className="text-xs text-zinc-500 mt-1">
                        Actual: {latest_prediction.actual_dt} min (diff: {latest_prediction.diff_dt} min)
                      </div>
                    )}
                  </div>

                  {/* Magnitude */}
                  <div className="bg-zinc-800/50 rounded-lg p-3">
                    <span className="text-zinc-400 text-xs block mb-1">Magnitude</span>
                    <div className="text-2xl font-bold text-red-500">
                      M{latest_prediction.predicted_mag?.toFixed(1)}
                    </div>
                    {latest_prediction.verified && latest_prediction.actual_mag !== null && (
                      <div className="text-xs text-zinc-500 mt-1">
                        Actual: M{latest_prediction.actual_mag?.toFixed(1)} (diff: {latest_prediction.diff_mag?.toFixed(1)})
                      </div>
                    )}
                  </div>
                </div>

                {/* Timestamp */}
                <div className="text-zinc-500 text-xs text-center">
                  Predicted at: {formatTime(latest_prediction.timestamp)}
                </div>

                {/* Latitude/Longitude Visualization */}
                <div className="bg-zinc-800/50 rounded-lg p-4">
                  <div className="text-zinc-400 text-xs mb-2">Location Scale</div>

                  {/* Latitude bar */}
                  <div className="mb-3">
                    <div className="text-zinc-500 text-xs mb-1">Latitude (-90° to +90°)</div>
                    <div className="relative h-3 bg-zinc-700 rounded-full overflow-hidden">
                      <div
                        className="absolute top-0 bottom-0 w-1 bg-orange-500"
                        style={{
                          left: `${((latest_prediction.predicted_lat + 90) / 180) * 100}%`,
                        }}
                      />
                      {latest_prediction.verified && latest_prediction.actual_lat !== null && (
                        <div
                          className="absolute top-0 bottom-0 w-1 bg-green-500"
                          style={{
                            left: `${((latest_prediction.actual_lat + 90) / 180) * 100}%`,
                          }}
                        />
                      )}
                    </div>
                  </div>

                  {/* Longitude bar */}
                  <div>
                    <div className="text-zinc-500 text-xs mb-1">Longitude (-180° to +180°)</div>
                    <div className="relative h-3 bg-zinc-700 rounded-full overflow-hidden">
                      <div
                        className="absolute top-0 bottom-0 w-1 bg-blue-500"
                        style={{
                          left: `${((latest_prediction.predicted_lon + 180) / 360) * 100}%`,
                        }}
                      />
                      {latest_prediction.verified && latest_prediction.actual_lon !== null && (
                        <div
                          className="absolute top-0 bottom-0 w-1 bg-green-500"
                          style={{
                            left: `${((latest_prediction.actual_lon + 180) / 360) * 100}%`,
                          }}
                        />
                      )}
                    </div>
                  </div>

                  <div className="flex gap-4 mt-3 text-xs justify-center">
                    <span className="flex items-center gap-1">
                      <div className="w-2 h-2 bg-orange-500 rounded" /> Lat Predicted
                    </span>
                    <span className="flex items-center gap-1">
                      <div className="w-2 h-2 bg-blue-500 rounded" /> Lon Predicted
                    </span>
                    <span className="flex items-center gap-1">
                      <div className="w-2 h-2 bg-green-500 rounded" /> Actual
                    </span>
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center py-8 text-zinc-500">
                No predictions yet. System will make predictions automatically.
              </div>
            )}
          </div>

          {/* Right: Recent Earthquakes */}
          <div className="card p-6">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
              <svg className="w-5 h-5 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 18.657A8 8 0 016.343 7.343S7 9 9 10c0-2 .5-5 2.986-7C14 5 16.09 5.777 17.656 7.343A7.975 7.975 0 0120 13a7.975 7.975 0 01-2.343 5.657z" />
              </svg>
              Recent Earthquakes (M4.0+)
            </h3>

            <div className="space-y-2 max-h-[400px] overflow-y-auto custom-scrollbar">
              {recent_earthquakes && recent_earthquakes.length > 0 ? (
                recent_earthquakes.map((eq, index) => (
                  <div
                    key={eq.id || index}
                    className="bg-zinc-800/50 rounded-lg p-3 hover:bg-zinc-800 transition-colors"
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <span className={`
                            px-2 py-0.5 rounded text-xs font-bold
                            ${eq.mag >= 6 ? 'bg-red-500 text-white' :
                              eq.mag >= 5 ? 'bg-orange-500 text-white' :
                              'bg-yellow-500 text-black'}
                          `}>
                            M{eq.mag?.toFixed(1)}
                          </span>
                          <span className="text-zinc-400 text-xs">
                            {formatTimeAgo(eq.time)}
                          </span>
                        </div>
                        <p className="text-white text-sm mt-1 truncate" title={eq.place}>
                          {eq.place || 'Unknown location'}
                        </p>
                        <div className="flex gap-4 mt-1 text-xs text-zinc-500">
                          <span>Lat: {eq.lat?.toFixed(2)}°</span>
                          <span>Lon: {eq.lon?.toFixed(2)}°</span>
                          <span>Depth: {eq.depth?.toFixed(1)} km</span>
                        </div>
                      </div>
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-center py-8 text-zinc-500">
                  No recent earthquakes data
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Success Rate Banner */}
        {stats && (
          <div className="mt-6 bg-gradient-to-r from-zinc-800 to-zinc-800/50 rounded-xl p-6 border border-zinc-700">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
              <div>
                <div className="text-3xl font-bold text-orange-500">
                  {stats.success_rate?.toFixed(1) || 0}%
                </div>
                <div className="text-zinc-400 text-sm">Success Rate</div>
              </div>
              <div>
                <div className="text-3xl font-bold text-white">
                  {stats.total_predictions || 0}
                </div>
                <div className="text-zinc-400 text-sm">Total Predictions</div>
              </div>
              <div>
                <div className="text-3xl font-bold text-green-500">
                  {stats.correct_predictions || 0}
                </div>
                <div className="text-zinc-400 text-sm">Correct</div>
              </div>
              <div>
                <div className="text-3xl font-bold text-zinc-400">
                  {stats.verified_predictions || 0}
                </div>
                <div className="text-zinc-400 text-sm">Verified</div>
              </div>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}

export default LiveDashboard;
