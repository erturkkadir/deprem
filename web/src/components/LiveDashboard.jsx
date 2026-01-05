import { useEffect, useState, useMemo, useRef } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { fetchLiveData } from '../store/earthquakeSlice';

// Helper to determine prediction status
function getPredictionStatus(pred) {
  if (!pred.verified) {
    if (pred.prediction_time) {
      const predTime = new Date(pred.prediction_time);
      const now = new Date();
      const predDate = new Date(predTime.getFullYear(), predTime.getMonth(), predTime.getDate());
      const todayDate = new Date(now.getFullYear(), now.getMonth(), now.getDate());
      if (predDate < todayDate) return 'missed';
    }
    return 'pending';
  }
  return pred.correct ? 'matched' : 'missed';
}

function LiveDashboard() {
  const dispatch = useDispatch();
  const { liveData, isLoadingLive, isPredicting } = useSelector((state) => state.earthquake);
  const [minMagFilter, setMinMagFilter] = useState(2);
  const [isFlashing, setIsFlashing] = useState(false);
  const [displayTime, setDisplayTime] = useState(new Date());
  const lastPredictionId = useRef(null);
  const lastMatchState = useRef(false);

  // Update display time every second (for UI refresh)
  useEffect(() => {
    const timer = setInterval(() => setDisplayTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  // Reset flash when prediction changes
  useEffect(() => {
    if (liveData?.latest_prediction?.id !== lastPredictionId.current) {
      lastPredictionId.current = liveData?.latest_prediction?.id;
      lastMatchState.current = false;
      setIsFlashing(false);
    }
  }, [liveData?.latest_prediction?.id]);

  // Extract all data from backend response
  const {
    latest_prediction,
    recent_earthquakes,
    stats,
    match_info,
    closest_match,
    prediction_status,
    server_time
  } = liveData || {};

  // Flash when a new match is detected
  useEffect(() => {
    const hasMatch = match_info?.is_match || match_info?.verified_match;
    if (hasMatch && !lastMatchState.current) {
      setIsFlashing(true);
      lastMatchState.current = true;
      setTimeout(() => setIsFlashing(false), 3000);
    }
  }, [match_info]);

  // Format time for display (local timezone)
  const formatTimeLocal = (isoString) => {
    if (!isoString) return 'N/A';
    const date = new Date(isoString);
    return date.toLocaleString('en-US', {
      month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit'
    });
  };

  // Format countdown/elapsed time from seconds
  const formatTimeRemaining = (seconds) => {
    if (seconds === null || seconds === undefined) return 'N/A';

    const absSeconds = Math.abs(seconds);
    const hours = Math.floor(absSeconds / 3600);
    const minutes = Math.floor((absSeconds % 3600) / 60);
    const secs = absSeconds % 60;

    let text;
    if (hours > 0) {
      text = `${hours}h ${minutes}m ${secs}s`;
    } else if (minutes > 0) {
      text = `${minutes}m ${secs}s`;
    } else {
      text = `${secs}s`;
    }

    return seconds < 0 ? `Expired ${text} ago` : text;
  };

  // Calculate progress (0-100) from backend status
  const getProgress = () => {
    if (!prediction_status || !latest_prediction?.predicted_dt) return 0;
    const totalSeconds = latest_prediction.predicted_dt * 60;
    const remaining = prediction_status.time_remaining_seconds;
    if (remaining <= 0) return 100;
    const elapsed = totalSeconds - remaining;
    return Math.min(100, Math.max(0, (elapsed / totalSeconds) * 100));
  };

  // Filter earthquakes by magnitude only (window filtering done by backend)
  const earthquakesInWindow = useMemo(() => {
    if (!recent_earthquakes || !prediction_status) return [];

    const startTime = new Date(prediction_status.start_time);
    const endTime = new Date(prediction_status.end_time);

    return recent_earthquakes
      .filter(eq => {
        if (eq.mag < minMagFilter) return false;
        const eqTime = new Date(eq.time);
        return eqTime >= startTime && eqTime <= endTime;
      })
      .map(eq => ({
        ...eq,
        distance: eq.distance ?? null,
        isMatch: eq.is_match ?? false
      }))
      .sort((a, b) => new Date(b.time) - new Date(a.time));
  }, [recent_earthquakes, minMagFilter, prediction_status]);

  // Get row background color based on magnitude
  const getMagRowColor = (mag, isMatch) => {
    if (isMatch) return 'bg-green-500/20 border border-green-500/50 shadow-lg shadow-green-500/20';
    if (mag >= 7) return 'bg-purple-900/40 border border-purple-500/30';
    if (mag >= 6) return 'bg-red-900/40 border border-red-500/30';
    if (mag >= 5) return 'bg-orange-900/30 border border-orange-500/20';
    if (mag >= 4) return 'bg-yellow-900/20 border border-yellow-500/20';
    if (mag >= 3) return 'bg-zinc-800/50 border border-zinc-600/20';
    return 'bg-zinc-800/30 border border-transparent';
  };

  // Best match from API (already detected by backend)
  const bestMatch = match_info?.is_match ? earthquakesInWindow.find(eq => eq.id === match_info.earthquake_id) : null;
  const closestEq = closest_match ? earthquakesInWindow.find(eq => eq.id === closest_match.earthquake_id) : earthquakesInWindow[0];

  // Use stats from backend (full database counts, not current page)
  const predictionCounts = useMemo(() => {
    if (!stats) {
      return { total: 0, matched: 0, pending: 0, missed: 0, successRate: 0 };
    }
    const total = parseInt(stats.total_predictions) || 0;
    const verified = parseInt(stats.verified_predictions) || 0;
    const matched = parseInt(stats.correct_predictions) || 0;
    const missed = verified - matched;
    const pending = total - verified;
    const successRate = parseFloat(stats.success_rate) || 0;
    return { total, matched, pending, missed, successRate };
  }, [stats]);


  const formatTimeAgo = (isoString) => {
    if (!isoString) return '';
    const date = new Date(isoString);
    if (isNaN(date.getTime())) return '';
    const diff = Math.floor((new Date() - date) / 1000);
    if (diff < 60) return 'just now';
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    return `${Math.floor(diff / 86400)}d ago`;
  };

  // Circular progress component
  const CircularProgress = ({ progress, size = 120, children }) => {
    const radius = (size - 12) / 2;
    const circumference = 2 * Math.PI * radius;
    const strokeDashoffset = circumference - (progress / 100) * circumference;
    const isUrgent = progress > 80;
    const isClose = progress > 60;

    return (
      <div className="relative" style={{ width: size, height: size }}>
        <svg className="transform -rotate-90" width={size} height={size}>
          <circle cx={size/2} cy={size/2} r={radius} stroke="#374151" strokeWidth="8" fill="none" />
          <circle
            cx={size/2} cy={size/2} r={radius}
            stroke={isUrgent ? '#ef4444' : isClose ? '#f59e0b' : '#22c55e'}
            strokeWidth="8" fill="none"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            className={`transition-all duration-1000 ${isUrgent ? 'animate-pulse' : ''}`}
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          {children}
        </div>
      </div>
    );
  };

  // Match indicator component
  const MatchIndicator = ({ distance }) => {
    const matchThreshold = 15;
    const percentage = Math.max(0, Math.min(100, ((matchThreshold - distance) / matchThreshold) * 100));
    const isMatch = distance <= matchThreshold;

    return (
      <div className="flex items-center gap-2">
        <div className="flex-1 h-2 bg-zinc-700 rounded-full overflow-hidden">
          <div
            className={`h-full transition-all duration-500 ${isMatch ? 'bg-green-500' : percentage > 50 ? 'bg-yellow-500' : 'bg-zinc-500'}`}
            style={{ width: `${percentage}%` }}
          />
        </div>
        <span className={`text-xs font-mono ${isMatch ? 'text-green-400' : 'text-zinc-400'}`}>
          {distance.toFixed(1)}°
        </span>
      </div>
    );
  };

  return (
    <section className="py-4 bg-zinc-900">
      <div className="max-w-7xl mx-auto px-4">
        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <div className="relative">
              <div className="w-2.5 h-2.5 bg-green-500 rounded-full animate-pulse" />
              <div className="absolute inset-0 w-2.5 h-2.5 bg-green-500 rounded-full animate-ping" />
            </div>
            Live Monitor
          </h2>
          <span className="text-zinc-500 text-xs">Updates every 10s</span>
        </div>

        {/* Main Grid */}
        <div className="grid lg:grid-cols-3 gap-4">

          {/* Left: Prediction Card */}
          <div className="lg:col-span-2 space-y-3">
            {latest_prediction ? (
              <>
                {/* Unified Prediction Card */}
                <div className={`card !p-0 overflow-hidden transition-all duration-500 ${
                  match_info?.is_match || match_info?.verified_match
                    ? 'ring-2 ring-green-500/50'
                    : ''
                } ${isFlashing ? 'animate-flash-match' : ''}`}>
                  {/* Header with status and location */}
                  <div className={`p-3 ${
                    match_info?.is_match || match_info?.verified_match
                      ? 'bg-gradient-to-r from-green-600/20 to-green-500/10'
                      : 'bg-gradient-to-r from-orange-600/20 to-orange-500/10'
                  }`}>
                    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                      <div className="flex-1 min-w-0">
                        <div className="flex flex-wrap items-center gap-2 mb-1">
                          <span className={`px-2 py-0.5 rounded-full text-xs font-semibold ${
                            match_info?.is_match || match_info?.verified_match
                              ? 'bg-green-500 text-white animate-pulse'
                              : latest_prediction.verified
                                ? (latest_prediction.correct ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400')
                                : 'bg-yellow-500/20 text-yellow-400'
                          }`}>
                            {match_info?.is_match || match_info?.verified_match
                              ? '✓ MATCH!'
                              : latest_prediction.verified
                                ? (latest_prediction.correct ? 'Verified' : 'Missed')
                                : 'Pending'}
                          </span>
                          <span className="text-zinc-500 text-xs">#{latest_prediction.id}</span>
                        </div>
                        <h3 className="text-lg sm:text-xl font-bold text-white flex items-center gap-2 truncate">
                          <svg className="w-5 h-5 text-orange-500 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                          </svg>
                          <span className="truncate">{latest_prediction.predicted_place || 'Ocean Region'}</span>
                        </h3>
                      </div>
                      <a
                        href={`https://www.google.com/maps?q=${latest_prediction.predicted_lat},${latest_prediction.predicted_lon}&z=5`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex items-center justify-center gap-2 px-3 py-2 bg-orange-500 hover:bg-orange-600 text-white rounded-lg transition-colors text-sm whitespace-nowrap"
                      >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
                        </svg>
                        Map
                      </a>
                    </div>
                  </div>

                  {/* Main Content Grid */}
                  <div className="p-3">
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                      {/* Countdown Section */}
                      <div className="flex flex-col items-center justify-center p-2 bg-zinc-800/50 rounded-lg">
                        <CircularProgress progress={getProgress()} size={90}>
                          <div className="text-center">
                            {prediction_status?.is_expired ? (
                              <div className="text-red-400">
                                <svg className="w-5 h-5 mx-auto" fill="currentColor" viewBox="0 0 20 20">
                                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z" clipRule="evenodd" />
                                </svg>
                                <div className="text-[10px] mt-1">Expired</div>
                              </div>
                            ) : (
                              <>
                                <div className={`text-lg font-bold font-mono ${
                                  getProgress() > 80 ? 'text-red-400' :
                                  getProgress() > 60 ? 'text-yellow-400' : 'text-green-400'
                                }`}>
                                  {formatTimeRemaining(prediction_status?.time_remaining_seconds)}
                                </div>
                                <div className="text-zinc-500 text-[9px]">remaining</div>
                              </>
                            )}
                          </div>
                        </CircularProgress>
                      </div>

                      {/* Prediction Values */}
                      <div className="grid grid-cols-2 gap-1.5">
                        <div className="bg-zinc-800/50 rounded p-1.5 text-center">
                          <div className="text-[9px] text-zinc-500 uppercase">Lat</div>
                          <div className="text-base font-bold text-orange-500">{latest_prediction.predicted_lat?.toFixed(1)}°</div>
                        </div>
                        <div className="bg-zinc-800/50 rounded p-1.5 text-center">
                          <div className="text-[9px] text-zinc-500 uppercase">Lon</div>
                          <div className="text-base font-bold text-blue-500">{latest_prediction.predicted_lon?.toFixed(1)}°</div>
                        </div>
                        <div className="bg-zinc-800/50 rounded p-1.5 text-center">
                          <div className="text-[9px] text-zinc-500 uppercase">Mag</div>
                          <div className={`text-base font-bold ${
                            latest_prediction.predicted_mag >= 6 ? 'text-red-500' :
                            latest_prediction.predicted_mag >= 5 ? 'text-orange-500' : 'text-yellow-500'
                          }`}>
                            M{latest_prediction.predicted_mag?.toFixed(1)}
                          </div>
                        </div>
                        <div className="bg-zinc-800/50 rounded p-1.5 text-center">
                          <div className="text-[9px] text-zinc-500 uppercase">Window</div>
                          <div className="text-base font-bold text-purple-500">{latest_prediction.predicted_dt || 60}m</div>
                        </div>
                      </div>

                      {/* Time Window */}
                      <div className="sm:col-span-2 lg:col-span-1 bg-zinc-800/50 rounded-lg p-2">
                        <div className="text-[9px] text-zinc-500 uppercase mb-1.5 text-center">Time Window</div>
                        <div className="space-y-0.5">
                          <div className="flex items-center justify-between text-[11px]">
                            <span className="text-zinc-400">Start:</span>
                            <span className="text-green-400 font-mono">
                              {formatTimeLocal(prediction_status?.start_time)}
                            </span>
                          </div>
                          <div className="flex items-center justify-between text-[11px]">
                            <span className="text-zinc-400">End:</span>
                            <span className="text-red-400 font-mono">
                              {formatTimeLocal(prediction_status?.end_time)}
                            </span>
                          </div>
                          <div className="flex items-center justify-between text-[11px] pt-1 border-t border-zinc-700 mt-1">
                            <span className="text-zinc-400">Now:</span>
                            <span className="text-white font-mono">
                              {formatTimeLocal(prediction_status?.current_time)}
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Closest Match Indicator */}
                {closestEq && !latest_prediction.verified && (
                  <div className="card !p-3">
                    <div className="flex items-center justify-between mb-1.5">
                      <span className="text-zinc-400 text-xs">Closest Match</span>
                      <span className={`text-[10px] px-1.5 py-0.5 rounded ${
                        closestEq.isMatch ? 'bg-green-500/20 text-green-400' : 'bg-zinc-700 text-zinc-400'
                      }`}>
                        {closestEq.isMatch ? 'Match!' : `${closestEq.distance.toFixed(1)}° away`}
                      </span>
                    </div>
                    <MatchIndicator distance={closestEq.distance} />
                    {(closestEq?.isMatch || match_info?.is_match) && (
                      <div className="mt-2 p-2 bg-green-500/10 rounded border border-green-500/30">
                        <div className="flex items-center gap-1.5 text-green-400 text-sm">
                          <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                          </svg>
                          <span className="font-semibold truncate">
                            M{match_info?.mag?.toFixed(1) || closestEq?.mag?.toFixed(1)} - {match_info?.place || closestEq?.place}
                          </span>
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* Stats Row - Inline */}
                <div className="grid grid-cols-4 gap-2">
                  <div className="bg-zinc-800/50 border border-zinc-700 rounded p-2 text-center">
                    <div className="text-lg font-bold text-green-500">{predictionCounts.matched}</div>
                    <div className="text-zinc-500 text-[9px]">Matched</div>
                  </div>
                  <div className="bg-zinc-800/50 border border-zinc-700 rounded p-2 text-center">
                    <div className="text-lg font-bold text-yellow-500">{predictionCounts.pending}</div>
                    <div className="text-zinc-500 text-[9px]">Pending</div>
                  </div>
                  <div className="bg-zinc-800/50 border border-zinc-700 rounded p-2 text-center">
                    <div className="text-lg font-bold text-red-500">{predictionCounts.missed}</div>
                    <div className="text-zinc-500 text-[9px]">Missed</div>
                  </div>
                  <div className="bg-zinc-800/50 border border-zinc-700 rounded p-2 text-center">
                    <div className={`text-lg font-bold ${predictionCounts.successRate > 50 ? 'text-green-500' : predictionCounts.successRate > 0 ? 'text-orange-500' : 'text-zinc-500'}`}>
                      {predictionCounts.successRate.toFixed(0)}%
                    </div>
                    <div className="text-zinc-500 text-[9px]">Success</div>
                  </div>
                </div>

              </>
            ) : (
              <div className="card !p-6 text-center">
                <svg className="w-10 h-10 mx-auto text-zinc-600 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
                <h3 className="text-base font-semibold text-zinc-400">No Active Prediction</h3>
                <p className="text-zinc-500 text-xs">Waiting for new prediction...</p>
              </div>
            )}
          </div>

          {/* Right: Earthquakes in Prediction Window */}
          <div className="card !p-3">
            <div className="mb-2">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold text-white flex items-center gap-1.5">
                  <svg className="w-4 h-4 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                  Quakes in Window
                </h3>
                <select
                  value={minMagFilter}
                  onChange={(e) => setMinMagFilter(Number(e.target.value))}
                  className="bg-zinc-800 text-zinc-300 text-[10px] px-1.5 py-0.5 rounded border border-zinc-700 focus:outline-none focus:border-orange-500"
                >
                  <option value={0}>All</option>
                  <option value={2}>M2+</option>
                  <option value={3}>M3+</option>
                  <option value={4}>M4+</option>
                  <option value={5}>M5+</option>
                </select>
              </div>
              {/* Window time range from backend */}
              {prediction_status && (
                <div className="mt-1.5 text-[10px] text-zinc-500 flex items-center gap-1 flex-wrap">
                  <span className="text-green-400 font-mono">
                    {formatTimeLocal(prediction_status.start_time)?.split(',')[1]?.trim() || ''}
                  </span>
                  <span>→</span>
                  <span className="text-red-400 font-mono">
                    {formatTimeLocal(prediction_status.end_time)?.split(',')[1]?.trim() || ''}
                  </span>
                  <span className="text-zinc-600">({earthquakesInWindow.length})</span>
                </div>
              )}
            </div>

            <div className="space-y-1.5 max-h-[400px] overflow-y-auto custom-scrollbar pr-1">
              {earthquakesInWindow.length > 0 ? (
                earthquakesInWindow.map((eq, index) => (
                  <div
                    key={eq.id || index}
                    className={`p-2 rounded transition-all duration-300 ${getMagRowColor(eq.mag, eq.isMatch)} ${
                      eq.isMatch ? 'ring-1 ring-green-400 animate-pulse-matched' : ''
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-1.5">
                          <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                            eq.mag >= 6 ? 'bg-red-500 text-white' :
                            eq.mag >= 5 ? 'bg-orange-500 text-white' : 'bg-yellow-500 text-black'
                          }`}>
                            M{eq.mag?.toFixed(1)}
                          </span>
                          <span className="text-zinc-400 text-[10px]">{formatTimeAgo(eq.time)}</span>
                          {eq.isMatch && (
                            <span className="px-1 py-0.5 bg-green-500 text-white text-[9px] rounded-full">Match</span>
                          )}
                        </div>
                        <p className="text-white text-xs mt-0.5 truncate">{eq.place || 'Unknown'}</p>
                      </div>
                      {eq.distance !== null && (
                        <div className={`text-xs font-mono ${
                          eq.isMatch ? 'text-green-400' :
                          eq.distance < 30 ? 'text-yellow-400' : 'text-zinc-500'
                        }`}>
                          {eq.distance.toFixed(0)}°
                        </div>
                      )}
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-center py-4 text-zinc-500">
                  <svg className="w-8 h-8 mx-auto mb-1 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2.586a1 1 0 00-.707.293l-2.414 2.414a1 1 0 01-.707.293h-3.172a1 1 0 01-.707-.293l-2.414-2.414A1 1 0 006.586 13H4" />
                  </svg>
                  <p className="text-xs">No quakes yet</p>
                </div>
              )}
            </div>
          </div>
        </div>

      </div>
    </section>
  );
}

export default LiveDashboard;
