import { useEffect, useState, useMemo, useRef } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { fetchLiveData } from '../store/earthquakeSlice';

function LiveDashboard() {
  const dispatch = useDispatch();
  const { liveData, isLoadingLive, isPredicting } = useSelector((state) => state.earthquake);
  const [currentTime, setCurrentTime] = useState(new Date());
  const [minMagFilter, setMinMagFilter] = useState(2); // Default M2.0+
  const [isFlashing, setIsFlashing] = useState(false);
  const lastPredictionId = useRef(null);
  const lastMatchState = useRef(false);

  // Update current time every second
  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
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

  const { latest_prediction, recent_earthquakes, stats, match_info, closest_match } = liveData || {};

  // Flash when a new match is detected (from backend)
  useEffect(() => {
    const hasMatch = match_info?.is_match || match_info?.verified_match;
    if (hasMatch && !lastMatchState.current) {
      // New match detected - trigger flash
      setIsFlashing(true);
      lastMatchState.current = true;
      setTimeout(() => setIsFlashing(false), 3000);
    }
  }, [match_info]);

  // Parse UTC time
  const parseUTC = (isoString) => {
    if (!isoString) return null;
    return new Date(isoString.endsWith('Z') ? isoString : isoString + 'Z');
  };

  // Calculate expected event time
  const expectedEventTime = useMemo(() => {
    if (!latest_prediction?.timestamp || !latest_prediction?.predicted_dt) return null;
    const predTime = parseUTC(latest_prediction.timestamp);
    return predTime ? new Date(predTime.getTime() + latest_prediction.predicted_dt * 60 * 1000) : null;
  }, [latest_prediction]);

  // Countdown calculation
  const countdownInfo = useMemo(() => {
    if (!expectedEventTime) return null;
    const diff = expectedEventTime.getTime() - currentTime.getTime();

    if (diff <= 0) return { expired: true, text: 'Window passed', progress: 100, seconds: 0 };

    const totalSeconds = Math.floor(diff / 1000);
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;

    // Calculate progress (0-100) based on predicted_dt
    const totalPredictedSeconds = (latest_prediction?.predicted_dt || 60) * 60;
    const elapsed = totalPredictedSeconds - totalSeconds;
    const progress = Math.min(100, Math.max(0, (elapsed / totalPredictedSeconds) * 100));

    const text = hours > 0 ? `${hours}h ${minutes}m ${seconds}s` :
                 minutes > 0 ? `${minutes}m ${seconds}s` : `${seconds}s`;

    return { expired: false, text, progress, seconds: totalSeconds };
  }, [expectedEventTime, currentTime, latest_prediction?.predicted_dt]);

  // Filter earthquakes by magnitude (distance already calculated by backend)
  const earthquakesWithDistance = useMemo(() => {
    if (!recent_earthquakes) return [];

    return recent_earthquakes
      .filter(eq => eq.mag >= minMagFilter)
      .map(eq => ({
        ...eq,
        // Use API-provided values, with fallbacks
        distance: eq.distance ?? null,
        isMatch: eq.is_match ?? false
      }))
      .sort((a, b) => new Date(b.time) - new Date(a.time));
  }, [recent_earthquakes, minMagFilter]);

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
  const bestMatch = match_info?.is_match ? earthquakesWithDistance.find(eq => eq.id === match_info.earthquake_id) : null;
  const closestEq = closest_match ? earthquakesWithDistance.find(eq => eq.id === closest_match.earthquake_id) : earthquakesWithDistance[0];

  // Format functions
  const formatTimeUTC = (date) => date?.toLocaleString('en-US', {
    month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit', timeZone: 'UTC'
  }) + ' UTC';

  const formatTimeLocal = (date) => date?.toLocaleString('en-US', {
    month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit'
  });

  const formatTimeAgo = (isoString) => {
    const date = parseUTC(isoString);
    if (!date) return '';
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
    <section className="py-8 bg-zinc-900">
      <div className="max-w-7xl mx-auto px-4">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-2xl font-bold text-white flex items-center gap-3">
              <div className="relative">
                <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse" />
                <div className="absolute inset-0 w-3 h-3 bg-green-500 rounded-full animate-ping" />
              </div>
              Live Prediction Monitor
            </h2>
            <p className="text-zinc-400 text-sm mt-1">
              Real-time earthquake prediction tracking • Updates every minute
            </p>
          </div>
          <div className="text-right">
            <div className="text-zinc-500 text-xs">Current Time (UTC)</div>
            <div className="text-white font-mono">{formatTimeUTC(currentTime)}</div>
          </div>
        </div>

        {/* Main Grid */}
        <div className="grid lg:grid-cols-3 gap-6">

          {/* Left: Prediction Card */}
          <div className="lg:col-span-2 space-y-4">
            {latest_prediction ? (
              <>
                {/* Location Header */}
                <div className={`rounded-xl p-6 border transition-all duration-500 ${
                  match_info?.is_match || match_info?.verified_match
                    ? 'bg-gradient-to-r from-green-600/20 to-green-500/10 border-green-500/50'
                    : 'bg-gradient-to-r from-orange-600/20 to-orange-500/10 border-orange-500/30'
                } ${isFlashing ? 'animate-flash-match' : ''}`}>
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="flex items-center gap-2 mb-2">
                        <span className={`px-3 py-1 rounded-full text-xs font-semibold ${
                          match_info?.is_match || match_info?.verified_match
                            ? 'bg-green-500 text-white animate-pulse'
                            : latest_prediction.verified
                              ? (latest_prediction.correct ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400')
                              : 'bg-yellow-500/20 text-yellow-400'
                        }`}>
                          {match_info?.is_match || match_info?.verified_match
                            ? '✓ MATCH FOUND!'
                            : latest_prediction.verified
                              ? (latest_prediction.correct ? 'Verified Correct' : 'Verified Incorrect')
                              : 'Awaiting Verification'}
                        </span>
                        {match_info?.verified_match && (
                          <span className="px-2 py-1 rounded-full text-xs bg-green-500/20 text-green-400">
                            ✓ Recorded
                          </span>
                        )}
                        <span className="text-zinc-500 text-xs">ID: #{latest_prediction.id}</span>
                      </div>
                      <h3 className="text-2xl font-bold text-white flex items-center gap-2">
                        <svg className="w-6 h-6 text-orange-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                        </svg>
                        {latest_prediction.predicted_place || 'Ocean Region'}
                      </h3>
                      <p className="text-zinc-400 text-sm mt-1">
                        {latest_prediction.predicted_lat?.toFixed(2)}°, {latest_prediction.predicted_lon?.toFixed(2)}°
                      </p>
                    </div>
                    <a
                      href={`https://www.google.com/maps?q=${latest_prediction.predicted_lat},${latest_prediction.predicted_lon}&z=5`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-2 px-4 py-2 bg-orange-500 hover:bg-orange-600 text-white rounded-lg transition-colors"
                    >
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
                      </svg>
                      View Map
                    </a>
                  </div>
                </div>

                {/* Countdown & Stats Grid */}
                <div className="grid md:grid-cols-2 gap-4">
                  {/* Countdown Timer */}
                  <div className="card p-6 flex items-center justify-center">
                    <div className="text-center">
                      <CircularProgress progress={countdownInfo?.progress || 0} size={140}>
                        <div className="text-center">
                          {countdownInfo?.expired ? (
                            isPredicting ? (
                              <div className="text-orange-400">
                                <svg className="w-8 h-8 mx-auto animate-spin" fill="none" viewBox="0 0 24 24">
                                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                                </svg>
                                <div className="text-xs mt-1">Loading...</div>
                              </div>
                            ) : (
                              <div className="text-zinc-400">
                                <svg className="w-8 h-8 mx-auto" fill="currentColor" viewBox="0 0 20 20">
                                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z" clipRule="evenodd" />
                                </svg>
                                <div className="text-xs mt-1">Refreshing...</div>
                              </div>
                            )
                          ) : (
                            <>
                              <div className={`text-2xl font-bold font-mono ${
                                countdownInfo?.progress > 80 ? 'text-red-400' :
                                countdownInfo?.progress > 60 ? 'text-yellow-400' : 'text-green-400'
                              }`}>
                                {countdownInfo?.text}
                              </div>
                              <div className="text-zinc-500 text-xs">remaining</div>
                            </>
                          )}
                        </div>
                      </CircularProgress>
                      <div className="mt-4 space-y-1">
                        <div className="text-zinc-400 text-xs">Expected Event</div>
                        <div className="text-white text-sm font-medium">
                          {expectedEventTime ? formatTimeUTC(expectedEventTime) : 'N/A'}
                        </div>
                        <div className="text-zinc-500 text-xs">
                          Local: {expectedEventTime ? formatTimeLocal(expectedEventTime) : 'N/A'}
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Prediction Details */}
                  <div className="card p-6">
                    <h4 className="text-zinc-400 text-sm mb-4">Prediction Details</h4>
                    <div className="space-y-4">
                      <div className="flex justify-between items-center">
                        <span className="text-zinc-500">Latitude</span>
                        <span className="text-xl font-bold text-orange-500">{latest_prediction.predicted_lat?.toFixed(1)}°</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-zinc-500">Longitude</span>
                        <span className="text-xl font-bold text-blue-500">{latest_prediction.predicted_lon?.toFixed(1)}°</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-zinc-500">Magnitude</span>
                        <span className={`text-xl font-bold ${
                          latest_prediction.predicted_mag >= 6 ? 'text-red-500' :
                          latest_prediction.predicted_mag >= 5 ? 'text-orange-500' : 'text-yellow-500'
                        }`}>
                          M{latest_prediction.predicted_mag?.toFixed(1)}
                        </span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-zinc-500">Time Window</span>
                        <span className="text-xl font-bold text-purple-500">{latest_prediction.predicted_dt || 60} min</span>
                      </div>
                      <div className="pt-2 border-t border-zinc-700">
                        <div className="text-zinc-500 text-xs">Predicted at</div>
                        <div className="text-zinc-300 text-sm">{formatTimeUTC(parseUTC(latest_prediction.timestamp))}</div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Closest Match Indicator */}
                {closestEq && !latest_prediction.verified && (
                  <div className="card p-4">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-zinc-400 text-sm">Closest Earthquake to Prediction</span>
                      <span className={`text-xs px-2 py-1 rounded ${
                        closestEq.isMatch ? 'bg-green-500/20 text-green-400' : 'bg-zinc-700 text-zinc-400'
                      }`}>
                        {closestEq.isMatch ? 'Within Range!' : `${closestEq.distance.toFixed(1)}° away`}
                      </span>
                    </div>
                    <MatchIndicator distance={closestEq.distance} />
                    {(closestEq?.isMatch || match_info?.is_match) && (
                      <div className="mt-3 p-3 bg-green-500/10 rounded-lg border border-green-500/30">
                        <div className="flex items-center gap-2 text-green-400">
                          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                          </svg>
                          <span className="font-semibold">
                            M{match_info?.mag?.toFixed(1) || closestEq?.mag?.toFixed(1)} - {match_info?.place || closestEq?.place}
                          </span>
                        </div>
                        <div className="text-green-300/70 text-xs mt-1">
                          {formatTimeAgo(closestEq?.time)} • Distance: {match_info?.distance || closestEq?.distance}°
                        </div>
                        <div className="mt-2 pt-2 border-t border-green-500/20 text-green-300 text-xs flex items-center gap-2">
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                          </svg>
                          New prediction starting automatically...
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* Visual Location Bars */}
                <div className="card p-4">
                  <div className="text-zinc-400 text-xs mb-3">Location Visualization</div>
                  <div className="space-y-3">
                    {/* Latitude */}
                    <div>
                      <div className="flex justify-between text-xs text-zinc-500 mb-1">
                        <span>-90°</span>
                        <span>Latitude</span>
                        <span>+90°</span>
                      </div>
                      <div className="relative h-4 bg-zinc-700 rounded-full overflow-hidden">
                        <div className="absolute top-0 bottom-0 w-0.5 bg-zinc-500" style={{ left: '50%' }} />
                        <div
                          className="absolute top-0.5 bottom-0.5 w-3 bg-orange-500 rounded-full shadow-lg shadow-orange-500/50"
                          style={{ left: `calc(${((latest_prediction.predicted_lat + 90) / 180) * 100}% - 6px)` }}
                        />
                        {(match_info?.is_match || match_info?.verified_match) && closestEq && (
                          <div
                            className="absolute top-0.5 bottom-0.5 w-3 bg-green-500 rounded-full animate-pulse shadow-lg shadow-green-500/50"
                            style={{ left: `calc(${((closestEq.lat + 90) / 180) * 100}% - 6px)` }}
                          />
                        )}
                      </div>
                    </div>
                    {/* Longitude */}
                    <div>
                      <div className="flex justify-between text-xs text-zinc-500 mb-1">
                        <span>-180°</span>
                        <span>Longitude</span>
                        <span>+180°</span>
                      </div>
                      <div className="relative h-4 bg-zinc-700 rounded-full overflow-hidden">
                        <div className="absolute top-0 bottom-0 w-0.5 bg-zinc-500" style={{ left: '50%' }} />
                        <div
                          className="absolute top-0.5 bottom-0.5 w-3 bg-blue-500 rounded-full shadow-lg shadow-blue-500/50"
                          style={{ left: `calc(${((latest_prediction.predicted_lon + 180) / 360) * 100}% - 6px)` }}
                        />
                        {(match_info?.is_match || match_info?.verified_match) && closestEq && (
                          <div
                            className="absolute top-0.5 bottom-0.5 w-3 bg-green-500 rounded-full animate-pulse shadow-lg shadow-green-500/50"
                            style={{ left: `calc(${((closestEq.lon + 180) / 360) * 100}% - 6px)` }}
                          />
                        )}
                      </div>
                    </div>
                  </div>
                  <div className="flex gap-6 mt-3 text-xs justify-center">
                    <span className="flex items-center gap-1"><div className="w-3 h-3 bg-orange-500 rounded-full" /> Predicted Lat</span>
                    <span className="flex items-center gap-1"><div className="w-3 h-3 bg-blue-500 rounded-full" /> Predicted Lon</span>
                    <span className="flex items-center gap-1"><div className="w-3 h-3 bg-green-500 rounded-full" /> Actual</span>
                  </div>
                </div>
              </>
            ) : (
              <div className="card p-12 text-center">
                <svg className="w-16 h-16 mx-auto text-zinc-600 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
                <h3 className="text-xl font-semibold text-zinc-400 mb-2">No Active Prediction</h3>
                <p className="text-zinc-500">Waiting for the system to generate a new prediction...</p>
              </div>
            )}
          </div>

          {/* Right: Recent Earthquakes */}
          <div className="card p-4">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                <svg className="w-5 h-5 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
                Recent Quakes
              </h3>
              <select
                value={minMagFilter}
                onChange={(e) => setMinMagFilter(Number(e.target.value))}
                className="bg-zinc-800 text-zinc-300 text-xs px-2 py-1 rounded border border-zinc-700 focus:outline-none focus:border-orange-500"
              >
                <option value={0}>All</option>
                <option value={2}>M2.0+</option>
                <option value={3}>M3.0+</option>
                <option value={4}>M4.0+</option>
                <option value={5}>M5.0+</option>
                <option value={6}>M6.0+</option>
              </select>
            </div>

            <div className="space-y-2 max-h-[600px] overflow-y-auto custom-scrollbar pr-1">
              {earthquakesWithDistance.length > 0 ? (
                earthquakesWithDistance.map((eq, index) => (
                  <div
                    key={eq.id || index}
                    className={`p-3 rounded-lg transition-all duration-300 ${getMagRowColor(eq.mag, eq.isMatch)} ${
                      eq.isMatch ? 'ring-2 ring-green-400 ring-offset-2 ring-offset-zinc-900 animate-pulse-matched' : ''
                    }`}
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 flex-wrap">
                          <span className={`px-2 py-0.5 rounded text-xs font-bold ${
                            eq.mag >= 6 ? 'bg-red-500 text-white' :
                            eq.mag >= 5 ? 'bg-orange-500 text-white' : 'bg-yellow-500 text-black'
                          }`}>
                            M{eq.mag?.toFixed(1)}
                          </span>
                          <span className="text-zinc-400 text-xs">{formatTimeAgo(eq.time)}</span>
                          {eq.isMatch && (
                            <span className="px-2 py-0.5 bg-green-500 text-white text-xs rounded-full animate-pulse">
                              Match!
                            </span>
                          )}
                        </div>
                        <p className="text-white text-sm mt-1 truncate">{eq.place || 'Unknown'}</p>
                        <div className="flex gap-3 mt-1 text-xs text-zinc-500">
                          <span>{eq.lat?.toFixed(1)}°</span>
                          <span>{eq.lon?.toFixed(1)}°</span>
                          <span>{eq.depth?.toFixed(0)}km</span>
                        </div>
                      </div>
                      {eq.distance !== null && (
                        <div className="text-right">
                          <div className={`text-xs font-mono ${
                            eq.isMatch ? 'text-green-400' :
                            eq.distance < 30 ? 'text-yellow-400' : 'text-zinc-500'
                          }`}>
                            {eq.distance.toFixed(0)}°
                          </div>
                          <div className="text-zinc-600 text-xs">dist</div>
                        </div>
                      )}
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-center py-8 text-zinc-500">
                  <svg className="w-12 h-12 mx-auto mb-2 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2.586a1 1 0 00-.707.293l-2.414 2.414a1 1 0 01-.707.293h-3.172a1 1 0 01-.707-.293l-2.414-2.414A1 1 0 006.586 13H4" />
                  </svg>
                  No recent data
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Stats Bar */}
        {stats && (
          <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="card p-4 text-center">
              <div className="text-3xl font-bold text-orange-500">{(parseFloat(stats.success_rate) || 0).toFixed(1)}%</div>
              <div className="text-zinc-500 text-sm">Success Rate</div>
            </div>
            <div className="card p-4 text-center">
              <div className="text-3xl font-bold text-white">{parseInt(stats.total_predictions) || 0}</div>
              <div className="text-zinc-500 text-sm">Total Predictions</div>
            </div>
            <div className="card p-4 text-center">
              <div className="text-3xl font-bold text-green-500">{parseInt(stats.correct_predictions) || 0}</div>
              <div className="text-zinc-500 text-sm">Correct</div>
            </div>
            <div className="card p-4 text-center">
              <div className="text-3xl font-bold text-zinc-400">{parseInt(stats.verified_predictions) || 0}</div>
              <div className="text-zinc-500 text-sm">Verified</div>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}

export default LiveDashboard;
