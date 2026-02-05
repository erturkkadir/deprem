import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';

const API_BASE = import.meta.env.VITE_API_URL || '';

function StatusBadge({ verified, correct }) {
  if (!verified) {
    return (
      <span className="inline-flex items-center gap-1 px-3 py-1.5 rounded-full text-sm bg-yellow-900/30 text-yellow-400 border border-yellow-600">
        <span className="w-2 h-2 rounded-full bg-yellow-400 animate-pulse" />
        Pending
      </span>
    );
  }

  if (correct) {
    return (
      <span className="inline-flex items-center gap-1 px-3 py-1.5 rounded-full text-sm bg-green-900/30 text-green-400 border border-green-600">
        <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
        </svg>
        Matched
      </span>
    );
  }

  return (
    <span className="inline-flex items-center gap-1 px-3 py-1.5 rounded-full text-sm bg-red-900/30 text-red-400 border border-red-600">
      <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
        <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
      </svg>
      Missed
    </span>
  );
}

function formatTime(timeStr) {
  if (!timeStr) return '—';
  try {
    const date = new Date(timeStr);
    if (isNaN(date.getTime())) return '—';
    return date.toLocaleString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
  } catch {
    return '—';
  }
}

function formatMinutes(minutes) {
  if (minutes === null || minutes === undefined) return '—';
  const absMin = Math.abs(minutes);
  const sign = minutes < 0 ? '-' : '+';
  if (absMin < 60) return `${sign}${absMin}m`;
  if (absMin < 1440) return `${sign}${(absMin / 60).toFixed(1)}h`;
  return `${sign}${(absMin / 1440).toFixed(1)}d`;
}

function calcMinutesAfter(predictionTime, actualTime) {
  if (!predictionTime || !actualTime) return null;
  try {
    const predDate = new Date(predictionTime);
    const actualDate = new Date(actualTime);
    if (isNaN(predDate.getTime()) || isNaN(actualDate.getTime())) return null;
    return Math.round((actualDate - predDate) / 60000);
  } catch {
    return null;
  }
}

export default function PredictionDetail() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filterMag, setFilterMag] = useState(2.0);
  const [showWindowOnly, setShowWindowOnly] = useState(true); // Default to show only window earthquakes

  useEffect(() => {
    async function fetchDetail() {
      try {
        setLoading(true);
        const response = await fetch(`${API_BASE}/api/prediction/${id}`);
        const result = await response.json();

        if (result.success) {
          setData(result);
        } else {
          setError(result.error || 'Failed to load prediction');
        }
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    }

    fetchDetail();
  }, [id]);

  if (loading) {
    return (
      <div className="min-h-screen bg-zinc-900 flex items-center justify-center">
        <div className="text-center">
          <div className="w-8 h-8 border-2 border-orange-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-zinc-400">Loading prediction details...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-zinc-900 flex items-center justify-center">
        <div className="text-center">
          <p className="text-red-400 mb-4">{error}</p>
          <button
            onClick={() => navigate('/')}
            className="px-4 py-2 bg-zinc-800 text-zinc-300 rounded-lg hover:bg-zinc-700"
          >
            Back to Home
          </button>
        </div>
      </div>
    );
  }

  const { prediction, earthquakes, earthquake_count, window_earthquakes, matching_earthquakes } = data;

  // Filter earthquakes by magnitude and optionally by window
  const filteredEarthquakes = earthquakes.filter(eq => {
    if (eq.mag < filterMag) return false;
    if (showWindowOnly && !eq.in_window) return false;
    return true;
  });

  return (
    <div className="min-h-screen bg-zinc-900">
      {/* Header */}
      <header className="bg-zinc-900 border-b border-zinc-800 py-4 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 flex items-center justify-between">
          <button
            onClick={() => navigate('/')}
            className="flex items-center gap-2 text-zinc-400 hover:text-white transition-colors"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
            Back
          </button>
          <h1 className="text-lg font-bold text-orange-500">Prediction #{prediction.id}</h1>
          <StatusBadge verified={prediction.verified} correct={prediction.correct} />
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-6">
        {/* Prediction Summary Card */}
        <div className="bg-zinc-800/50 rounded-xl p-6 mb-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Predicted Values */}
            <div>
              <h3 className="text-sm font-medium text-orange-400 mb-3 flex items-center gap-2">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
                Predicted
              </h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-zinc-500">Location</span>
                  <span className="text-white font-mono">
                    {prediction.predicted_lat?.toFixed(1)}°, {prediction.predicted_lon?.toFixed(1)}°
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-zinc-500">Place</span>
                  <span className="text-white truncate ml-4">{prediction.predicted_place || '—'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-zinc-500">Magnitude</span>
                  <span className="text-white font-mono">M{prediction.predicted_mag?.toFixed(1)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-zinc-500">Time Window</span>
                  <span className="text-white font-mono">{formatMinutes(prediction.predicted_dt)}</span>
                </div>
              </div>
            </div>

            {/* Actual Values (if verified) */}
            <div>
              <h3 className="text-sm font-medium text-green-400 mb-3 flex items-center gap-2">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
                Actual {prediction.verified ? '' : '(Pending)'}
              </h3>
              {prediction.verified && prediction.correct ? (
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-zinc-500">Location</span>
                    <span className="text-white font-mono">
                      {prediction.actual_lat?.toFixed(1)}°, {prediction.actual_lon?.toFixed(1)}°
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-zinc-500">Place</span>
                    <span className="text-white truncate ml-4">{prediction.actual_place || '—'}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-zinc-500">Magnitude</span>
                    <span className="text-white font-mono">M{prediction.actual_mag?.toFixed(1)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-zinc-500">Occurred After</span>
                    <span className="text-white font-mono">
                      {formatMinutes(calcMinutesAfter(prediction.prediction_time, prediction.actual_time))}
                    </span>
                  </div>
                </div>
              ) : (
                <p className="text-zinc-500 text-sm">
                  {prediction.verified ? 'No matching earthquake found' : 'Waiting for verification...'}
                </p>
              )}
            </div>
          </div>

          {/* Time Window Info - Clear comparison */}
          <div className="mt-6 pt-4 border-t border-zinc-700">
            <h4 className="text-xs text-zinc-500 uppercase tracking-wide mb-3">Time Comparison</h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
              <div className="bg-zinc-900/50 rounded-lg p-3">
                <div className="text-[10px] text-orange-400 uppercase mb-1">Prediction Made</div>
                <div className="text-white font-mono text-sm">{formatTime(prediction.prediction_time)}</div>
              </div>
              <div className="bg-zinc-900/50 rounded-lg p-3">
                <div className="text-[10px] text-purple-400 uppercase mb-1">Window End ({prediction.predicted_dt || 60}m later)</div>
                <div className="text-white font-mono text-sm">{formatTime(prediction.window_end)}</div>
              </div>
              {prediction.actual_time && (
                <div className={`rounded-lg p-3 ${
                  calcMinutesAfter(prediction.prediction_time, prediction.actual_time) < 0
                    ? 'bg-red-900/30 border border-red-600'
                    : 'bg-green-900/30 border border-green-600'
                }`}>
                  <div className={`text-[10px] uppercase mb-1 ${
                    calcMinutesAfter(prediction.prediction_time, prediction.actual_time) < 0
                      ? 'text-red-400'
                      : 'text-green-400'
                  }`}>
                    Match Time {calcMinutesAfter(prediction.prediction_time, prediction.actual_time) < 0 ? '(BEFORE prediction!)' : ''}
                  </div>
                  <div className={`font-mono text-sm ${
                    calcMinutesAfter(prediction.prediction_time, prediction.actual_time) < 0
                      ? 'text-red-300'
                      : 'text-green-300'
                  }`}>{formatTime(prediction.actual_time)}</div>
                  <div className={`text-xs mt-1 ${
                    calcMinutesAfter(prediction.prediction_time, prediction.actual_time) < 0
                      ? 'text-red-400'
                      : 'text-green-400'
                  }`}>
                    {formatMinutes(calcMinutesAfter(prediction.prediction_time, prediction.actual_time))} from prediction
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Earthquakes in Window */}
        <div className="bg-zinc-800/50 rounded-xl p-6">
          <div className="flex flex-wrap items-center justify-between gap-4 mb-4">
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              <svg className="w-5 h-5 text-orange-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
              Earthquakes {showWindowOnly ? '(in prediction window)' : '(24h after prediction)'}
            </h2>
            <div className="flex flex-wrap items-center gap-4">
              <div className="flex items-center gap-2 text-xs">
                <span className="px-2 py-1 bg-purple-900/30 text-purple-400 rounded">
                  {window_earthquakes} in window
                </span>
                <span className="px-2 py-1 bg-green-900/30 text-green-400 rounded">
                  {matching_earthquakes} matches
                </span>
                <span className="px-2 py-1 bg-zinc-700 text-zinc-400 rounded">
                  {earthquake_count} total (24h)
                </span>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setShowWindowOnly(!showWindowOnly)}
                  className={`px-2 py-1 text-xs rounded border transition-colors ${
                    showWindowOnly
                      ? 'bg-purple-900/30 text-purple-400 border-purple-600'
                      : 'bg-zinc-700 text-zinc-400 border-zinc-600'
                  }`}
                >
                  {showWindowOnly ? 'Window Only' : 'Show 24h'}
                </button>
              </div>
              <div className="flex items-center gap-2">
                <label className="text-zinc-500 text-xs">Min Mag:</label>
                <select
                  value={filterMag}
                  onChange={(e) => setFilterMag(parseFloat(e.target.value))}
                  className="bg-zinc-700 text-white text-xs rounded px-2 py-1 border border-zinc-600"
                >
                  <option value={2.0}>2.0+</option>
                  <option value={3.0}>3.0+</option>
                  <option value={4.0}>4.0+</option>
                  <option value={5.0}>5.0+</option>
                </select>
              </div>
            </div>
          </div>

          {filteredEarthquakes.length === 0 ? (
            <div className="text-center py-8 text-zinc-500">
              <p>No earthquakes M{filterMag}+ in this time window</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-zinc-500 border-b border-zinc-700">
                    <th className="pb-2 pr-4">Time</th>
                    <th className="pb-2 pr-4">After</th>
                    <th className="pb-2 pr-4">Location</th>
                    <th className="pb-2 pr-4">Mag</th>
                    <th className="pb-2 pr-4">Distance</th>
                    <th className="pb-2">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredEarthquakes.map((eq, idx) => (
                    <tr
                      key={eq.id || idx}
                      className={`border-b border-zinc-800 ${
                        eq.is_match
                          ? 'bg-green-900/20'
                          : eq.in_window
                          ? 'bg-purple-900/10'
                          : ''
                      }`}
                    >
                      <td className="py-2 pr-4 text-zinc-300 whitespace-nowrap">
                        {formatTime(eq.time)}
                      </td>
                      <td className="py-2 pr-4 font-mono">
                        <span className={eq.in_window ? 'text-purple-400' : 'text-zinc-500'}>
                          +{formatMinutes(eq.minutes_after)}
                        </span>
                      </td>
                      <td className="py-2 pr-4 text-zinc-300 max-w-[200px] truncate" title={eq.place}>
                        {eq.place || `${eq.lat?.toFixed(1)}°, ${eq.lon?.toFixed(1)}°`}
                      </td>
                      <td className="py-2 pr-4">
                        <span className={`font-mono font-bold ${
                          eq.mag >= 5 ? 'text-red-400' : eq.mag >= 4 ? 'text-orange-400' : 'text-zinc-400'
                        }`}>
                          M{eq.mag?.toFixed(1)}
                        </span>
                      </td>
                      <td className="py-2 pr-4 font-mono text-zinc-400">
                        {eq.distance !== null ? `${eq.distance}km` : '—'}
                      </td>
                      <td className="py-2">
                        {eq.is_match ? (
                          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs bg-green-900/30 text-green-400">
                            <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                              <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                            </svg>
                            Match
                          </span>
                        ) : eq.in_window ? (
                          <span className="px-2 py-0.5 rounded text-xs bg-purple-900/30 text-purple-400">
                            In Window
                          </span>
                        ) : (
                          <span className="text-zinc-600 text-xs">Outside</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
