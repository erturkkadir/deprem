import { useEffect, useState, useMemo, useRef } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { fetchLiveData } from '../store/earthquakeSlice';

const MATCH_RADIUS_KM = 500;

function LiveDashboard() {
  const dispatch = useDispatch();
  const { liveData } = useSelector((state) => state.earthquake);
  const [minMagFilter, setMinMagFilter] = useState(2);
  const [isFlashing, setIsFlashing] = useState(false);
  const [, setTick] = useState(0);
  const lastMatchState = useRef(false);
  const lastPredId = useRef(null);

  // Tick every second for countdown
  useEffect(() => {
    const t = setInterval(() => setTick(n => n + 1), 1000);
    return () => clearInterval(t);
  }, []);

  const {
    latest_prediction: pred,
    recent_earthquakes,
    stats,
    match_info,
    closest_match,
    prediction_status,
  } = liveData || {};

  // Flash on new match
  useEffect(() => {
    if (pred?.id !== lastPredId.current) {
      lastPredId.current = pred?.id;
      lastMatchState.current = false;
      setIsFlashing(false);
    }
    const hasMatch = match_info?.is_match || match_info?.verified_match;
    if (hasMatch && !lastMatchState.current) {
      setIsFlashing(true);
      lastMatchState.current = true;
      setTimeout(() => setIsFlashing(false), 3000);
    }
  }, [pred?.id, match_info]);

  // ── helpers ────────────────────────────────────────────────────
  const fmtTime = (iso) => {
    if (!iso) return '—';
    return new Date(iso).toLocaleString('en-US', {
      month: 'short', day: 'numeric',
      hour: '2-digit', minute: '2-digit', second: '2-digit'
    });
  };

  const fmtTimeShort = (iso) => {
    if (!iso) return '—';
    return new Date(iso).toLocaleTimeString('en-US', {
      hour: '2-digit', minute: '2-digit', second: '2-digit'
    });
  };

  const fmtCountdown = (secs) => {
    if (secs == null) return '—';
    const abs = Math.abs(secs);
    const h = Math.floor(abs / 3600);
    const m = Math.floor((abs % 3600) / 60);
    const s = abs % 60;
    const str = h > 0 ? `${h}h ${m}m ${s}s` : m > 0 ? `${m}m ${s}s` : `${s}s`;
    return secs < 0 ? `Expired ${str} ago` : str;
  };

  const fmtAgo = (iso) => {
    if (!iso) return '';
    const diff = Math.floor((Date.now() - new Date(iso)) / 1000);
    if (diff < 60) return 'just now';
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    return `${Math.floor(diff / 86400)}d ago`;
  };

  const getMagColor = (mag) => {
    if (mag >= 7) return 'text-purple-400';
    if (mag >= 6) return 'text-red-400';
    if (mag >= 5) return 'text-orange-400';
    return 'text-yellow-400';
  };

  const getMagBg = (mag, isMatch) => {
    if (isMatch) return 'bg-green-500/20 border border-green-500/40';
    if (mag >= 7) return 'bg-purple-900/30 border border-purple-500/20';
    if (mag >= 6) return 'bg-red-900/30 border border-red-500/20';
    if (mag >= 5) return 'bg-orange-900/20 border border-orange-500/20';
    if (mag >= 4) return 'bg-yellow-900/10 border border-yellow-500/10';
    return 'bg-zinc-800/30 border border-transparent';
  };

  // Countdown state
  const remaining = prediction_status?.time_remaining_seconds;
  const isExpired = prediction_status?.is_expired;
  const countdownColor = isExpired ? 'text-red-400' :
    (remaining < 300 ? 'text-red-400' : remaining < 900 ? 'text-yellow-400' : 'text-green-400');

  // Status
  const isMatch = match_info?.is_match || match_info?.verified_match;
  const isVerified = pred?.verified;
  const isCorrect = pred?.correct;

  const statusLabel = isMatch || (isVerified && isCorrect) ? '✓ MATCHED' :
    isVerified ? 'Missed' : 'Pending';
  const statusClass = isMatch || (isVerified && isCorrect)
    ? 'bg-green-500 text-white animate-pulse'
    : isVerified
      ? 'bg-red-500/20 text-red-400'
      : 'bg-yellow-500/20 text-yellow-400';

  // Stats
  const counts = useMemo(() => {
    if (!stats) return { total: 0, matched: 0, pending: 0, missed: 0, rate: 0 };
    const total = parseInt(stats.total_predictions) || 0;
    const verified = parseInt(stats.verified_predictions) || 0;
    const matched = parseInt(stats.correct_predictions) || 0;
    return { total, matched, pending: total - verified, missed: verified - matched, rate: parseFloat(stats.success_rate) || 0 };
  }, [stats]);

  // Earthquakes in window
  const quakesInWindow = useMemo(() => {
    if (!recent_earthquakes || !prediction_status) return [];
    const s = new Date(prediction_status.start_time);
    const e = new Date(prediction_status.end_time);
    return recent_earthquakes
      .filter(q => q.mag >= minMagFilter && new Date(q.time) >= s && new Date(q.time) <= e)
      .map(q => ({ ...q, isMatch: q.is_match ?? false }))
      .sort((a, b) => new Date(b.time) - new Date(a.time));
  }, [recent_earthquakes, minMagFilter, prediction_status]);

  const mapUrl = (lat, lon) =>
    `/map.html?plat=${lat}&plon=${lon}&pmag=${pred?.predicted_mag || ''}&pdt=${pred?.predicted_dt || ''}&pplace=${encodeURIComponent(pred?.predicted_place || '')}&time=${encodeURIComponent(pred?.prediction_time || '')}&wend=${encodeURIComponent(pred?.window_end || '')}&id=${pred?.id || ''}&verified=${isVerified ? 'true' : 'false'}&correct=${isCorrect ? 'true' : 'false'}`;

  return (
    <section className="py-4 bg-zinc-900">
      <div className="max-w-7xl mx-auto px-4">

        {/* Header */}
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-base font-bold text-white flex items-center gap-2">
            <span className="relative flex h-2.5 w-2.5">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-green-500"></span>
            </span>
            Live Monitor
          </h2>
          <span className="text-zinc-500 text-xs">Updates every 10s</span>
        </div>

        <div className="grid lg:grid-cols-3 gap-3">

          {/* ── LEFT: Prediction + Candidates + Stats ── */}
          <div className="lg:col-span-2 space-y-3">

            {pred ? (
              <>
                {/* ── PREDICTION CARD ── */}
                <div className={`card !p-0 overflow-hidden transition-all duration-500 ${isMatch || (isVerified && isCorrect) ? 'ring-2 ring-green-500/40' : ''} ${isFlashing ? 'animate-flash-match' : ''}`}>

                  {/* Top bar: status + location + map btn */}
                  <div className={`px-4 py-3 flex items-start justify-between gap-3 ${isMatch || (isVerified && isCorrect) ? 'bg-green-600/15' : 'bg-orange-600/10'}`}>
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2 mb-1 flex-wrap">
                        <span className={`px-2 py-0.5 rounded-full text-[11px] font-bold ${statusClass}`}>
                          {statusLabel}
                        </span>
                        <span className="text-zinc-500 text-xs">#{pred.id}</span>
                        {!isVerified && (
                          <span className={`font-mono text-sm font-bold ${countdownColor}`}>
                            {isExpired ? fmtCountdown(remaining) : fmtCountdown(remaining)}
                          </span>
                        )}
                      </div>
                      <p className="text-white font-semibold text-sm truncate">
                        {pred.predicted_place || `${pred.predicted_lat?.toFixed(1)}°, ${pred.predicted_lon?.toFixed(1)}°`}
                      </p>
                    </div>
                    <a
                      href={mapUrl(pred.predicted_lat, pred.predicted_lon)}
                      target="_blank" rel="noopener noreferrer"
                      className="flex-shrink-0 flex items-center gap-1.5 px-3 py-1.5 bg-orange-500 hover:bg-orange-600 text-white rounded-lg transition-colors text-xs font-medium"
                    >
                      <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
                      </svg>
                      Map
                    </a>
                  </div>

                  {/* Details row */}
                  <div className="px-4 py-2.5 grid grid-cols-2 sm:grid-cols-5 gap-3 border-b border-zinc-800">
                    <div>
                      <div className="text-[9px] text-zinc-500 uppercase mb-0.5">Latitude</div>
                      <div className="text-orange-400 font-bold font-mono">{pred.predicted_lat?.toFixed(1)}°</div>
                    </div>
                    <div>
                      <div className="text-[9px] text-zinc-500 uppercase mb-0.5">Longitude</div>
                      <div className="text-blue-400 font-bold font-mono">{pred.predicted_lon?.toFixed(1)}°</div>
                    </div>
                    <div>
                      <div className="text-[9px] text-zinc-500 uppercase mb-0.5">Magnitude</div>
                      <div className={`font-bold ${getMagColor(pred.predicted_mag)}`}>M{pred.predicted_mag?.toFixed(1)}</div>
                    </div>
                    <div>
                      <div className="text-[9px] text-zinc-500 uppercase mb-0.5">Window</div>
                      <div className="text-purple-400 font-bold">{pred.predicted_dt || 120}m</div>
                    </div>
                    <div>
                      <div className="text-[9px] text-zinc-500 uppercase mb-0.5">Uncertainty</div>
                      <div className="text-zinc-300 font-bold font-mono">
                        {pred.sigma_km != null ? `±${Math.round(pred.sigma_km)} km` : '—'}
                      </div>
                    </div>
                  </div>

                  {/* Time window */}
                  <div className="px-4 py-2 flex items-center gap-3 text-[11px] flex-wrap">
                    <span className="text-zinc-500">Window:</span>
                    <span className="text-green-400 font-mono">{fmtTimeShort(prediction_status?.start_time)}</span>
                    <span className="text-zinc-600">→</span>
                    <span className="text-red-400 font-mono">{fmtTimeShort(prediction_status?.end_time)}</span>
                    <span className="text-zinc-600 text-[10px]">·</span>
                    <span className="text-zinc-400 font-mono">Now: {fmtTimeShort(prediction_status?.current_time)}</span>
                  </div>

                  {/* Matched earthquake info (when correct) */}
                  {(isMatch || (isVerified && isCorrect)) && (
                    <div className="px-4 pb-3">
                      <div className="bg-green-500/10 border border-green-500/30 rounded-lg px-3 py-2 flex items-center gap-2">
                        <svg className="w-4 h-4 text-green-400 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                        </svg>
                        <div className="text-sm">
                          <span className="text-green-400 font-semibold">
                            M{(match_info?.mag || pred.actual_mag)?.toFixed(1)}
                          </span>
                          <span className="text-zinc-300 ml-1">
                            {match_info?.place || '— earthquake matched'}
                          </span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                {/* ── STATS ROW ── */}
                <div className="grid grid-cols-4 gap-2">
                  {[
                    { val: counts.matched, label: 'Matched', color: 'text-green-500' },
                    { val: counts.pending, label: 'Pending', color: 'text-yellow-500' },
                    { val: counts.missed,  label: 'Missed',  color: 'text-red-500' },
                    { val: `${counts.rate.toFixed(0)}%`, label: 'Success', color: counts.rate > 30 ? 'text-green-500' : counts.rate > 0 ? 'text-orange-500' : 'text-zinc-500' },
                  ].map(({ val, label, color }) => (
                    <div key={label} className="bg-zinc-800/50 border border-zinc-700 rounded-lg p-2.5 text-center">
                      <div className={`text-xl font-bold ${color}`}>{val}</div>
                      <div className="text-zinc-500 text-[10px] mt-0.5">{label}</div>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <div className="card p-8 text-center">
                <div className="w-12 h-12 rounded-full bg-zinc-800 flex items-center justify-center mx-auto mb-3">
                  <svg className="w-6 h-6 text-zinc-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                  </svg>
                </div>
                <p className="text-zinc-400 text-sm font-medium">No active prediction</p>
                <p className="text-zinc-600 text-xs mt-1">Waiting for model...</p>
              </div>
            )}
          </div>

          {/* ── RIGHT: Quakes in Window ── */}
          <div className="card !p-3">
            <div className="flex items-center justify-between mb-2">
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

            {prediction_status && (
              <div className="text-[10px] text-zinc-500 mb-2 flex items-center gap-1 flex-wrap">
                <span className="text-green-400 font-mono">{fmtTimeShort(prediction_status.start_time)}</span>
                <span>→</span>
                <span className="text-red-400 font-mono">{fmtTimeShort(prediction_status.end_time)}</span>
                <span className="text-zinc-600">({quakesInWindow.length})</span>
              </div>
            )}

            <div className="space-y-1.5 max-h-[420px] overflow-y-auto custom-scrollbar pr-0.5">
              {quakesInWindow.length > 0 ? quakesInWindow.map((eq, i) => (
                <div key={eq.id || i} className={`p-2 rounded ${getMagBg(eq.mag, eq.isMatch)} ${eq.isMatch ? 'ring-1 ring-green-400' : ''}`}>
                  <div className="flex items-center justify-between gap-1">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-1.5 mb-0.5">
                        <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${eq.mag >= 6 ? 'bg-red-500 text-white' : eq.mag >= 5 ? 'bg-orange-500 text-white' : 'bg-yellow-500 text-black'}`}>
                          M{eq.mag?.toFixed(1)}
                        </span>
                        <span className="text-zinc-500 text-[10px]">{fmtAgo(eq.time)}</span>
                        {eq.isMatch && <span className="px-1 py-0.5 bg-green-500 text-white text-[9px] rounded-full font-bold">✓</span>}
                      </div>
                      <p className="text-white text-xs truncate">{eq.place || 'Unknown'}</p>
                      <div className="flex items-center gap-2 mt-0.5">
                        <span className="text-zinc-500 text-[10px] font-mono">{eq.lat?.toFixed(1)}°, {eq.lon?.toFixed(1)}°</span>
                        <a href={`/map.html?plat=${eq.lat}&plon=${eq.lon}&pmag=${eq.mag || ''}&pplace=${encodeURIComponent(eq.place || '')}`}
                          target="_blank" rel="noopener noreferrer"
                          className="text-blue-400 hover:text-blue-300 text-[10px]" onClick={e => e.stopPropagation()}>
                          map
                        </a>
                      </div>
                    </div>
                    {eq.distance != null && (
                      <div className={`text-xs font-mono flex-shrink-0 ${eq.isMatch ? 'text-green-400' : eq.distance < MATCH_RADIUS_KM ? 'text-yellow-400' : 'text-zinc-500'}`}>
                        {eq.distance.toFixed(0)}km
                      </div>
                    )}
                  </div>
                </div>
              )) : (
                <div className="text-center py-6 text-zinc-600">
                  <svg className="w-8 h-8 mx-auto mb-2 opacity-40" fill="none" stroke="currentColor" viewBox="0 0 24 24">
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
