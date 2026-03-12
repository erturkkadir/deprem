import { useEffect, useState, useMemo, useRef } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { fetchLiveData } from '../store/earthquakeSlice';

const MATCH_RADIUS_KM = 250;

// ── Circular countdown clock ────────────────────────────────────────────────
function CountdownClock({ totalSecs, remainingSecs, isExpired }) {
  const cx = 50, cy = 50;
  const trackR = 30;
  const circ = 2 * Math.PI * trackR;

  const progress = isExpired
    ? 1
    : Math.max(0, Math.min(1, 1 - remainingSecs / totalSecs));
  const dashOffset = circ * (1 - progress);

  const color = isExpired || remainingSecs < 300
    ? '#ef4444'
    : remainingSecs < 900 ? '#eab308' : '#22c55e';

  const abs = Math.abs(remainingSecs || 0);
  const h = Math.floor(abs / 3600);
  const m = Math.floor((abs % 3600) / 60);
  const s = abs % 60;
  const timeStr = h > 0
    ? `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
    : `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;

  // 12 major tick marks
  const majorTicks = Array.from({ length: 12 }, (_, i) => {
    const a = (i / 12) * 2 * Math.PI - Math.PI / 2;
    return {
      x1: cx + 36 * Math.cos(a), y1: cy + 36 * Math.sin(a),
      x2: cx + 44 * Math.cos(a), y2: cy + 44 * Math.sin(a),
    };
  });

  // 60 minor tick marks (skip every 5th — already covered by major)
  const minorTicks = Array.from({ length: 60 }, (_, i) => {
    if (i % 5 === 0) return null;
    const a = (i / 60) * 2 * Math.PI - Math.PI / 2;
    return {
      x1: cx + 40 * Math.cos(a), y1: cy + 40 * Math.sin(a),
      x2: cx + 44 * Math.cos(a), y2: cy + 44 * Math.sin(a),
    };
  }).filter(Boolean);

  return (
    <div className="relative flex-shrink-0" style={{ width: 96, height: 96 }}>
      <svg viewBox="0 0 100 100" style={{ width: 96, height: 96, transform: 'rotate(-90deg)' }}>
        {/* Outer bezel circle */}
        <circle cx={cx} cy={cy} r={44} fill="#18181b" stroke="#3f3f46" strokeWidth="1" />

        {/* Minor ticks */}
        {minorTicks.map((t, i) => (
          <line key={i} x1={t.x1} y1={t.y1} x2={t.x2} y2={t.y2}
            stroke="#3f3f46" strokeWidth="1" strokeLinecap="round" />
        ))}

        {/* Major ticks */}
        {majorTicks.map((t, i) => (
          <line key={i} x1={t.x1} y1={t.y1} x2={t.x2} y2={t.y2}
            stroke="#71717a" strokeWidth="2" strokeLinecap="round" />
        ))}

        {/* Track ring */}
        <circle cx={cx} cy={cy} r={trackR} fill="none" stroke="#27272a" strokeWidth="8" />

        {/* Progress arc */}
        <circle
          cx={cx} cy={cy} r={trackR}
          fill="none"
          stroke={color}
          strokeWidth="8"
          strokeDasharray={circ}
          strokeDashoffset={dashOffset}
          strokeLinecap="round"
          style={{ transition: 'stroke-dashoffset 0.9s linear, stroke 0.5s ease' }}
        />

        {/* Glowing center dot */}
        <circle cx={cx} cy={cy} r={3} fill={color} opacity="0.8" />
      </svg>

      {/* Center text — unrotated overlay */}
      <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
        <span className="font-mono font-bold leading-none" style={{ fontSize: 11, color }}>
          {timeStr}
        </span>
        <span className="text-zinc-600 leading-none mt-1" style={{ fontSize: 8 }}>
          {isExpired ? 'expired' : 'left'}
        </span>
      </div>
    </div>
  );
}

// ── Main component ──────────────────────────────────────────────────────────
function LiveDashboard() {
  const dispatch = useDispatch();
  const { liveData } = useSelector((state) => state.earthquake);
  const [minMagFilter, setMinMagFilter] = useState(2);
  const [isFlashing, setIsFlashing] = useState(false);
  const [tick, setTick] = useState(0);
  const lastMatchState = useRef(false);
  const lastPredId = useRef(null);

  // Smooth countdown: track last server value + when we got it
  const serverRemRef = useRef(null);
  const serverRemAtRef = useRef(null);

  // Tick every second for smooth clock
  useEffect(() => {
    const t = setInterval(() => setTick(n => n + 1), 1000);
    return () => clearInterval(t);
  }, []);

  const {
    latest_prediction: pred,
    group_zones,
    recent_earthquakes,
    stats,
    match_info,
    prediction_status,
  } = liveData || {};

  // Update smooth countdown reference when server sends new value
  const serverRemaining = prediction_status?.time_remaining_seconds;
  useEffect(() => {
    if (serverRemaining != null) {
      serverRemRef.current = serverRemaining;
      serverRemAtRef.current = Date.now();
    }
  }, [serverRemaining]);

  // Compute locally-interpolated remaining (re-runs each tick)
  const smoothRemaining = (() => {
    if (serverRemRef.current == null) return serverRemaining ?? 0;
    const elapsed = (Date.now() - serverRemAtRef.current) / 1000;
    return Math.round(serverRemRef.current - elapsed);
  })();

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
  const fmtTimeShort = (iso) => {
    if (!iso) return '—';
    return new Date(iso).toLocaleTimeString('en-US', {
      hour: '2-digit', minute: '2-digit', second: '2-digit'
    });
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

  const getMagBg = (mag, inWindow, isMatch) => {
    if (isMatch) return 'bg-green-500/20 border border-green-500/40';
    if (inWindow && mag >= 4) return 'bg-orange-900/20 border border-orange-500/30';
    if (inWindow) return 'bg-zinc-800/60 border border-zinc-600/50';
    if (mag >= 7) return 'bg-purple-900/30 border border-purple-500/20';
    if (mag >= 6) return 'bg-red-900/30 border border-red-500/20';
    if (mag >= 5) return 'bg-orange-900/10 border border-orange-500/10';
    return 'bg-zinc-800/20 border border-zinc-800';
  };

  // Status
  const isMatch = match_info?.is_match || match_info?.verified_match;
  const isVerified = pred?.verified;
  const isCorrect = pred?.correct;
  const isExpired = prediction_status?.is_expired || smoothRemaining < 0;

  // A "Late Catch" = verified+correct but the actual event was after the window end
  const predWindowEnd = prediction_status?.end_time ? new Date(prediction_status.end_time) : null;
  const actualTime = pred?.actual_time ? new Date(pred.actual_time) : null;
  const isLateCatch = isVerified && isCorrect && actualTime && predWindowEnd && actualTime > predWindowEnd;

  const statusLabel = isLateCatch ? '⏱ LATE CATCH'
    : isMatch || (isVerified && isCorrect) ? '✓ MATCHED'
    : isVerified && isExpired ? 'Missed — checking 24h'
    : isVerified ? 'Missed'
    : isExpired ? 'Expired — next starting'
    : 'Active';
  const statusClass = isLateCatch
    ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30'
    : isMatch || (isVerified && isCorrect)
      ? 'bg-green-500 text-white animate-pulse'
      : isVerified
        ? 'bg-red-500/20 text-red-400'
        : isExpired
          ? 'bg-yellow-500/20 text-yellow-400'
          : 'bg-orange-500/20 text-orange-400';

  // Clock params
  const totalSecs = (pred?.predicted_dt || 60) * 60;

  // Stats
  const counts = useMemo(() => {
    if (!stats) return { total: 0, matched: 0, pending: 0, missed: 0, rate: 0 };
    const total = parseInt(stats.total_predictions) || 0;
    const verified = parseInt(stats.verified_predictions) || 0;
    const matched = parseInt(stats.correct_predictions) || 0;
    return { total, matched, pending: total - verified, missed: verified - matched, rate: parseFloat(stats.success_rate) || 0 };
  }, [stats]);

  // Window boundaries for badge marking
  const windowStart = prediction_status?.start_time ? new Date(prediction_status.start_time) : null;
  const windowEnd   = prediction_status?.end_time   ? new Date(prediction_status.end_time)   : null;

  // All recent earthquakes filtered by mag — mark in-window ones
  const recentQuakes = useMemo(() => {
    if (!recent_earthquakes) return [];
    return recent_earthquakes
      .filter(q => q.mag >= minMagFilter)
      .map(q => {
        const t = new Date(q.time);
        const inWindow = windowStart && windowEnd && t >= windowStart && t <= windowEnd;
        return { ...q, inWindow, isMatch: q.is_match ?? false };
      })
      .sort((a, b) => new Date(b.time) - new Date(a.time));
  }, [recent_earthquakes, minMagFilter, windowStart, windowEnd]);

  const inWindowCount = recentQuakes.filter(q => q.inWindow).length;

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

          {/* ── LEFT: Prediction + Stats ── */}
          <div className="lg:col-span-2 space-y-3">

            {pred ? (
              <>
                {/* ── PREDICTION CARD ── */}
                <div className={`card !p-0 overflow-hidden transition-all duration-500 ${isMatch || (isVerified && isCorrect) ? 'ring-2 ring-green-500/40' : ''} ${isFlashing ? 'animate-flash-match' : ''}`}>

                  {/* Top bar: status + location + clock + map */}
                  <div className={`px-4 py-3 flex items-center gap-4 ${isMatch || (isVerified && isCorrect) ? 'bg-green-600/15' : 'bg-orange-600/10'}`}>

                    {/* Left: status + place */}
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2 mb-1 flex-wrap">
                        <span className={`px-2 py-0.5 rounded-full text-[11px] font-bold ${statusClass}`}>
                          {statusLabel}
                        </span>
                        <span className="text-zinc-500 text-xs">#{pred.id}</span>
                      </div>
                      <p className="text-white font-semibold text-sm truncate">
                        {pred.predicted_place || `${pred.predicted_lat?.toFixed(1)}°, ${pred.predicted_lon?.toFixed(1)}°`}
                      </p>
                      <p className="text-zinc-500 text-[10px] mt-0.5 font-mono">
                        {windowStart ? `${fmtTimeShort(prediction_status.start_time)} → ${fmtTimeShort(prediction_status.end_time)}` : ''}
                      </p>
                    </div>

                    {/* Right: clock + map button */}
                    <div className="flex-shrink-0 flex flex-col items-center gap-1.5">
                      {!isVerified ? (
                        <CountdownClock
                          totalSecs={totalSecs}
                          remainingSecs={smoothRemaining}
                          isExpired={isExpired}
                        />
                      ) : (
                        <div className={`w-24 h-24 rounded-full flex items-center justify-center border-2 ${isCorrect ? 'border-green-500 bg-green-500/10' : 'border-red-500 bg-red-500/10'}`}>
                          <span className={`text-2xl font-bold ${isCorrect ? 'text-green-400' : 'text-red-400'}`}>
                            {isCorrect ? '✓' : '✗'}
                          </span>
                        </div>
                      )}
                      <a
                        href={mapUrl(pred.predicted_lat, pred.predicted_lon)}
                        target="_blank" rel="noopener noreferrer"
                        className="flex items-center gap-1 px-2.5 py-1 bg-orange-500 hover:bg-orange-600 text-white rounded-md transition-colors text-[11px] font-medium"
                      >
                        <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
                        </svg>
                        Map
                      </a>
                    </div>
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
                      <div className="text-purple-400 font-bold">
                        {(() => { const m = pred.predicted_dt || 60; return m >= 60 ? `${(m/60).toFixed(0)}h` : `${m}m`; })()}
                      </div>
                    </div>
                    <div>
                      <div className="text-[9px] text-zinc-500 uppercase mb-0.5">Uncertainty</div>
                      <div className="text-zinc-300 font-bold font-mono">
                        {pred.sigma_km != null ? `±${Math.round(pred.sigma_km)} km` : '—'}
                      </div>
                    </div>
                  </div>

                  {/* ── GROUP ZONES ── */}
                  {group_zones && group_zones.length > 1 && (
                    <div className="px-4 py-2.5 border-b border-zinc-800">
                      <div className="text-[9px] text-zinc-500 uppercase mb-1.5 tracking-wider">
                        Prediction Zones ({group_zones.length})
                      </div>
                      <div className="flex flex-wrap gap-1.5">
                        {group_zones.map((z) => (
                          <a
                            key={z.pr_id}
                            href={`/map.html?plat=${z.lat}&plon=${z.lon}&pmag=${pred?.predicted_mag || ''}&pdt=${pred?.predicted_dt || ''}&pplace=${encodeURIComponent(z.place || '')}&time=${encodeURIComponent(pred?.prediction_time || '')}&wend=${encodeURIComponent(pred?.window_end || '')}&id=${pred?.id || ''}`}
                            target="_blank" rel="noopener noreferrer"
                            className="flex items-center gap-1 px-2 py-0.5 rounded-full border border-zinc-700 bg-zinc-800/60 hover:border-orange-500/50 hover:bg-zinc-700/60 transition-colors cursor-pointer"
                            title={`Rank ${z.rank}: ${z.lat?.toFixed(1)}°, ${z.lon?.toFixed(1)}°${z.sigma_km ? ` ±${Math.round(z.sigma_km)}km` : ''}`}
                          >
                            <span className="text-orange-500 font-bold text-[9px]">#{z.rank}</span>
                            <span className="text-zinc-300 text-[10px] truncate max-w-[120px]">
                              {z.place
                                ? z.place.split(',')[0]
                                : `${z.lat?.toFixed(1)}°, ${z.lon?.toFixed(1)}°`}
                            </span>
                          </a>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Matched earthquake banner */}
                  {(isMatch || (isVerified && isCorrect)) && (
                    <div className="px-4 pb-3 pt-2">
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

                {/* ── HOW IT WORKS ── */}
                <div className="flex items-start gap-2 px-3 py-2 bg-zinc-800/30 rounded-lg border border-zinc-700/50 text-[10px] text-zinc-500">
                  <svg className="w-3 h-3 mt-0.5 flex-shrink-0 text-zinc-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <span>
                    Each cycle predicts <span className="text-zinc-400">1 zone</span> for the next <span className="text-zinc-400">90 min</span> within <span className="text-zinc-400">250 km</span>.
                    If no M4+ hits, the prediction is marked <span className="text-yellow-500">Missed</span> and a new cycle starts immediately.
                    Missed predictions continue checking for <span className="text-zinc-400">up to 48 h</span> — a late match counts as a <span className="text-green-500">Late Catch</span>.
                  </span>
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

          {/* ── RIGHT: Recent Earthquakes ── */}
          <div className="card !p-3">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-semibold text-white flex items-center gap-1.5">
                <svg className="w-4 h-4 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
                Recent Earthquakes
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

            {/* Window badge summary */}
            {inWindowCount > 0 && (
              <div className="text-[10px] text-orange-400 mb-2 flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-orange-400 inline-block"></span>
                {inWindowCount} in prediction window
              </div>
            )}

            <div className="space-y-1.5 max-h-[440px] overflow-y-auto custom-scrollbar pr-0.5">
              {recentQuakes.length > 0 ? recentQuakes.slice(0, 10).map((eq, i) => (
                <div key={eq.id || i} className={`p-2 rounded ${getMagBg(eq.mag, eq.inWindow, eq.isMatch)} ${eq.isMatch ? 'ring-1 ring-green-400' : ''}`}>
                  <div className="flex items-center justify-between gap-1">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-1.5 mb-0.5 flex-wrap">
                        <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${eq.mag >= 6 ? 'bg-red-500 text-white' : eq.mag >= 5 ? 'bg-orange-500 text-white' : 'bg-yellow-500/80 text-black'}`}>
                          M{eq.mag?.toFixed(1)}
                        </span>
                        <span className="text-zinc-500 text-[10px]">{fmtAgo(eq.time)}</span>
                        {eq.isMatch && (
                          <span className="px-1 py-0.5 bg-green-500 text-white text-[9px] rounded-full font-bold">✓ match</span>
                        )}
                        {eq.inWindow && !eq.isMatch && (
                          <span className="px-1 py-0.5 bg-orange-500/20 text-orange-300 border border-orange-500/30 text-[9px] rounded-full">window</span>
                        )}
                      </div>
                      <p className="text-white text-xs truncate">{eq.place || 'Unknown'}</p>
                      <div className="flex items-center gap-2 mt-0.5">
                        <span className="text-zinc-500 text-[10px] font-mono">{eq.lat?.toFixed(1)}°, {eq.lon?.toFixed(1)}°</span>
                        <a
                          href={`/map.html?plat=${eq.lat}&plon=${eq.lon}&pmag=${eq.mag || ''}&pplace=${encodeURIComponent(eq.place || '')}`}
                          target="_blank" rel="noopener noreferrer"
                          className="text-blue-400 hover:text-blue-300 text-[10px]"
                          onClick={e => e.stopPropagation()}
                        >
                          map
                        </a>
                      </div>
                    </div>
                    {eq.distance != null && (
                      <div className={`text-xs font-mono flex-shrink-0 ${eq.isMatch ? 'text-green-400' : eq.distance < MATCH_RADIUS_KM ? 'text-yellow-400' : 'text-zinc-600'}`}>
                        {eq.distance.toFixed(0)}km
                      </div>
                    )}
                  </div>
                </div>
              )) : (
                <div className="text-center py-8 text-zinc-600">
                  <svg className="w-8 h-8 mx-auto mb-2 opacity-30" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                  <p className="text-xs">No recent earthquakes</p>
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
