import { useState } from 'react';
import { useSelector } from 'react-redux';

const FILTERS = [
  { key: 'all', label: 'All' },
  { key: 'pending', label: 'Pending' },
  { key: 'matched', label: 'Matched' },
  { key: 'missed', label: 'Missed' },
];

function getStatus(pred) {
  if (!pred.verified) {
    // If prediction date is before today (in local time), treat as missed
    if (pred.prediction_time) {
      const predTime = new Date(pred.prediction_time);
      const now = new Date();
      // Compare dates only (ignore time) in local timezone
      const predDate = new Date(predTime.getFullYear(), predTime.getMonth(), predTime.getDate());
      const todayDate = new Date(now.getFullYear(), now.getMonth(), now.getDate());
      if (predDate < todayDate) return 'missed';
    }
    return 'pending';
  }
  return pred.correct ? 'matched' : 'missed';
}

function FilterButton({ active, label, count, onClick, colorClass }) {
  return (
    <button
      onClick={onClick}
      className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
        active
          ? `${colorClass} ring-1 ring-current`
          : 'bg-zinc-800 text-zinc-400 hover:text-zinc-200'
      }`}
    >
      {label}
      {count !== undefined && (
        <span className={`ml-1.5 px-1.5 py-0.5 rounded text-xs ${
          active ? 'bg-black/20' : 'bg-zinc-700'
        }`}>
          {count}
        </span>
      )}
    </button>
  );
}

function StatusBadge({ verified, correct }) {
  if (!verified) {
    return (
      <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs bg-yellow-900/30 text-yellow-400 border border-yellow-600">
        <span className="w-1.5 h-1.5 rounded-full bg-yellow-400 animate-pulse" />
        Pending
      </span>
    );
  }

  if (correct) {
    return (
      <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs bg-green-900/30 text-green-400 border border-green-600">
        <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
        </svg>
        Matched
      </span>
    );
  }

  return (
    <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs bg-red-900/30 text-red-400 border border-red-600">
      <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
        <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
      </svg>
      Miss
    </span>
  );
}

function ParamCompare({ label, predicted, actual, diff, unit, tolerance, colorClass }) {
  const hasActual = actual !== null && actual !== undefined;
  const hasDiff = diff !== null && diff !== undefined;
  const isWithinTolerance = hasDiff && Math.abs(diff) <= tolerance;

  return (
    <div className="bg-zinc-800/50 rounded-lg p-3 min-w-[140px]">
      <div className={`text-xs font-medium mb-2 ${colorClass}`}>{label}</div>

      <div className="flex items-center justify-between gap-3">
        {/* Predicted */}
        <div className="text-center flex-1">
          <div className="text-[10px] text-zinc-500 uppercase tracking-wide">Pred</div>
          <div className={`font-mono font-bold text-lg ${colorClass}`}>
            {predicted !== null && predicted !== undefined
              ? `${Number(predicted).toFixed(1)}${unit}`
              : '—'}
          </div>
        </div>

        {/* Arrow or waiting */}
        <div className="text-zinc-600">
          {hasActual ? (
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
            </svg>
          ) : (
            <span className="text-xs">...</span>
          )}
        </div>

        {/* Actual */}
        <div className="text-center flex-1">
          <div className="text-[10px] text-zinc-500 uppercase tracking-wide">Actual</div>
          <div className="font-mono font-bold text-lg text-white">
            {hasActual ? `${Number(actual).toFixed(1)}${unit}` : '—'}
          </div>
        </div>
      </div>

      {/* Difference bar */}
      {hasDiff && (
        <div className={`mt-2 text-center text-xs font-medium px-2 py-1 rounded ${
          isWithinTolerance
            ? 'bg-green-900/30 text-green-400'
            : 'bg-red-900/30 text-red-400'
        }`}>
          {isWithinTolerance ? '✓' : '✗'} Diff: {diff > 0 ? '+' : ''}{Number(diff).toFixed(1)}{unit}
          <span className="text-zinc-500 ml-1">(±{tolerance}{unit})</span>
        </div>
      )}
    </div>
  );
}

export default function PredictionsTable() {
  const { predictions } = useSelector((state) => state.earthquake);
  const [filter, setFilter] = useState('all');

  const sortedPredictions = [...predictions].reverse();

  // Count by status
  const counts = {
    all: sortedPredictions.length,
    pending: sortedPredictions.filter(p => getStatus(p) === 'pending').length,
    matched: sortedPredictions.filter(p => getStatus(p) === 'matched').length,
    missed: sortedPredictions.filter(p => getStatus(p) === 'missed').length,
  };

  // Filter predictions
  const filteredPredictions = filter === 'all'
    ? sortedPredictions
    : sortedPredictions.filter(p => getStatus(p) === filter);

  const filterColors = {
    all: 'bg-orange-900/30 text-orange-400',
    pending: 'bg-yellow-900/30 text-yellow-400',
    matched: 'bg-green-900/30 text-green-400',
    missed: 'bg-red-900/30 text-red-400',
  };

  const formatTime = (timeStr) => {
    if (!timeStr) return '—';
    try {
      const date = new Date(timeStr);
      if (isNaN(date.getTime())) return '—';
      return date.toLocaleString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      });
    } catch {
      return '—';
    }
  };

  const getTimeAgo = (timeStr) => {
    if (!timeStr) return '';
    try {
      const date = new Date(timeStr);
      if (isNaN(date.getTime())) return '';
      const now = new Date();
      const diffMs = now - date;
      const diffMins = Math.floor(diffMs / 60000);
      const diffHours = Math.floor(diffMs / 3600000);
      const diffDays = Math.floor(diffMs / 86400000);

      if (diffMins < 1) return 'just now';
      if (diffMins < 60) return `${diffMins}m ago`;
      if (diffHours < 24) return `${diffHours}h ago`;
      return `${diffDays}d ago`;
    } catch {
      return '';
    }
  };

  return (
    <section className="py-8">
      <div className="max-w-7xl mx-auto px-4">
        <div className="flex flex-wrap items-center justify-between gap-4 mb-6">
          <h2 className="text-2xl font-bold text-orange-500">
            Recent Predictions
          </h2>
          <div className="flex gap-2">
            {FILTERS.map(f => (
              <FilterButton
                key={f.key}
                active={filter === f.key}
                label={f.label}
                count={counts[f.key]}
                onClick={() => setFilter(f.key)}
                colorClass={filterColors[f.key]}
              />
            ))}
          </div>
        </div>

        {filteredPredictions.length === 0 ? (
          <div className="card text-center py-12 text-zinc-500">
            <svg className="w-16 h-16 mx-auto mb-4 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
            <p>{filter === 'all' ? 'No predictions yet' : `No ${filter} predictions`}</p>
          </div>
        ) : (
          <div className="space-y-4">
            {filteredPredictions.map((pred, index) => (
              <div
                key={pred.id}
                className={`card p-4 ${index === 0 ? 'ring-2 ring-orange-500/50' : ''}`}
              >
                {/* Header row */}
                <div className="flex items-center justify-between mb-4 pb-3 border-b border-zinc-700">
                  <div className="flex items-center gap-4">
                    <span className="text-2xl font-bold text-orange-500">#{pred.id}</span>
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="text-white font-medium">{formatTime(pred.prediction_time)}</span>
                        <span className="text-xs text-zinc-500 bg-zinc-800 px-2 py-0.5 rounded">{getTimeAgo(pred.prediction_time)}</span>
                      </div>
                      <div className="flex items-center gap-2 text-sm">
                        <svg className="w-4 h-4 text-orange-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                        </svg>
                        <span className="text-orange-300 font-medium">
                          {pred.predicted_place || `Ocean (${pred.predicted_lat}°, ${pred.predicted_lon}°)`}
                        </span>
                      </div>
                      {pred.actual_place && (
                        <div className="text-zinc-400 text-sm mt-1">Actual: {pred.actual_place}</div>
                      )}
                    </div>
                  </div>
                  <StatusBadge verified={pred.verified} correct={pred.correct} />
                </div>

                {/* 4 Parameter Grid */}
                <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
                  <ParamCompare
                    label="LATITUDE"
                    predicted={pred.predicted_lat}
                    actual={pred.actual_lat}
                    diff={pred.diff_lat}
                    unit="°"
                    tolerance={10}
                    colorClass="text-orange-400"
                  />
                  <ParamCompare
                    label="LONGITUDE"
                    predicted={pred.predicted_lon}
                    actual={pred.actual_lon}
                    diff={pred.diff_lon}
                    unit="°"
                    tolerance={20}
                    colorClass="text-blue-400"
                  />
                  <ParamCompare
                    label="TIME DIFF"
                    predicted={pred.predicted_dt}
                    actual={pred.actual_dt}
                    diff={pred.diff_dt}
                    unit="m"
                    tolerance={30}
                    colorClass="text-purple-400"
                  />
                  <ParamCompare
                    label="MAGNITUDE"
                    predicted={pred.predicted_mag}
                    actual={pred.actual_mag}
                    diff={pred.diff_mag}
                    unit=""
                    tolerance={1.0}
                    colorClass="text-red-400"
                  />
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </section>
  );
}
